import csv
import logging
import urllib.request
import zipfile

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

from ml.paths import DATA_DIR, SEED
from ml.text import normalize

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SHADE_REPO = "Shade63/scam-classification-multiclass"
SHADE_FILE = "sentinel_dataset_multiclass.csv"
RAW_DIR = DATA_DIR / "raw"
KAGGLE_SYNTH = DATA_DIR / "external" / "synthetic_train_2737.csv"   # Kaggle india_fraud_detection, is_synthetic=True rows
REAL_TEST = DATA_DIR / "real_test_463.csv"                          # real messages: NEVER train on these

log = logging.getLogger("prepare_sms")


def load_uci() -> pd.DataFrame:
    archive = RAW_DIR / "uci_sms_spam.zip"
    if not archive.exists():
        urllib.request.urlretrieve(UCI_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.endswith("SMSSpamCollection"))
        with zf.open(member) as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["category", "text"],
                             quoting=csv.QUOTE_NONE, encoding_errors="replace")
    return pd.DataFrame({
        "text": df["text"].astype(str),
        "label": (df["category"] == "spam").astype(int),
        "category": df["category"].astype(str),
        "source": "uci",
    })


def load_shade() -> pd.DataFrame:
    files = HfApi().list_repo_files(SHADE_REPO, repo_type="dataset")
    name = next((f for f in files if f.endswith(SHADE_FILE)), None)
    if name is None:
        raise SystemExit(f"{SHADE_FILE} not found in {SHADE_REPO}: {files}")
    df = pd.read_csv(hf_hub_download(SHADE_REPO, name, repo_type="dataset", local_dir=RAW_DIR / "shade63"))
    category = df["category_name"] if "category_name" in df else df["label"].map({0: "legit", 1: "scam"})
    return pd.DataFrame({
        "text": df["text"].astype(str),
        "label": df["label"].astype(int),
        "category": category.astype(str),
        "source": "shade63",
    })


def load_kaggle_synth() -> pd.DataFrame:
    if not KAGGLE_SYNTH.exists():
        raise SystemExit(f"Missing {KAGGLE_SYNTH}. Copy synthetic_train_2737.csv there first.")
    df = pd.read_csv(KAGGLE_SYNTH)
    return pd.DataFrame({
        "text": df["text"].astype(str),
        "label": df["label"].astype(int),
        "category": df["category"].astype(str),
        "source": "kaggle_synth",
    })


def drop_real_test(df: pd.DataFrame) -> pd.DataFrame:
    """Remove anything that also appears in the real test set, so the real test stays unseen."""
    if not REAL_TEST.exists():
        log.warning("No %s found; skipping leak check", REAL_TEST)
        return df
    real = set(pd.read_csv(REAL_TEST)["text"].astype(str).map(normalize).str.lower())
    leaked = df["text"].str.lower().isin(real)
    log.info("Removed %d messages that are also in the real test set", int(leaked.sum()))
    return df[~leaked].reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(text=df["text"].map(normalize))
    df = df[df["text"].str.len() >= 3]
    duplicated = df["text"].str.lower().duplicated()
    log.info("Removed %d duplicates, %d messages remain", int(duplicated.sum()), int((~duplicated).sum()))
    return df[~duplicated].reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return {"train": train, "val": val, "test_public": test}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sources = [load_uci(), load_shade(), load_kaggle_synth()]
    for source in sources:
        log.info("%s: %d rows, categories %s", source["source"].iloc[0], len(source),
                 source["category"].value_counts().to_dict())

    df = drop_real_test(clean(pd.concat(sources, ignore_index=True)))
    for name, part in split(df).items():
        path = DATA_DIR / f"sms_{name}.csv"
        part.to_csv(path, index=False)
        log.info("%-12s %6d rows  scam=%.2f%%  -> %s", name, len(part), 100 * part["label"].mean(), path)
    log.info("Message length p95: %d chars", int(df["text"].str.len().quantile(0.95)))


if __name__ == "__main__":
    main()
