"""Build the final training data: all 463 real messages, their synthetic paraphrases,
and 20% of the cleaned public data (UCI + Shade63, near-copies of real messages removed).
Writes data/sms_train.csv, sms_val.csv and sms_test_public.csv for ml.train_sms.
Run from ~/scam_shield:  python -m ml.prepare_v2
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.paths import DATA_DIR, SEED
from ml.text import normalize

log = logging.getLogger("prepare_v2")
COLS = ["text", "label", "category", "source"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    real = pd.read_csv(DATA_DIR / "real_463_grouped.csv").assign(source="real")
    synth = pd.read_csv(DATA_DIR / "external" / "synth_grouped.csv").assign(source="kaggle_synth")
    public = pd.read_csv(DATA_DIR / "public_20.csv")
    df = pd.concat([real[COLS], synth[COLS], public[COLS]], ignore_index=True)
    df["text"] = df["text"].astype(str).map(normalize)
    df = df[~df["text"].str.lower().duplicated()].reset_index(drop=True)
    log.info("Sources: %s", df["source"].value_counts().to_dict())

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    for name, part in {"train": train, "val": val, "test_public": test}.items():
        part.to_csv(DATA_DIR / f"sms_{name}.csv", index=False)
        log.info("%-12s %5d rows  scam=%.1f%%", name, len(part), 100 * part["label"].mean())
    log.info("The real messages are now in training: report the accuracy from ml.cv_real, not eval_real.")


if __name__ == "__main__":
    main()
