import json
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from ml.metrics import binary_metrics
from ml.paths import DATA_DIR, RESULTS_DIR, SEED

log = logging.getLogger("baseline_sms")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    train_path, test_path = DATA_DIR / "sms_train.csv", DATA_DIR / "sms_test_public.csv"
    if not (train_path.exists() and test_path.exists()):
        raise SystemExit("Missing data splits. Run: python -m ml.prepare_sms")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    model.fit(train["text"], train["label"])
    test = test.assign(pred=model.predict(test["text"]))

    metrics = {"model": "tfidf_char_lr", "test_set": "public", **binary_metrics(test["label"], test["pred"])}
    (RESULTS_DIR / "baseline_sms.json").write_text(json.dumps(metrics, indent=2))
    test[test["label"] != test["pred"]].to_csv(RESULTS_DIR / "baseline_sms_errors.csv", index=False)
    log.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
