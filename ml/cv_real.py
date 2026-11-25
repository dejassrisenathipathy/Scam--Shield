"""Honest real-message accuracy with grouped 5-fold cross-validation.

Each fold trains on 4/5 of the real message groups (plus their synthetic paraphrases and
20% of the public data) and tests on the other 1/5, which the model never saw in any form.
Run from ~/scam_shield:  python -m ml.cv_real
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from ml.char_cnn import CONFIGS
from ml.paths import DATA_DIR, RESULTS_DIR, SEED
from ml.text import normalize
from ml.train_sms import predict, train
from ml.vocab import CharVocab

log = logging.getLogger("cv_real")
COLS = ["text", "label"]


def scores(y: np.ndarray, p: np.ndarray, th: float) -> dict:
    h = (p >= th).astype(int)
    tp, fp = int(((h == 1) & (y == 1)).sum()), int(((h == 1) & (y == 0)).sum())
    fn, tn = int(((h == 0) & (y == 1)).sum()), int(((h == 0) & (y == 0)).sum())
    return {"accuracy": round((tp + tn) / len(y), 4), "recall": round(tp / (tp + fn), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "false_alarm_rate": round(fp / (fp + tn), 4), "false_alarms": fp, "missed_scams": fn}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    real = pd.read_csv(DATA_DIR / "real_463_grouped.csv")
    synth = pd.read_csv(DATA_DIR / "external" / "synth_grouped.csv")
    public = pd.read_csv(DATA_DIR / "public_20.csv")
    for df in (real, synth, public):
        df["text"] = df["text"].astype(str).map(normalize)
    y, groups = real["label"].to_numpy(), real["group"].to_numpy()

    results = {}
    for name in CONFIGS:
        probs, ths = np.zeros(len(real)), np.zeros(len(real))
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(real, y, groups), 1):
            keep = synth["group"].isin(set(groups[tr])) | (synth["group"] == -1)
            pool = pd.concat([real.iloc[tr][COLS], synth[keep][COLS], public[COLS]], ignore_index=True)
            fit, val = train_test_split(pool, test_size=0.1, stratify=pool["label"], random_state=SEED)
            vocab = CharVocab.build(fit["text"])
            model, th, _ = train(CONFIGS[name], vocab, fit, val, epochs=12, batch_size=128, lr=2e-3, patience=3)
            probs[te] = predict(model, vocab.encode_batch(real.iloc[te]["text"]))
            ths[te] = th
            log.info("%s fold %d: %s", name, fold, scores(y[te], probs[te], th))
        hits = (probs >= ths).astype(int)
        results[name] = {"fold_thresholds": scores(y, np.where(hits == 1, 1.0, 0.0), 0.5),
                         "threshold_0.5": scores(y, probs, 0.5)}
        out = real.assign(score=probs.round(4), threshold=ths, wrong=(hits != y).astype(int))
        out.to_csv(RESULTS_DIR / f"cv_real_{name}.csv", index=False)
        log.info("\n%s ON 463 REAL MESSAGES (never seen in training):\n  %s\n", name.upper(), results[name])
    (RESULTS_DIR / "cv_real_summary.json").write_text(json.dumps(results, indent=2))
    log.info("Saved results/cv_real_summary.json and results/cv_real_<model>.csv (wrong=1 marks mistakes)")


if __name__ == "__main__":
    main()
