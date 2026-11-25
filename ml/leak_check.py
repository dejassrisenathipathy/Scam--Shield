"""Proof that the real-message test is clean.

1. How many synthetic Kaggle messages are near-copies (70%+ similar, digits ignored) of the
   463 real messages. This is why the old 98.7% result was not valid.
2. For grouped 5-fold cross-validation (ml.cv_real), how similar each test message is to the
   closest message in that fold's training data.
Run from ~/scam_shield:  python -m ml.leak_check
"""
import difflib
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

from ml.paths import DATA_DIR


def key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", str(text).lower())).strip()


def closest(queries: list[str], pool: list[str], top: int = 30) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(queries + pool)
    sim = cosine_similarity(vec.transform(queries), vec.transform(pool))
    return np.array([max(difflib.SequenceMatcher(None, q, pool[j]).ratio() for j in np.argsort(-row)[:top])
                     for q, row in zip(queries, sim)])


def main() -> None:
    real = pd.read_csv(DATA_DIR / "real_463_grouped.csv")
    synth = pd.read_csv(DATA_DIR / "external" / "synth_grouped.csv")
    public = pd.read_csv(DATA_DIR / "public_20.csv")
    real_k, synth_k = real["text"].map(key).tolist(), synth["text"].map(key).tolist()

    s = closest(synth_k, real_k)
    print(f"1) Synthetic messages that near-copy a real one (>=0.7): {(s >= 0.7).sum()} of {len(synth)}")

    best = np.zeros(len(real))
    for tr, te in GroupKFold(n_splits=5).split(real, real["label"], real["group"]):
        groups = set(real["group"].iloc[tr])
        keep = synth["group"].isin(groups) | (synth["group"] == -1)
        pool = [real_k[i] for i in tr] + list(synth[keep]["text"].map(key)) + list(public["text"].map(key))
        best[te] = closest([real_k[i] for i in te], pool)
    print("2) Each test message vs the closest message in its fold's training data:")
    for t in (0.9, 0.8, 0.7, 0.6):
        print(f"   >= {t:.1f} similar: {(best >= t).sum()} of {len(real)}")
    print(f"   median similarity: {np.median(best):.2f}")


if __name__ == "__main__":
    main()
