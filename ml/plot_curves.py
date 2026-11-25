"""Training vs validation curves for one cross-validation fold (for the report).

Retrains fold 1 of ml.cv_real exactly the same way and plots the history train() returns.
Run from ~/scam_shield:  python -m ml.plot_curves --model deep
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from ml.char_cnn import CONFIGS
from ml.paths import DATA_DIR, RESULTS_DIR, SEED
from ml.text import normalize
from ml.train_sms import train
from ml.vocab import CharVocab

COLS = ["text", "label"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deep")
    ap.add_argument("--fold", type=int, default=1)
    a = ap.parse_args()

    real = pd.read_csv(DATA_DIR / "real_463_grouped.csv")
    synth = pd.read_csv(DATA_DIR / "external" / "synth_grouped.csv")
    public = pd.read_csv(DATA_DIR / "public_20.csv")
    for df in (real, synth, public):
        df["text"] = df["text"].astype(str).map(normalize)
    y, groups = real["label"].to_numpy(), real["group"].to_numpy()

    folds = list(GroupKFold(n_splits=5).split(real, y, groups))
    tr, _te = folds[a.fold - 1]
    keep = synth["group"].isin(set(groups[tr])) | (synth["group"] == -1)
    pool = pd.concat([real.iloc[tr][COLS], synth[keep][COLS], public[COLS]], ignore_index=True)
    fit, val = train_test_split(pool, test_size=0.1, stratify=pool["label"], random_state=SEED)
    vocab = CharVocab.build(fit["text"])
    _model, _th, history = train(CONFIGS[a.model], vocab, fit, val,
                                 epochs=12, batch_size=128, lr=2e-3, patience=3)

    h = pd.DataFrame(history)          # works for a list of dicts or a dict of lists
    print("History columns:", list(h.columns))
    print(h.round(4).to_string())
    num = h.select_dtypes("number").drop(columns=[c for c in h.columns if c.lower() == "epoch"], errors="ignore")
    x = h["epoch"] if "epoch" in h.columns else range(1, len(h) + 1)

    loss_cols = [c for c in num.columns if "loss" in c.lower()]
    other_cols = [c for c in num.columns if c not in loss_cols]
    groups_to_plot = [g for g in (loss_cols, other_cols) if g]
    fig, axes = plt.subplots(1, len(groups_to_plot), figsize=(5 * len(groups_to_plot), 3.2), squeeze=False)
    for ax, cols in zip(axes[0], groups_to_plot):
        for c in cols:
            ax.plot(list(x), h[c], marker="o", label=c)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0][0].set_title(f"{a.model} model, fold {a.fold}: training vs validation")
    plt.tight_layout()
    out = RESULTS_DIR / f"curves_{a.model}_fold{a.fold}.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
