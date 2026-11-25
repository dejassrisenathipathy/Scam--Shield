"""Pick the Safe / Suspicious / Scam cut-offs honestly, on the 463 real messages.

Uses the out-of-fold model scores that ml.cv_real already saved, adds the app's rules
(ml.risk_rules, a copy of RiskEngine.kt), and picks 2 cut-offs on the final risk score.

Honest: each fold's messages are judged with cut-offs chosen ONLY from the other 4 folds,
so no message helps choose the cut-off it is judged by. No retraining needed.

Run from ~/scam_shield, after python -m ml.cv_real:
    python -m ml.cutoffs_real --model tiny
    python -m ml.cutoffs_real --model tiny --max-fa 0.03 --min-recall 0.93
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ml.paths import DATA_DIR, RESULTS_DIR
from ml.risk_rules import assess

LEVELS = ["Safe", "Suspicious", "Scam"]


def pick(risk: np.ndarray, y: np.ndarray, max_fa: float, min_recall: float) -> tuple[float, float]:
    """Scam cut-off: at most max_fa of genuine messages reach it.
    Suspicious cut-off: at least min_recall of scams reach it."""
    gen = np.sort(risk[y == 0])
    allowed = int(np.floor(max_fa * len(gen)))
    scam_cut = float(gen[len(gen) - allowed - 1] + 1e-6)
    scams = np.sort(risk[y == 1])[::-1]
    need = int(np.ceil(min_recall * len(scams)))
    sus_cut = float(scams[max(need, 1) - 1])
    return min(sus_cut, scam_cut), scam_cut


def level(r: np.ndarray, sus: float, scam: float) -> np.ndarray:
    return np.where(r >= scam, 2, np.where(r >= sus, 1, 0))


def table(y: np.ndarray, lv: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame(0, index=["Real scam", "Genuine"], columns=LEVELS)
    for lab, name in ((1, "Real scam"), (0, "Genuine")):
        for i, col in enumerate(LEVELS):
            t.loc[name, col] = int(((y == lab) & (lv == i)).sum())
    return t


def rates(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {"scams_caught": round(float(tp / (y == 1).sum()), 4),
            "false_alarm_rate": round(float(fp / (y == 0).sum()), 4),
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="which cv_real_<model>.csv to use")
    ap.add_argument("--max-fa", type=float, default=0.02, help="max share of genuine SMS marked Scam")
    ap.add_argument("--min-recall", type=float, default=0.95, help="min share of scams marked Suspicious or Scam")
    a = ap.parse_args()

    raw = pd.read_csv(DATA_DIR / "real_463_grouped.csv")          # raw text: rules run on it, like the app
    cv = pd.read_csv(RESULTS_DIR / f"cv_real_{a.model}.csv")       # out-of-fold model scores
    assert len(raw) == len(cv) and (raw["label"].to_numpy() == cv["label"].to_numpy()).all(), \
        "Row order differs: rerun python -m ml.cv_real"
    y, groups = raw["label"].to_numpy(), raw["group"].to_numpy()
    senders = raw["sender"].astype(str) if "sender" in raw else pd.Series(["unknown"] * len(raw))
    print(f"Sender IDs in data: {'yes' if 'sender' in raw else 'NO (sender checks not tested)'}")

    rows = [assess(s, str(t), float(p)) for s, t, p in zip(senders, raw["text"], cv["score"])]
    risk = np.array([r["risk"] for r in rows])

    # Same folds as ml.cv_real (GroupKFold has no randomness)
    lv = np.zeros(len(raw), dtype=int)
    cuts = []
    print(f"\n{'fold':>4} {'Suspicious >=':>14} {'Scam >=':>9}   (chosen on the other 4 folds)")
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(raw, y, groups), 1):
        sus, scam = pick(risk[tr], y[tr], a.max_fa, a.min_recall)
        lv[te] = level(risk[te], sus, scam)
        cuts.append((sus, scam))
        print(f"{k:>4} {sus:14.3f} {scam:9.3f}")

    print(f"\nHONEST RESULT on {len(y)} real messages (each judged with cut-offs it did not help choose):")
    print(table(y, lv).to_string())
    warned, scam_lv = rates(y, lv >= 1), rates(y, lv == 2)
    print(f"\nWarned (Suspicious or Scam): {warned}")
    print(f"Marked Scam:                 {scam_lv}")
    model_only = rates(y, cv["score"].to_numpy() >= 0.5)
    print(f"Model alone at 0.5:          {model_only}")

    sus_f = float(np.median([c[0] for c in cuts]))
    scam_f = float(np.median([c[1] for c in cuts]))
    sus_f = min(sus_f, scam_f)
    print(f"\nPUT IN THE APP:  Suspicious if risk >= {sus_f:.2f},  Scam if risk >= {scam_f:.2f}")

    out = raw.assign(model_score=cv["score"], rule_part=[r["rule_part"] for r in rows],
                     risk=risk.round(4), level=[LEVELS[i] for i in lv],
                     reasons=[" | ".join(r["reasons"]) for r in rows])
    out.to_csv(RESULTS_DIR / f"risk_real_{a.model}.csv", index=False)
    (RESULTS_DIR / f"cutoffs_{a.model}.json").write_text(json.dumps({
        "model": a.model, "targets": {"max_fa_at_scam": a.max_fa, "min_recall_at_suspicious": a.min_recall},
        "app_cutoffs": {"suspicious": round(sus_f, 3), "scam": round(scam_f, 3)},
        "fold_cutoffs": [[round(s, 3), round(c, 3)] for s, c in cuts],
        "honest_table": table(y, lv).to_dict(), "warned": warned, "marked_scam": scam_lv,
        "model_alone_0.5": model_only}, indent=2))
    print(f"Saved results/cutoffs_{a.model}.json and results/risk_real_{a.model}.csv")


if __name__ == "__main__":
    main()
