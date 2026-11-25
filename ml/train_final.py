"""Train the FINAL SMS models for the app, the same way ml.cv_real measured them (90.9%).

cv_real trained 5 temporary models (one per fold) and saved none. This trains one model per
size on ALL the data cv_real used (463 real + all their synthetic rewordings + public_20),
then exports fp32 and int8 .pte files, vocab.json and test_vectors.json for the app.

The old models in models/ are moved to models/models_old_before_final/ first.

Run from ~/scam_shield with .venv active:   python -m ml.train_final
Then give the app: models/sms_deep_int8.pte + models/vocab.json + models/test_vectors.json

NOTE: these models trained on the 463 real messages, so their score on them is NOT a test
result. The honest accuracy stays the cross-validation figure from ml.cv_real / ml.cutoffs_real.
"""
import json
import logging
import shutil
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ml.char_cnn import CONFIGS, parameter_count
from ml.export_sms import lower, operator_report, quantize, run_module, run_pte
from ml.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED
from ml.text import normalize
from ml.train_sms import train
from ml.vocab import CharVocab

log = logging.getLogger("train_final")
COLS = ["text", "label"]
OLD_FILES = ["sms_tiny.pt", "sms_deep.pt", "sms_tiny_fp32.pte", "sms_tiny_int8.pte",
             "sms_deep_fp32.pte", "sms_deep_int8.pte", "vocab.json", "test_vectors.json", "models.md",
             "probe.pte"]


def backup_old_models() -> None:
    backup = MODELS_DIR / "models_old_before_final"
    if backup.exists():
        log.info("Backup folder already exists, leaving it: %s", backup)
        return
    backup.mkdir()
    for f in OLD_FILES:
        if (MODELS_DIR / f).exists():
            shutil.move(str(MODELS_DIR / f), str(backup / f))
    log.info("Moved old models to %s", backup)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    torch.manual_seed(SEED)

    # Same data as ml.cv_real, but every fold's real messages are used for training
    real = pd.read_csv(DATA_DIR / "real_463_grouped.csv")
    synth = pd.read_csv(DATA_DIR / "external" / "synth_grouped.csv")
    public = pd.read_csv(DATA_DIR / "public_20.csv")
    for df in (real, synth, public):
        df["text"] = df["text"].astype(str).map(normalize)
    pool = pd.concat([real[COLS], synth[COLS], public[COLS]], ignore_index=True)
    fit, val = train_test_split(pool, test_size=0.1, stratify=pool["label"], random_state=SEED)
    log.info("Training pool: %d messages (%d real, %d synthetic, %d public)",
             len(pool), len(real), len(synth), len(public))

    backup_old_models()
    vocab = CharVocab.build(fit["text"])
    vocab.save(MODELS_DIR / "vocab.json")
    log.info("Vocabulary: %d symbols", len(vocab))

    calibration = vocab.encode_batch(fit["text"].sample(512, random_state=SEED))
    x_val = vocab.encode_batch(val["text"])
    sample = x_val[:1]

    # Test vectors: 5 scam + 5 genuine real messages, for checking the app gives the same scores
    vec_rows = pd.concat([real[real["label"] == c].head(5) for c in (1, 0)])
    vec_ids = vocab.encode_batch(vec_rows["text"])
    vectors = [{"text": t, "label": int(y), "input_ids": vec_ids[i].tolist(), "expected": {}}
               for i, (t, y) in enumerate(zip(vec_rows["text"], vec_rows["label"]))]

    rows = []
    for name in ("tiny", "deep"):
        config = CONFIGS[name]
        model, threshold, history = train(config, vocab, fit, val,
                                          epochs=12, batch_size=128, lr=2e-3, patience=3)
        model.eval()
        torch.save({"config": asdict(config), "vocab_size": len(vocab), "threshold": threshold,
                    "state_dict": model.state_dict()}, MODELS_DIR / f"sms_{name}.pt")

        for precision, module in {"fp32": model, "int8": quantize(model, sample, calibration)}.items():
            file = f"sms_{name}_{precision}.pte"
            path = MODELS_DIR / file
            edge = lower(module, sample)
            path.write_bytes(edge.to_executorch().buffer)

            n = min(200, len(x_val))
            probs = run_pte(path, x_val[:n])
            ref = run_module(module, x_val[:n])
            fp32_ref = run_module(model, x_val[:n])
            for v, p in zip(vectors, run_pte(path, vec_ids)):
                v["expected"][file] = round(float(p), 6)
            row = {"file": file, "size_kb": round(path.stat().st_size / 1024, 1),
                   "params": parameter_count(model), "best_epoch_val_f1": max(h["f1"] for h in history),
                   "max_diff_vs_pytorch": float(np.abs(probs - ref).max()),
                   "max_diff_vs_fp32": float(np.abs(probs - fp32_ref).max()),
                   **operator_report(edge)}
            rows.append(row)
            log.info("%-20s %6.1f KB  params %d  diff vs PyTorch %.2e  diff vs fp32 %.3f",
                     file, row["size_kb"], row["params"], row["max_diff_vs_pytorch"], row["max_diff_vs_fp32"])

    (MODELS_DIR / "test_vectors.json").write_text(json.dumps(vectors, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "train_final.json").write_text(json.dumps(rows, indent=2))

    md = ["# SMS scam detector models (final, trained 19 Nov)", "",
          "Trained the same way as the honest cross-validation (ml.cv_real): 463 real Indian SMS + their "
          "synthetic rewordings + 20% of public data. Honest accuracy (cross-validated, deep + rules): "
          "90.9%. These files trained on all 463 real messages, so do not score them on those messages.", "",
          "App cut-offs on the combined risk score (deep model): Suspicious >= 0.39, Scam >= 0.92.", "",
          "| File | Size (KB) | Params | Max diff vs PyTorch | Max diff vs fp32 |", "|---|---|---|---|---|"]
    md += [f"| `{r['file']}` | {r['size_kb']} | {r['params']:,} | {r['max_diff_vs_pytorch']:.2e} | "
           f"{r['max_diff_vs_fp32']:.3f} |" for r in rows]
    md += ["", "## Interface", "",
           "- Runtime: ExecuTorch, XNNPACK backend, method `forward`",
           "- Input: `int64` tensor `[1, 200]`, character ids from `vocab.json` (this export's vocab only)",
           "- Output: `float32` tensor `[1]`, scam probability in `[0, 1]`",
           "- Reference inputs and expected outputs: `test_vectors.json`", "", "## Operators", ""]
    for r in rows:
        outside = ", ".join(f"{k} x{v}" for k, v in r["ops_outside_xnnpack"].items()) or "none"
        md.append(f"- `{r['file']}`: {r['xnnpack_delegate_blocks']} XNNPACK delegate block(s); outside XNNPACK: {outside}")
    (MODELS_DIR / "models.md").write_text("\n".join(md) + "\n")
    log.info("Done. Give the app: models/sms_deep_int8.pte, models/vocab.json, models/test_vectors.json")


if __name__ == "__main__":
    main()
