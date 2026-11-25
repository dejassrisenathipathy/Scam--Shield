import json
import logging
import operator
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

logging.getLogger("torch").setLevel(logging.ERROR)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # noqa: E402
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # noqa: E402
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

try:
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402
except ImportError:
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

from ml.char_cnn import CharCNN, CharCNNConfig, parameter_count  # noqa: E402
from ml.metrics import binary_metrics  # noqa: E402
from ml.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from ml.vocab import CharVocab  # noqa: E402

CALIBRATION_SAMPLES = 512
PARITY_SAMPLES = 200
TEST_VECTORS_PER_CLASS = 5

log = logging.getLogger("export_sms")


def load_checkpoint(name: str) -> tuple[CharCNN, float]:
    checkpoint = torch.load(MODELS_DIR / f"sms_{name}.pt", weights_only=False)
    config = checkpoint["config"]
    model = CharCNN(checkpoint["vocab_size"], CharCNNConfig(**{**config, "kernel_sizes": tuple(config["kernel_sizes"])}))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["threshold"]


def quantize(model: CharCNN, sample: torch.Tensor, calibration: torch.Tensor) -> nn.Module:
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(torch.export.export(model, (sample,)).module(), quantizer)
    with torch.no_grad():
        for i in range(len(calibration)):
            prepared(calibration[i:i + 1])
    return convert_pt2e(prepared)


def lower(module: nn.Module, sample: torch.Tensor) -> EdgeProgramManager:
    return to_edge_transform_and_lower(torch.export.export(module, (sample,)), partitioner=[XnnpackPartitioner()])


def operator_report(edge: EdgeProgramManager) -> dict:
    delegated, outside = 0, Counter()
    for node in edge.exported_program().graph_module.graph.nodes:
        if node.op != "call_function" or node.target is operator.getitem:
            continue
        name = getattr(node.target, "__name__", str(node.target))
        if "call_delegate" in name:
            delegated += 1
        else:
            outside[name] += 1
    return {"xnnpack_delegate_blocks": delegated, "ops_outside_xnnpack": dict(outside)}


@torch.no_grad()
def run_pte(path: Path, ids: torch.Tensor) -> np.ndarray:
    method = Runtime.get().load_program(str(path)).load_method("forward")
    return np.array([float(method.execute([ids[i:i + 1]])[0].reshape(-1)[0]) for i in range(len(ids))])


@torch.no_grad()
def run_module(module: nn.Module, ids: torch.Tensor) -> np.ndarray:
    return np.array([float(module(ids[i:i + 1]).reshape(-1)[0]) for i in range(len(ids))])


def write_models_md(rows: list[dict], vocab: CharVocab) -> None:
    lines = [
        "# SMS scam detector models",
        "",
        "| File | Size (KB) | Params | Threshold | Accuracy | Precision | Recall | F1 | FPR | Max diff vs PyTorch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['file']}` | {r['size_kb']} | {r['params']:,} | {r['threshold']} | {r['accuracy']} | "
                     f"{r['precision']} | {r['recall']} | {r['f1']} | {r['fpr']} | {r['max_abs_diff']:.2e} |")
    lines += [
        "",
        "Metrics are on the public test split (UCI + Shade63). Real Indian SMS metrics follow after the test set freeze.",
        "",
        "## Interface",
        "",
        "- Runtime: ExecuTorch, XNNPACK backend, method `forward`",
        f"- Input: `int64` tensor `[1, {len(vocab.encode(''))}]`, character ids from `vocab.json`",
        "- Output: `float32` tensor `[1]`, scam probability in `[0, 1]`; scam if `>= threshold`",
        "- Tokenizer: none. Character lookup via `vocab.json` (NFKC, collapse whitespace, strip, "
        "first 200 Unicode code points, pad with 0, unknown = 1)",
        "- Reference inputs and expected outputs: `test_vectors.json`",
        "",
        "## Operators",
        "",
    ]
    for r in rows:
        outside = ", ".join(f"{k} x{v}" for k, v in r["ops_outside_xnnpack"].items()) or "none"
        lines.append(f"- `{r['file']}`: {r['xnnpack_delegate_blocks']} XNNPACK delegate block(s); outside XNNPACK: {outside}")
    (MODELS_DIR / "models.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    torch.manual_seed(SEED)
    vocab = CharVocab.load(MODELS_DIR / "vocab.json")
    train_df = pd.read_csv(DATA_DIR / "sms_train.csv")
    test_df = pd.read_csv(DATA_DIR / "sms_test_public.csv")
    x_test, y_test = vocab.encode_batch(test_df["text"]), test_df["label"].to_numpy()
    calibration = vocab.encode_batch(train_df["text"].sample(CALIBRATION_SAMPLES, random_state=SEED))
    sample = x_test[:1]

    vector_rows = pd.concat([test_df[test_df["label"] == c].head(TEST_VECTORS_PER_CLASS) for c in (1, 0)])
    vector_ids = vocab.encode_batch(vector_rows["text"])
    vectors = [{"text": t, "label": int(y), "input_ids": vector_ids[i].tolist(), "expected": {}}
               for i, (t, y) in enumerate(zip(vector_rows["text"], vector_rows["label"]))]

    rows = []
    for name in ("tiny", "deep"):
        model, threshold = load_checkpoint(name)
        variants = {"fp32": model, "int8": quantize(model, sample, calibration)}
        for precision, module in variants.items():
            file = f"sms_{name}_{precision}.pte"
            path = MODELS_DIR / file
            edge = lower(module, sample)
            path.write_bytes(edge.to_executorch().buffer)

            probs = run_pte(path, x_test)
            reference = run_module(module, x_test[:PARITY_SAMPLES])
            metrics = binary_metrics(y_test, (probs >= threshold).astype(int))
            for vector, p in zip(vectors, run_pte(path, vector_ids)):
                vector["expected"][file] = round(float(p), 6)

            row = {"file": file, "size_kb": round(path.stat().st_size / 1024, 1), "params": parameter_count(model),
                   "threshold": threshold, **{k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "fpr")},
                   "confusion": metrics["confusion"],
                   "max_abs_diff": float(np.abs(probs[:PARITY_SAMPLES] - reference).max()),
                   **operator_report(edge)}
            rows.append(row)
            log.info("%-20s %7.1f KB  f1 %.4f  fpr %.4f  diff %.2e  outside xnnpack %s", file, row["size_kb"],
                     row["f1"], row["fpr"], row["max_abs_diff"], row["ops_outside_xnnpack"] or "none")

    (MODELS_DIR / "test_vectors.json").write_text(json.dumps(vectors, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "export_sms.json").write_text(json.dumps(rows, indent=2))
    write_models_md(rows, vocab)
    log.info("Wrote %s", MODELS_DIR / "models.md")


if __name__ == "__main__":
    main()
