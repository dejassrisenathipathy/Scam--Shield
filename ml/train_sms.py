import argparse
import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.char_cnn import CONFIGS, CharCNN, CharCNNConfig, parameter_count
from ml.metrics import binary_metrics
from ml.paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, SEED
from ml.vocab import CharVocab

log = logging.getLogger("train_sms")


@torch.no_grad()
def predict(model: CharCNN, ids: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat([model(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]).numpy()


def best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [binary_metrics(labels, (probs >= t).astype(int))["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def train(config: CharCNNConfig, vocab: CharVocab, train_df: pd.DataFrame, val_df: pd.DataFrame,
          epochs: int, batch_size: int, lr: float, patience: int) -> tuple[CharCNN, float, list[dict]]:
    torch.manual_seed(SEED)
    model = CharCNN(len(vocab), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train = vocab.encode_batch(train_df["text"])
    y_train = torch.tensor(train_df["label"].to_numpy(), dtype=torch.float32)
    x_val, y_val = vocab.encode_batch(val_df["text"]), val_df["label"].to_numpy()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    best_f1, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model.logits(ids), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        val_metrics = binary_metrics(y_val, (predict(model, x_val) >= 0.5).astype(int))
        history.append({"epoch": epoch, "train_loss": round(total / len(x_train), 4), **val_metrics})
        log.info("%s epoch %2d  loss %.4f  val f1 %.4f  val fpr %.4f",
                 config.name, epoch, total / len(x_train), val_metrics["f1"], val_metrics["fpr"])
        if val_metrics["f1"] > best_f1:
            best_f1, stale = val_metrics["f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_threshold(y_val, predict(model, x_val)), history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny and deep character-level CNN SMS scam detectors.")
    parser.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    train_df = pd.read_csv(DATA_DIR / "sms_train.csv")
    val_df = pd.read_csv(DATA_DIR / "sms_val.csv")
    test_df = pd.read_csv(DATA_DIR / "sms_test_public.csv")
    vocab = CharVocab.build(train_df["text"])
    vocab.save(MODELS_DIR / "vocab.json")
    log.info("Vocabulary: %d symbols", len(vocab))

    x_test, y_test = vocab.encode_batch(test_df["text"]), test_df["label"].to_numpy()
    for name in args.models:
        config = CONFIGS[name]
        model, threshold, history = train(config, vocab, train_df, val_df,
                                          args.epochs, args.batch_size, args.lr, args.patience)
        metrics = binary_metrics(y_test, (predict(model, x_test) >= threshold).astype(int))
        torch.save({"config": asdict(config), "vocab_size": len(vocab), "threshold": threshold,
                    "state_dict": model.state_dict()}, MODELS_DIR / f"sms_{name}.pt")
        summary = {"model": f"sms_{name}", "params": parameter_count(model), "threshold": threshold,
                   "test_set": "public", **metrics, "history": history}
        (RESULTS_DIR / f"train_sms_{name}.json").write_text(json.dumps(summary, indent=2))
        log.info("%s params=%d threshold=%.2f test %s", name, summary["params"], threshold,
                 {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "fpr")})


if __name__ == "__main__":
    main()
