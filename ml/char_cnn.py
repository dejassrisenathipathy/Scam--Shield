from dataclasses import dataclass

import torch
from torch import nn

from ml.vocab import PAD_ID


@dataclass(frozen=True)
class CharCNNConfig:
    name: str
    embed_dim: int
    kernel_sizes: tuple[int, ...]
    filters: int
    hidden: int
    dropout: float


CONFIGS = {
    "tiny": CharCNNConfig("tiny", embed_dim=32, kernel_sizes=(5,), filters=64, hidden=0, dropout=0.1),
    "deep": CharCNNConfig("deep", embed_dim=64, kernel_sizes=(3, 5, 7), filters=128, hidden=64, dropout=0.3),
}


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, config: CharCNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            nn.Conv1d(config.embed_dim, config.filters, k, padding=k // 2) for k in config.kernel_sizes
        )
        features = config.filters * len(config.kernel_sizes)
        layers: list[nn.Module] = [nn.Dropout(config.dropout)]
        if config.hidden:
            layers += [nn.Linear(features, config.hidden), nn.ReLU(), nn.Dropout(config.dropout)]
            features = config.hidden
        layers.append(nn.Linear(features, 1))
        self.head = nn.Sequential(*layers)

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids).transpose(1, 2)
        x = torch.cat([torch.relu(conv(x)).amax(dim=2) for conv in self.convs], dim=1)
        return self.head(x).squeeze(1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(ids))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
