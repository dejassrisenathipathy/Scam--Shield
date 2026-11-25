import logging
import sys

import torch
from torch import nn

logging.getLogger("torch").setLevel(logging.ERROR)

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # noqa: E402
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

from ml.paths import MODELS_DIR  # noqa: E402

log = logging.getLogger("check_export")


class ProbeModel(nn.Module):
    def __init__(self, vocab_size: int = 100, dim: int = 16, filters: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.conv = nn.Conv1d(dim, filters, kernel_size=5, padding=2)
        self.head = nn.Linear(filters, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids).transpose(1, 2)
        x = torch.relu(self.conv(x)).amax(dim=2)
        return torch.sigmoid(self.head(x))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    torch.manual_seed(0)
    model = ProbeModel().eval()
    sample = torch.randint(0, 100, (1, 200), dtype=torch.long)
    path = MODELS_DIR / "probe.pte"

    program = to_edge_transform_and_lower(
        torch.export.export(model, (sample,)), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    path.write_bytes(program.buffer)

    method = Runtime.get().load_program(str(path)).load_method("forward")
    expected = model(sample).item()
    actual = method.execute([sample])[0].item()
    diff = abs(expected - actual)
    log.info("torch=%.6f executorch=%.6f diff=%.2e size=%d bytes", expected, actual, diff, path.stat().st_size)
    if diff > 1e-3:
        log.error("Parity check failed")
        sys.exit(1)
    log.info("Parity check passed")


if __name__ == "__main__":
    main()
