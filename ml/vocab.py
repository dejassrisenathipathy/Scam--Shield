import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch

from ml.text import normalize

PAD_ID = 0
UNK_ID = 1
MAX_LEN = 200


class CharVocab:
    def __init__(self, chars: Iterable[str]):
        self.index = {ch: i + 2 for i, ch in enumerate(chars)}

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 5) -> "CharVocab":
        counts = Counter(ch for text in texts for ch in normalize(text))
        return cls(sorted(ch for ch, n in counts.items() if n >= min_count))

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        chars = json.loads(path.read_text(encoding="utf-8"))["chars"]
        return cls(sorted(chars, key=chars.get))

    def __len__(self) -> int:
        return len(self.index) + 2

    def encode(self, text: str) -> list[int]:
        ids = [self.index.get(ch, UNK_ID) for ch in normalize(text)[:MAX_LEN]]
        return ids + [PAD_ID] * (MAX_LEN - len(ids))

    def encode_batch(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def save(self, path: Path) -> None:
        spec = {
            "max_len": MAX_LEN,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "unit": "unicode_code_point",
            "normalization": ["NFKC", "collapse whitespace to single space", "strip"],
            "truncation": "keep first max_len code points, right-pad with pad_id",
            "chars": self.index,
        }
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
