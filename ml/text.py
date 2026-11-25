import re
import unicodedata


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()
