from __future__ import annotations

import hashlib
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".mobi"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
