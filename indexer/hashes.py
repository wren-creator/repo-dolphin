"""Tracks per-file content hashes across index builds, so `index.py` can tell
which files actually changed since the last run and skip re-embedding the
rest. Stored as a small JSON sidecar next to the index (e.g.
`data/index.hashes.json`) rather than inside the npz, since it's read and
written independently of the vectors/chunks.
"""

from __future__ import annotations

import json
from pathlib import Path


def hash_sidecar_path(index_path: Path) -> Path:
    return index_path.with_suffix(".hashes.json")


def load_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_hashes(path: Path, hashes: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True))


def file_key(repo: str, rel_path: str) -> str:
    return f"{repo}:{rel_path}"
