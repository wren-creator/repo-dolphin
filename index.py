"""Builds (or rebuilds) the vector index from the repos listed in config.yaml.

Run after a work session on any of the configured repos to pick up changes:

    python index.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from indexer.chunker import chunk_repo
from indexer.store import VectorStore

ROOT = Path(__file__).resolve().parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    config = load_config()
    all_chunks = []
    for repo in config["repos"]:
        repo_root = Path(repo["path"]).expanduser()
        if not repo_root.is_dir():
            print(f"skipping {repo['name']}: {repo_root} not found")
            continue
        chunks = chunk_repo(repo["name"], repo_root)
        print(f"{repo['name']}: {len(chunks)} chunks from {repo_root}")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No chunks found — check the paths in config.yaml.")

    print(f"embedding {len(all_chunks)} chunks with {config['embed_model']}...")
    store = VectorStore.build(all_chunks)

    index_path = ROOT / config["index_path"]
    store.save(index_path)
    print(f"saved index to {index_path}")


if __name__ == "__main__":
    main()
