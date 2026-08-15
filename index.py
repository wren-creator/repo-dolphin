"""Builds (or incrementally rebuilds) the vector index from the repos listed
in config.yaml.

Run after a work session on any of the configured repos to pick up changes:

    python index.py

By default this only re-embeds files whose content hash changed since the
last run (new, edited, or deleted files); everything else reuses its stored
embedding. Pass --full to force a from-scratch rebuild, e.g. after changing
the chunking logic itself, which a file hash wouldn't detect.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from indexer.chunker import chunk_file, file_hash, iter_repo_files
from indexer.embed import embed_texts
from indexer.hashes import file_key, hash_sidecar_path, load_hashes, save_hashes
from indexer.store import VectorStore

ROOT = Path(__file__).resolve().parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="force a full rebuild instead of reusing unchanged files' embeddings",
    )
    args = parser.parse_args()

    config = load_config()
    index_path = ROOT / config["index_path"]
    hashes_path = hash_sidecar_path(index_path)

    previous_hashes = {} if args.full else load_hashes(hashes_path)
    previous_by_file: dict[tuple[str, str], list] = defaultdict(list)
    if previous_hashes and index_path.exists():
        previous_store = VectorStore.load(index_path)
        for chunk, vector in zip(previous_store.chunks, previous_store.vectors):
            previous_by_file[(chunk.repo, chunk.file_path)].append((chunk, vector))

    reused_chunks, reused_vectors = [], []
    chunks_to_embed = []
    new_hashes: dict[str, str] = {}
    reused_files = changed_files = 0

    for repo in config["repos"]:
        repo_root = Path(repo["path"]).expanduser()
        if not repo_root.is_dir():
            print(f"skipping {repo['name']}: {repo_root} not found")
            continue
        for path in iter_repo_files(repo_root):
            rel_path = str(path.relative_to(repo_root))
            key = file_key(repo["name"], rel_path)
            current_hash = file_hash(path)
            new_hashes[key] = current_hash

            cached = previous_by_file.get((repo["name"], rel_path))
            if cached and previous_hashes.get(key) == current_hash:
                for chunk, vector in cached:
                    reused_chunks.append(chunk)
                    reused_vectors.append(vector)
                reused_files += 1
            else:
                chunks_to_embed.extend(chunk_file(repo["name"], repo_root, path))
                changed_files += 1

    all_chunks = reused_chunks + chunks_to_embed
    if not all_chunks:
        raise SystemExit("No chunks found — check the paths in config.yaml.")

    print(f"{reused_files} files unchanged (reusing embeddings), {changed_files} new/changed files to embed")
    if chunks_to_embed:
        print(f"embedding {len(chunks_to_embed)} chunks with {config['embed_model']}...")
        new_vectors = embed_texts([c.text for c in chunks_to_embed])
        all_vectors = np.vstack([np.array(reused_vectors), new_vectors]) if reused_vectors else new_vectors
    else:
        print("nothing to embed, all files unchanged")
        all_vectors = np.array(reused_vectors)

    store = VectorStore(chunks=all_chunks, vectors=all_vectors)
    store.save(index_path)
    save_hashes(hashes_path, new_hashes)
    print(f"saved index to {index_path} ({len(all_chunks)} chunks total)")


if __name__ == "__main__":
    main()
