"""A flat, in-memory vector store — no Chroma/FAISS/Qdrant. At the scale of a
handful of personal repos (thousands of chunks, not millions), a plain numpy
matrix and a cosine-similarity dot product is fast enough that a real vector
database would only add a dependency without adding speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from indexer.chunker import Chunk
from indexer.embed import embed_texts


@dataclass
class VectorStore:
    chunks: list[Chunk]
    vectors: np.ndarray  # (n_chunks, embedding_dim), L2-normalized

    def search(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vec = embed_texts([query])[0]
        scores = self.vectors @ query_vec  # both sides L2-normalized => cosine similarity
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            repos=np.array([c.repo for c in self.chunks], dtype=object),
            file_paths=np.array([c.file_path for c in self.chunks], dtype=object),
            texts=np.array([c.text for c in self.chunks], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        data = np.load(path, allow_pickle=True)
        chunks = [
            Chunk(repo=r, file_path=f, text=t)
            for r, f, t in zip(data["repos"], data["file_paths"], data["texts"])
        ]
        return cls(chunks=chunks, vectors=data["vectors"])
