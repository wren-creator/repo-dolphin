"""Talks to Ollama's local embedding endpoint. No API key, no network call
leaves the machine — the model, the embedding step, and the answer generation
all run on this laptop.
"""

from __future__ import annotations

import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 64  # keeps a single request body (and Ollama's own batch) reasonably sized


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Returns an (n_texts, embedding_dim) array, L2-normalized so a plain dot
    product between rows is cosine similarity."""
    if not texts:
        return np.zeros((0, 0))

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": batch}, timeout=120)
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return arr / norms
