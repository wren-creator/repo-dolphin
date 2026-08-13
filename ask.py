"""Ask a question about your indexed repos.

    python ask.py "how does the combat system in mud work"

Retrieves the most relevant chunks from the vector index, hands them to the
local chat model as grounding context, and prints its answer. Everything
here runs locally through Ollama — no network call leaves the machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

from indexer.store import VectorStore

ROOT = Path(__file__).resolve().parent
OLLAMA_URL = "http://localhost:11434"


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def build_prompt(question: str, hits: list[tuple]) -> str:
    context = "\n\n".join(f"[{chunk.repo}/{chunk.file_path}]\n{chunk.text}" for chunk, _ in hits)
    return (
        "You are answering questions about the user's own code repositories, using only the "
        "context below. Cite the repo and file path when you reference something specific. If "
        "the context doesn't answer the question, say so plainly instead of guessing.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )


def load_store(config: dict) -> VectorStore:
    index_path = ROOT / config["index_path"]
    if not index_path.exists():
        raise SystemExit(f"No index found at {index_path}. Run `python index.py` first.")
    return VectorStore.load(index_path)


def answer(question: str, store: VectorStore, config: dict) -> str:
    """Answers one question against an already-loaded store — the piece repl.py
    reuses so a session doesn't reload the index from disk on every question."""
    hits = store.search(question, top_k=config.get("top_k", 6))

    print("--- retrieved ---", file=sys.stderr)
    for chunk, score in hits:
        print(f"  [{score:.3f}] {chunk.repo}/{chunk.file_path}", file=sys.stderr)
    print(file=sys.stderr)

    prompt = build_prompt(question, hits)
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": config["chat_model"], "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def ask(question: str, config: dict) -> str:
    """One-shot convenience: load the store fresh, answer, done. Fine for a
    single CLI invocation; repl.py calls load_store()/answer() directly
    instead so a multi-question session only pays the load cost once."""
    return answer(question, load_store(config), config)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about your indexed repos.")
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()

    config = load_config()
    result = ask(" ".join(args.question), config)
    print(result)


if __name__ == "__main__":
    _cli()
