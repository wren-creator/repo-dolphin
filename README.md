# repo-dolphin

A local, private Q&A assistant over your own repositories: ask a question in
plain English, get an answer grounded in your actual code and docs, cited by
file. Everything runs on this machine through Ollama, nothing leaves it.

## Why RAG, not fine-tuning

The obvious-sounding approach, fine-tune a model on your repos, is the wrong
tool for this job. A fine-tuned adapter is a frozen snapshot: the moment you
commit again, its "knowledge" of your codebase is stale, and re-training on
every commit isn't realistic. Retrieval-augmented generation instead indexes
your repos' actual current content and pulls the relevant pieces into the
prompt at question time, so re-running `index.py` after a work session is
all it takes to stay current. Fine-tuning still has a real place here, just
a narrower one: teaching a model *how to answer* (citation format, house
style, unusual syntax like RPG IV/DDS), never *what's in the repos*.

## Stack

- **Chat model**: `dolphin-mistral` (7B, Q4, via Ollama) — uncensored
  instruction-tuned model, good for freely discussing security-tooling
  internals without a safety-tuned model getting cagey about it.
- **Embedding model**: `nomic-embed-text` (via Ollama) — 768-dim, cheap
  enough to run alongside the chat model without competing for memory.
- **Vector store**: a flat numpy matrix, cosine similarity via a dot
  product. At the scale of a few repos (thousands of chunks, not millions),
  a real vector database (Chroma, FAISS, Qdrant) would add a dependency
  without adding speed.
- **Chunking**: Python files split at function/class boundaries with the
  stdlib `ast` module (a chunk is always a whole function, never a
  fragment). Everything else falls back to line-boundary-aware windows.
  Every chunk gets a small header (repo name, file path) prepended before
  embedding, so it still makes sense pulled out of context.

Sized for a 16GB Apple Silicon MacBook Pro: `dolphin-mistral` at Q4 runs
~5-6GB, comfortably inside 16GB unified memory with headroom for everything
else. See `LLM-calc` (sibling repo) for the sizing math if moving to
different hardware or a bigger model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull dolphin-mistral
ollama pull nomic-embed-text

cp config.example.yaml config.yaml   # then edit the repo list, config.yaml is gitignored so local paths stay local
```

## Usage

`ask.sh` and `index.sh` handle venv activation for you, that's the whole
point of them:

```bash
./index.sh                    # build (or rebuild, after any indexed repo changes) the index
./ask.sh "how does the combat system in mud work"   # one-shot answer
./ask.sh                      # interactive session — loads the index once, ask as many questions as you want
```

The interactive session (`repl.py` underneath) matters for more than
convenience: `ask.py` reloads the whole index from disk on every single
invocation, fine for a one-off question, wasteful for a back-and-forth. The
session loads once and reuses it.

Edit `config.yaml` to change which repos get indexed — it's a plain list of
`name`/`path` pairs, not auto-discovered, so you decide what's in scope.

## What "start small" means here

Currently indexing 4 repos (see `config.yaml`), not all ~30 in `git/`. Prove
the pipeline is actually useful on repos you reference often before paying
the indexing cost (and the retrieval-quality cost of a much bigger, noisier
index) to point it at everything.

## Roadmap

- [x] Incremental re-indexing (hash files, only re-embed what changed)
      instead of a full rebuild every time
- [ ] Hybrid search (dense + BM25/keyword) for exact identifier lookups
      dense retrieval alone tends to miss
- [ ] A small reranker pass if retrieval quality needs sharpening once more
      repos are added
- [ ] LoRA on `dolphin-mistral`, scoped to answer format/citation style —
      only once the pure-RAG baseline is actually in daily use and the gap
      it would close is concrete, not hypothetical
