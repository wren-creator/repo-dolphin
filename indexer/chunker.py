"""Turns a repo's files into retrievable chunks.

Two chunking strategies:
- Python files get split at function/class boundaries using the stdlib `ast`
  module, so a chunk is always a whole function or class, never a fragment
  of one.
- Everything else (other languages, config, docs) falls back to fixed-size
  windows snapped to line boundaries, since most languages don't have a
  zero-dependency way to parse their structure, but a chunk ending mid-line
  is still a lot better than one ending mid-word.

Every chunk gets a small metadata header (repo name, file path, and for
Python, the enclosing function/class) prepended before it's embedded, so the
embedding captures *what this code is* alongside the code itself, and the
chunk still makes sense on its own when it's later pulled out of context.
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build",
    ".idea", ".pytest_cache", "checkpoints", ".mypy_cache", "target",
}
SKIP_FILE_PATTERNS = [
    "*.pyc", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg", "*.pdf",
    "*.mp4", "*.mp3", "*.wav", "*.zip", "*.tar", "*.gz", "*.woff*", "*.ttf",
    "*.bin", "*.pt", "*.pth", "*.safetensors", "*.lock", "*.log", ".DS_Store",
]
MAX_FILE_BYTES = 500_000  # skip anything bigger — almost certainly generated/data, not something worth chunking

LINE_CHUNK_SIZE = 1500
LINE_CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    repo: str
    file_path: str  # relative to the repo root
    text: str  # the header + source, exactly what gets embedded

    def key(self) -> str:
        return f"{self.repo}:{self.file_path}:{hash(self.text) & 0xffffffff:x}"


def _should_skip_file(path: Path) -> bool:
    if any(fnmatch.fnmatch(path.name, pat) for pat in SKIP_FILE_PATTERNS):
        return True
    try:
        if path.stat().st_size > MAX_FILE_BYTES or path.stat().st_size == 0:
            return True
    except OSError:
        return True
    return False


def iter_repo_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if _should_skip_file(path):
            continue
        yield path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None  # binary or unreadable — not a chunkable file


def _chunk_python(repo: str, rel_path: str, source: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _chunk_lines(repo, rel_path, source)

    chunks: list[Chunk] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        segment = ast.get_source_segment(source, node)
        if not segment or not segment.strip():
            continue
        header = f"Repository: {repo}\nFile: {rel_path}\nDefines: {node.name}\n---\n"
        chunks.append(Chunk(repo=repo, file_path=rel_path, text=header + segment))

    if not chunks:
        # a file with no top-level def/class (e.g. pure module-level script) —
        # fall back to line chunking rather than skipping it entirely
        return _chunk_lines(repo, rel_path, source)
    return chunks


def _chunk_lines(repo: str, rel_path: str, source: str) -> list[Chunk]:
    lines = source.splitlines(keepends=True)
    chunks: list[Chunk] = []
    start = 0
    step = LINE_CHUNK_SIZE - LINE_CHUNK_OVERLAP
    while start < len(lines):
        end = min(start + LINE_CHUNK_SIZE, len(lines))
        body = "".join(lines[start:end]).strip()
        if body:
            header = f"Repository: {repo}\nFile: {rel_path}\n---\n"
            chunks.append(Chunk(repo=repo, file_path=rel_path, text=header + body))
        if end >= len(lines):
            break
        start += step
    return chunks


def chunk_file(repo: str, repo_root: Path, path: Path) -> list[Chunk]:
    source = _read_text(path)
    if source is None:
        return []
    rel_path = str(path.relative_to(repo_root))
    if path.suffix == ".py":
        return _chunk_python(repo, rel_path, source)
    return _chunk_lines(repo, rel_path, source)


def chunk_repo(repo: str, repo_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_repo_files(repo_root):
        chunks.extend(chunk_file(repo, repo_root, path))
    return chunks
