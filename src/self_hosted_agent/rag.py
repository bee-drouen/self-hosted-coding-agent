from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import AgentConfig


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".agent_state",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
}


def iter_text_files(root: Path, exclude_dirs: Sequence[str] = DEFAULT_EXCLUDE_DIRS) -> Iterable[Path]:
    exclude_set = set(exclude_dirs)
    for path in root.rglob("*"):
        if any(part in exclude_set for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mov", ".zip"}:
            yield path


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            if overlap > 0:
                overlap_text = "\n".join(current)[-overlap:]
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            else:
                current = []
                current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_index(config: AgentConfig) -> None:
    model = SentenceTransformer(config.embedding_model)
    texts: List[str] = []
    sources: List[str] = []

    roots: List[Path] = []
    for code_dir in config.code_dirs:
        candidate = Path(code_dir)
        if candidate.exists():
            roots.append(candidate)
    if config.context_dir.exists():
        roots.append(config.context_dir)

    if not roots:
        raise FileNotFoundError(
            "No source directories found to index. Configure code_dirs or create a .context directory."
        )

    for root in roots:
        for file_path in iter_text_files(root):
            chunks = chunk_text(
                file_path.read_text(encoding="utf-8", errors="ignore"),
                max_chars=config.chunk_size,
                overlap=config.chunk_overlap,
            )
            texts.extend(chunks)
            sources.extend([str(file_path)] * len(chunks))

    if not texts:
        raise ValueError("No text files found in the context directory.")

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.index_path))

    payload_path = config.store_path
    records: List[str] = []
    for source, text in zip(sources, texts):
        safe_text = text.replace("\n", "\\n")
        records.append(f"{source}\t{safe_text}")
    payload_path.write_text("\n".join(records))


def load_index(config: AgentConfig) -> faiss.IndexFlatL2:
    if not config.index_path.exists():
        raise FileNotFoundError("RAG index missing. Run `self-hosted-agent index` first.")
    return faiss.read_index(str(config.index_path))


def retrieve(config: AgentConfig, query: str, k: int | None = None) -> List[RetrievedChunk]:
    index = load_index(config)
    if not config.store_path.exists():
        raise FileNotFoundError("Stored chunks missing. Rebuild the RAG index.")

    records = config.store_path.read_text().splitlines()
    sources: List[str] = []
    texts: List[str] = []
    for line in records:
        if "\t" not in line:
            continue
        source, text = line.split("\t", 1)
        sources.append(source)
        texts.append(text.replace("\\n", "\n"))

    model = SentenceTransformer(config.embedding_model)
    query_vec = model.encode([query], convert_to_numpy=True)
    search_k = k if k is not None else config.retrieval_k
    scores, indices = index.search(
        np.array(query_vec, dtype=np.float32), search_k
    )

    results: List[RetrievedChunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(texts):
            results.append(
                RetrievedChunk(
                    content=texts[idx],
                    source=sources[idx],
                    score=float(score),
                )
            )
    return results
