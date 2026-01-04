from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import AgentConfig


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float


def iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mov", ".zip"}:
            yield path


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
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

    if not config.context_dir.exists():
        raise FileNotFoundError(
            f"Context directory {config.context_dir} not found. Add files to .context first."
        )

    for file_path in iter_text_files(config.context_dir):
        chunks = chunk_text(file_path.read_text(encoding="utf-8", errors="ignore"))
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


def retrieve(config: AgentConfig, query: str, k: int = 4) -> List[RetrievedChunk]:
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
    scores, indices = index.search(np.array(query_vec, dtype=np.float32), k)

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
