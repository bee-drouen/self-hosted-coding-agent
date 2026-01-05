from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
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

    repo_root = Path(".").resolve()
    git_files = _git_tracked_files(repo_root)
    source_files: List[Path] = []
    exclude_set = set(DEFAULT_EXCLUDE_DIRS)
    for rel_path in git_files:
        full_path = repo_root / rel_path
        if any(part in exclude_set for part in full_path.parts):
            continue
        if full_path.is_file() and full_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mov", ".zip"}:
            source_files.append(full_path)

    if config.context_dir.exists():
        source_files.extend(list(iter_text_files(config.context_dir)))

    if not source_files:
        raise FileNotFoundError(
            "No source files found to index. Ensure this is a git repository and files are tracked."
        )

    previous_manifest, previous_embeddings = _load_previous_state(config)
    previous_lookup = _build_manifest_lookup(previous_manifest)

    texts: List[str] = []
    sources: List[str] = []
    embeddings_list: List[np.ndarray] = []
    bm25_corpus: List[List[str]] = []

    for file_path in source_files:
        file_hash = _hash_file(file_path)
        chunk_size, chunk_overlap = _chunk_params_for_path(file_path, config)
        chunks = chunk_text(
            file_path.read_text(encoding="utf-8", errors="ignore"),
            max_chars=chunk_size,
            overlap=chunk_overlap,
        )
        embedding_inputs = [f"{file_path}:\n{chunk}" for chunk in chunks]

        reuse_slice = _reuse_slice(previous_lookup, file_path, file_hash, len(chunks))
        if reuse_slice and previous_embeddings is not None:
            start, count = reuse_slice
            embeddings = previous_embeddings[start : start + count]
        else:
            embeddings = model.encode(
                embedding_inputs, convert_to_numpy=True, show_progress_bar=False
            )

        texts.extend(chunks)
        sources.extend([str(file_path)] * len(chunks))
        embeddings_list.append(np.array(embeddings, dtype=np.float32))
        bm25_corpus.extend([chunk.split() for chunk in chunks])

    if not texts:
        raise ValueError("No text files found in the context directory.")

    all_embeddings = np.vstack(embeddings_list)
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.index_path))

    payload_path = config.store_path
    records: List[str] = []
    for source, text in zip(sources, texts):
        safe_text = text.replace("\n", "\\n")
        records.append(f"{source}\t{safe_text}")
    payload_path.write_text("\n".join(records))

    np.save(config.embeddings_path, all_embeddings)
    manifest_json = _build_manifest(config, source_files)
    config.manifest_path.write_text(manifest_json)

    bm25 = BM25Okapi(bm25_corpus)
    with open(config.bm25_path, "wb") as fp:
        pickle.dump(bm25, fp)


def load_index(config: AgentConfig) -> faiss.IndexFlatL2:
    if not config.index_path.exists():
        raise FileNotFoundError("RAG index missing. Run `self-hosted-agent index` first.")
    return faiss.read_index(str(config.index_path))


def retrieve(config: AgentConfig, query: str, k: int | None = None) -> List[RetrievedChunk]:
    index = load_index(config)
    if not config.store_path.exists():
        raise FileNotFoundError("Stored chunks missing. Rebuild the RAG index.")
    if not config.bm25_path.exists():
        raise FileNotFoundError("BM25 index missing. Rebuild the RAG index.")

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

    dense_scores, dense_indices = index.search(
        np.array(query_vec, dtype=np.float32), search_k * 2
    )
    dense_dict = {
        int(idx): 1.0 / (1.0 + float(score))
        for idx, score in zip(dense_indices[0], dense_scores[0])
        if idx < len(texts)
    }

    with open(config.bm25_path, "rb") as fp:
        bm25: BM25Okapi = pickle.load(fp)
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][: search_k * 2]
    bm25_dict = {
        int(idx): float(bm25_scores[idx])
        for idx in bm25_top_idx
        if idx < len(texts)
    }

    candidates = set(dense_dict.keys()) | set(bm25_dict.keys())
    if not candidates:
        return []

    dense_max = max(dense_dict.values()) if dense_dict else 1.0
    bm25_max = max(bm25_dict.values()) if bm25_dict else 1.0

    combined: List[tuple[float, int]] = []
    for idx in candidates:
        d_score = dense_dict.get(idx, 0.0) / dense_max
        b_score = bm25_dict.get(idx, 0.0) / bm25_max
        score = config.dense_weight * d_score + config.bm25_weight * b_score
        combined.append((score, idx))

    combined.sort(key=lambda x: x[0], reverse=True)
    top = combined[:search_k]

    results: List[RetrievedChunk] = []
    for score, idx in top:
        if idx < len(texts):
            results.append(
                RetrievedChunk(
                    content=texts[idx],
                    source=sources[idx],
                    score=float(score),
                )
            )
    return results


def _git_tracked_files(repo_root: Path) -> List[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to list git-tracked files: {result.stderr.strip() or result.stdout.strip()}"
        )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    sha.update(path.read_bytes())
    return sha.hexdigest()


def _load_previous_state(config: AgentConfig):
    if not (config.manifest_path.exists() and config.embeddings_path.exists()):
        return None, None
    manifest_data = json.loads(config.manifest_path.read_text())
    if (
        manifest_data.get("embedding_model") != config.embedding_model
        or manifest_data.get("chunk_size") != config.chunk_size
        or manifest_data.get("chunk_overlap") != config.chunk_overlap
    ):
        return None, None
    embeddings = np.load(config.embeddings_path)
    return manifest_data, embeddings


def _build_manifest_lookup(manifest: dict | None):
    lookup = {}
    if not manifest:
        return lookup
    for file_entry in manifest.get("files", []):
        lookup[file_entry["path"]] = (
            file_entry["start"],
            file_entry["count"],
            file_entry["hash"],
        )
    return lookup


def _reuse_slice(lookup, path: Path, file_hash: str, chunk_count: int):
    entry = lookup.get(str(path))
    if not entry:
        return None
    start, count, prev_hash = entry
    if prev_hash == file_hash and count == chunk_count:
        return start, count
    return None


def _build_manifest(config: AgentConfig, files: List[Path]) -> str:
    manifest_files = []
    offset = 0
    for file_path in files:
        chunk_size, chunk_overlap = _chunk_params_for_path(file_path, config)
        chunks = chunk_text(
            file_path.read_text(encoding="utf-8", errors="ignore"),
            max_chars=chunk_size,
            overlap=chunk_overlap,
        )
        manifest_files.append(
            {
                "path": str(file_path),
                "hash": _hash_file(file_path),
                "start": offset,
                "count": len(chunks),
            }
        )
        offset += len(chunks)
    manifest = {
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
        "dense_weight": config.dense_weight,
        "bm25_weight": config.bm25_weight,
        "total_chunks": offset,
        "files": manifest_files,
    }
    return json.dumps(manifest, indent=2)


def _chunk_params_for_path(path: Path, config: AgentConfig) -> tuple[int, int]:
    ext = path.suffix.lower()
    if ext == ".md":
        return 900, 200
    if ext == ".tsx":
        return 1800, 250
    if ext == ".ts":
        return 1200, 200
    return config.chunk_size, config.chunk_overlap
