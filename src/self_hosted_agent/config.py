from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json


DEFAULT_BASE_DIR = Path(".agent_state")
DEFAULT_CONFIG_PATH = DEFAULT_BASE_DIR / "config.json"
DEFAULT_INDEX_PATH = DEFAULT_BASE_DIR / "rag.index"
DEFAULT_STORE_PATH = DEFAULT_BASE_DIR / "documents.jsonl"
DEFAULT_CHAT_LOG_PATH = DEFAULT_BASE_DIR / "chat_history.jsonl"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_BASE_DIR / "embeddings.npy"
DEFAULT_MANIFEST_PATH = DEFAULT_BASE_DIR / "manifest.json"
DEFAULT_BM25_PATH = DEFAULT_BASE_DIR / "bm25.pkl"
DEFAULT_CONTEXT_DIR = Path(".context")
DEFAULT_SYSTEM_PROMPT = (
    "You are a meticulous coding agent assisting on a large monorepo. "
    "Use only the provided context and prior chat history. "
    "Cite file paths when referencing code, reason step-by-step, and avoid speculation. "
    "Resolve workspace aliases (e.g., @services/foo/bar) to real repository paths when reasoning about handlers or imports."
)


@dataclass
class RunPodSettings:
    api_key: str
    template_id: Optional[str] = None
    image_name: str = "runpod/ai-workstation"
    machine_type: str = "NVIDIA RTX A6000"
    cloud_type: str = "SECURE"
    idle_timeout_minutes: int = 30
    pod_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    ssh_command: Optional[str] = None


@dataclass
class AgentConfig:
    base_dir: Path = DEFAULT_BASE_DIR
    context_dir: Path = DEFAULT_CONTEXT_DIR
    index_path: Path = DEFAULT_INDEX_PATH
    store_path: Path = DEFAULT_STORE_PATH
    chat_log_path: Path = DEFAULT_CHAT_LOG_PATH
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    bm25_path: Path = DEFAULT_BM25_PATH
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 8
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    model_endpoint: str = "http://localhost:11434/v1/chat/completions"
    model_name: str = "qwen2.5-coder-32b-instruct"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    auto_shutdown_minutes: int = 30
    runpod: Optional[RunPodSettings] = None


def ensure_base_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AgentConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. Run `self-hosted-agent config init` first."
        )

    data = json.loads(path.read_text())
    runpod_settings = (
        RunPodSettings(**data["runpod"]) if data.get("runpod") else None
    )
    return AgentConfig(
        base_dir=Path(data.get("base_dir", DEFAULT_BASE_DIR)),
        context_dir=Path(data.get("context_dir", DEFAULT_CONTEXT_DIR)),
        index_path=Path(data.get("index_path", DEFAULT_INDEX_PATH)),
        store_path=Path(data.get("store_path", DEFAULT_STORE_PATH)),
        chat_log_path=Path(data.get("chat_log_path", DEFAULT_CHAT_LOG_PATH)),
        embeddings_path=Path(data.get("embeddings_path", DEFAULT_EMBEDDINGS_PATH)),
        manifest_path=Path(data.get("manifest_path", DEFAULT_MANIFEST_PATH)),
        bm25_path=Path(data.get("bm25_path", DEFAULT_BM25_PATH)),
        chunk_size=data.get("chunk_size", 1200),
        chunk_overlap=data.get("chunk_overlap", 200),
        retrieval_k=data.get("retrieval_k", 8),
        dense_weight=data.get("dense_weight", 0.6),
        bm25_weight=data.get("bm25_weight", 0.4),
        embedding_model=data.get("embedding_model", "jinaai/jina-embeddings-v2-base-code"),
        model_endpoint=data.get("model_endpoint", "http://localhost:11434/v1/chat/completions"),
        model_name=data.get("model_name", "qwen2.5-coder-32b-instruct"),
        system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        temperature=data.get("temperature", 0.2),
        top_p=data.get("top_p", 0.9),
        max_tokens=data.get("max_tokens", 2048),
        auto_shutdown_minutes=data.get("auto_shutdown_minutes", 30),
        runpod=runpod_settings,
    )


def save_config(config: AgentConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    ensure_base_dir(config.base_dir)
    serializable = {
        "base_dir": str(config.base_dir),
        "context_dir": str(config.context_dir),
        "index_path": str(config.index_path),
        "store_path": str(config.store_path),
        "chat_log_path": str(config.chat_log_path),
        "embeddings_path": str(config.embeddings_path),
        "manifest_path": str(config.manifest_path),
        "bm25_path": str(config.bm25_path),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
        "dense_weight": config.dense_weight,
        "bm25_weight": config.bm25_weight,
        "embedding_model": config.embedding_model,
        "model_endpoint": config.model_endpoint,
        "model_name": config.model_name,
        "system_prompt": config.system_prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "auto_shutdown_minutes": config.auto_shutdown_minutes,
        "runpod": config.runpod.__dict__ if config.runpod else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))
