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
DEFAULT_CONTEXT_DIR = Path(".context")


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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_endpoint: str = "http://localhost:11434/v1/chat/completions"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
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
        embedding_model=data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        model_endpoint=data.get("model_endpoint", "http://localhost:11434/v1/chat/completions"),
        model_name=data.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3"),
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
        "embedding_model": config.embedding_model,
        "model_endpoint": config.model_endpoint,
        "model_name": config.model_name,
        "auto_shutdown_minutes": config.auto_shutdown_minutes,
        "runpod": config.runpod.__dict__ if config.runpod else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2))
