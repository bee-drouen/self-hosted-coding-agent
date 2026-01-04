from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import requests

from .config import AgentConfig
from .rag import RetrievedChunk, retrieve


Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )


def _chat_log_path(config: AgentConfig) -> Path:
    config.chat_log_path.parent.mkdir(parents=True, exist_ok=True)
    return config.chat_log_path


def load_history(config: AgentConfig) -> List[Message]:
    path = _chat_log_path(config)
    if not path.exists():
        return []
    history: List[Message] = []
    for line in path.read_text().splitlines():
        data = json.loads(line)
        retrieved_chunks = [
            RetrievedChunk(**chunk) for chunk in data.get("retrieved", [])
        ]
        history.append(
            Message(
                role=data["role"],
                content=data["content"],
                retrieved=retrieved_chunks,
                timestamp=data.get("timestamp", ""),
            )
        )
    return history


def append_history(config: AgentConfig, message: Message) -> None:
    path = _chat_log_path(config)
    with path.open("a") as fp:
        fp.write(
            json.dumps(
                {
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp,
                    "retrieved": [chunk.__dict__ for chunk in message.retrieved],
                }
            )
            + "\n"
        )


def build_prompt(
    config: AgentConfig, user_input: str, include_rag: bool
) -> tuple[list[dict], List[RetrievedChunk]]:
    history = load_history(config)
    messages = [{"role": msg.role, "content": msg.content} for msg in history]

    retrieved: List[RetrievedChunk] = []
    if include_rag:
        retrieved = retrieve(config, user_input)
        if retrieved:
            context_block = "\n\n".join(
                [f"{chunk.source}:\n{chunk.content}" for chunk in retrieved]
            )
            messages.append(
                {
                    "role": "system",
                    "content": f"Context from workspace files:\n{context_block}",
                }
            )
    messages.append({"role": "user", "content": user_input})
    return messages, retrieved


def send_chat(
    config: AgentConfig, user_input: str, include_rag: bool = True
) -> str:
    messages, retrieved = build_prompt(config, user_input, include_rag)
    payload = {
        "model": config.model_name,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(config.model_endpoint, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "No response returned from model.")
    )

    append_history(config, Message(role="user", content=user_input, retrieved=retrieved))
    append_history(config, Message(role="assistant", content=content))
    return content
