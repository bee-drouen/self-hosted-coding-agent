from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import AgentConfig, ensure_base_dir


DEFAULT_MCP_PATH = Path(".agent_state") / "mcp-servers.json"


def default_servers() -> Dict[str, Any]:
    return {
        "servers": {
            "local-shell": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-unix-shell"],
                "env": {"MCP_SERVER_NAME": "local-shell"},
                "disabled": False,
                "capabilities": ["terminal"],
            },
            "local-files": {
                "command": "npx",
                "args": ["@modelcontextprotocol/server-filesystem"],
                "env": {"MCP_SERVER_NAME": "local-files", "ROOT": "."},
                "disabled": False,
                "capabilities": ["fs"],
            },
        }
    }


def write_default_mcp(config: AgentConfig, path: Path = DEFAULT_MCP_PATH) -> Path:
    ensure_base_dir(config.base_dir)
    content = default_servers()
    path.write_text(json.dumps(content, indent=2))
    return path
