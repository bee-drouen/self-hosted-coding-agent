from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import json
import requests

from .config import AgentConfig, RunPodSettings, ensure_base_dir, save_config


RUNPOD_URL = "https://api.runpod.io/graphql"
LAST_ACTIVITY_FILE = "last_activity.txt"


class RunPodError(RuntimeError):
    pass


@dataclass
class PodStatus:
    id: str
    status: str
    endpoint_url: Optional[str]
    ssh_command: Optional[str]


def _headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _post(api_key: str, query: str, variables: dict | None = None) -> dict:
    payload = {"query": query, "variables": variables or {}}
    response = requests.post(
        RUNPOD_URL, headers=_headers(api_key), json=payload, timeout=60
    )
    if response.status_code >= 400:
        raise RunPodError(f"RunPod API error: {response.status_code} {response.text}")
    data = response.json()
    if "errors" in data:
        raise RunPodError(f"RunPod GraphQL error: {data['errors']}")
    return data["data"]


def create_pod(config: AgentConfig, template_id: str | None = None) -> PodStatus:
    if not config.runpod:
        raise RunPodError("RunPod settings missing from config.")

    settings = config.runpod
    mutation = """
    mutation PodCreate($input: PodCreateInput!){
      podCreate(input: $input){
        id
        status
        name
        imageName
        machineId
        ports{
          ip
          isIpPublic
          privatePort
          publicPort
        }
        rpcProxy{
          url
        }
      }
    }
    """
    variables = {
        "input": {
            "templateId": template_id or settings.template_id,
            "name": "self-hosted-coding-agent",
            "cloudType": settings.cloud_type,
            "machineType": settings.machine_type,
            "imageName": settings.image_name,
        }
    }
    result = _post(settings.api_key, mutation, variables)
    pod = result["podCreate"]
    endpoint_url = None
    if pod.get("rpcProxy"):
        endpoint_url = pod["rpcProxy"].get("url")
    ssh_command = None
    for port in pod.get("ports", []):
        if port.get("isIpPublic"):
            ssh_command = f"ssh root@{port['ip']} -p {port['publicPort']}"
            break

    settings.pod_id = pod["id"]
    settings.endpoint_url = endpoint_url
    settings.ssh_command = ssh_command
    save_config(config)
    _touch_activity(config.base_dir)
    return PodStatus(
        id=pod["id"],
        status=pod["status"],
        endpoint_url=endpoint_url,
        ssh_command=ssh_command,
    )


def terminate_pod(config: AgentConfig) -> None:
    if not config.runpod or not config.runpod.pod_id:
        raise RunPodError("No pod is currently tracked in the config.")
    mutation = """
    mutation PodTerminate($podId: String!){
      podTerminate(input:{podId:$podId})
    }
    """
    _post(config.runpod.api_key, mutation, {"podId": config.runpod.pod_id})
    config.runpod.pod_id = None
    config.runpod.endpoint_url = None
    config.runpod.ssh_command = None
    save_config(config)


def get_pod_status(config: AgentConfig) -> PodStatus:
    if not config.runpod or not config.runpod.pod_id:
        raise RunPodError("No pod is currently tracked in the config.")
    query = """
    query PodFind($podId: String!){
      pod(input:{podId:$podId}){
        id
        status
        imageName
        machineId
        ports{
          ip
          isIpPublic
          privatePort
          publicPort
        }
        rpcProxy{
          url
        }
      }
    }
    """
    data = _post(config.runpod.api_key, query, {"podId": config.runpod.pod_id})
    pod = data["pod"]
    endpoint_url = None
    if pod.get("rpcProxy"):
        endpoint_url = pod["rpcProxy"].get("url")
    ssh_command = None
    for port in pod.get("ports", []):
        if port.get("isIpPublic"):
            ssh_command = f"ssh root@{port['ip']} -p {port['publicPort']}"
            break
    return PodStatus(
        id=pod["id"],
        status=pod["status"],
        endpoint_url=endpoint_url,
        ssh_command=ssh_command,
    )


def record_activity(base_dir: Path) -> None:
    _touch_activity(base_dir)


def should_shutdown(base_dir: Path, idle_minutes: int) -> bool:
    activity_path = base_dir / LAST_ACTIVITY_FILE
    if not activity_path.exists():
        return False
    last = datetime.fromisoformat(activity_path.read_text().strip())
    return datetime.utcnow() - last > timedelta(minutes=idle_minutes)


def _touch_activity(base_dir: Path) -> None:
    ensure_base_dir(base_dir)
    (base_dir / LAST_ACTIVITY_FILE).write_text(datetime.utcnow().isoformat())
