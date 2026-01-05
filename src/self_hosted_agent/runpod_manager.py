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
    
    # Updated mutation to match current RunPod API
    mutation = """
    mutation {
      podFindAndDeployOnDemand(
        input: {
          cloudType: SECURE
          gpuCount: 1
          volumeInGb: 50
          containerDiskInGb: 50
          minVcpuCount: 2
          minMemoryInGb: 15
          gpuTypeId: "%s"
          name: "self-hosted-coding-agent"
          imageName: "%s"
          dockerArgs: ""
          ports: "8888/http,22/tcp"
          volumeMountPath: "/workspace"
          env: [
            {key: "JUPYTER_PASSWORD", value: "runpod"}
          ]
        }
      ) {
        id
        imageName
        env
        machineId
        machine {
          podHostId
        }
      }
    }
    """ % (settings.machine_type, settings.image_name)
    
    # Note: The new API requires gpuTypeId instead of machineType
    # Common GPU type IDs:
    # "NVIDIA RTX A6000" -> you'll need to get the actual ID
    # You may need to query available GPU types first
    
    result = _post(settings.api_key, mutation, {})
    pod = result["podFindAndDeployOnDemand"]
    
    # Wait a moment for the pod to initialize
    import time
    time.sleep(2)
    
    # Get the full pod details
    pod_details = get_pod_status_by_id(config, pod["id"])
    
    settings.pod_id = pod["id"]
    settings.endpoint_url = pod_details.endpoint_url
    settings.ssh_command = pod_details.ssh_command
    save_config(config)
    _touch_activity(config.base_dir)
    
    return pod_details


def terminate_pod(config: AgentConfig) -> None:
    if not config.runpod or not config.runpod.pod_id:
        raise RunPodError("No pod is currently tracked in the config.")
    
    mutation = """
    mutation {
      podTerminate(input: {podId: "%s"})
    }
    """ % config.runpod.pod_id
    
    _post(config.runpod.api_key, mutation, {})
    config.runpod.pod_id = None
    config.runpod.endpoint_url = None
    config.runpod.ssh_command = None
    save_config(config)


def get_pod_status(config: AgentConfig) -> PodStatus:
    if not config.runpod or not config.runpod.pod_id:
        raise RunPodError("No pod is currently tracked in the config.")
    return get_pod_status_by_id(config, config.runpod.pod_id)


def get_pod_status_by_id(config: AgentConfig, pod_id: str) -> PodStatus:
    if not config.runpod:
        raise RunPodError("RunPod settings missing from config.")
    
    query = """
    query {
      pod(input: {podId: "%s"}) {
        id
        name
        runtime {
          uptimeInSeconds
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }
          gpus {
            id
            gpuUtilPercent
            memoryUtilPercent
          }
          container {
            cpuPercent
            memoryPercent
          }
        }
        machineId
        machine {
          gpuDisplayName
        }
      }
    }
    """ % pod_id
    
    data = _post(config.runpod.api_key, query, {})
    pod = data["pod"]
    
    endpoint_url = None
    ssh_command = None
    
    if pod.get("runtime") and pod["runtime"].get("ports"):
        for port in pod["runtime"]["ports"]:
            if port.get("isIpPublic"):
                if port.get("privatePort") == 8888:
                    endpoint_url = f"http://{port['ip']}:{port['publicPort']}"
                elif port.get("privatePort") == 22:
                    ssh_command = f"ssh root@{port['ip']} -p {port['publicPort']}"
    
    status = "running" if pod.get("runtime") else "pending"
    
    return PodStatus(
        id=pod["id"],
        status=status,
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
