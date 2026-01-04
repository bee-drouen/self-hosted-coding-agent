from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from . import __version__ as version  # type: ignore
from .chat_session import send_chat
from .config import (
    AgentConfig,
    RunPodSettings,
    DEFAULT_CONFIG_PATH,
    load_config,
    save_config,
)
from .mcp_config import write_default_mcp
from .rag import build_index
from .runpod_manager import (
    RunPodError,
    create_pod,
    get_pod_status,
    record_activity,
    should_shutdown,
    terminate_pod,
)


app = typer.Typer(add_completion=False, help="Self-hosted coding agent CLI.")
console = Console()


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> AgentConfig:
    return load_config(path)


@app.command()
def version_info() -> None:
    """Print the package version."""
    print(version)


@app.command("config")
def init_config(
    api_key: str = typer.Option(..., prompt=True, hide_input=True, help="RunPod API key"),
    template_id: Optional[str] = typer.Option(
        None, help="Optional RunPod template ID for the workstation."
    ),
    image_name: str = typer.Option(
        "runpod/ai-workstation", help="Docker image for the RunPod workspace."
    ),
    machine_type: str = typer.Option(
        "NVIDIA RTX A6000", help="Machine type for the RunPod pod."
    ),
    context_dir: Path = typer.Option(
        Path(".context"),
        help="Directory containing context files injected into the model.",
    ),
) -> None:
    """Initialize the agent config file."""
    runpod_settings = RunPodSettings(
        api_key=api_key,
        template_id=template_id,
        image_name=image_name,
        machine_type=machine_type,
    )
    config = AgentConfig(
        runpod=runpod_settings,
        context_dir=context_dir,
    )
    save_config(config)
    write_default_mcp(config)
    console.print(f"[green]Config saved to {DEFAULT_CONFIG_PATH}[/green]")
    console.print(
        "[blue]MCP servers defined for local terminal and filesystem access.[/blue]"
    )


@app.command()
def index(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Path to the config file."),
) -> None:
    """(Re)build the RAG index from the .context directory."""
    config = _load_config(config_path)
    build_index(config)
    console.print("[green]RAG index built successfully.[/green]")


@app.command()
def launch(
    template_id: Optional[str] = typer.Option(None, help="Override template ID."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config path."),
) -> None:
    """Spin up a RunPod workstation for the agent."""
    config = _load_config(config_path)
    try:
        status = create_pod(config, template_id=template_id)
    except RunPodError as exc:
        typer.echo(f"Failed to create pod: {exc}")
        raise typer.Exit(code=1)

    table = Table(title="RunPod Workspace")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Endpoint URL")
    table.add_column("SSH")
    table.add_row(
        status.id,
        status.status,
        status.endpoint_url or "pending",
        status.ssh_command or "pending",
    )
    console.print(table)


@app.command()
def shutdown(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config path."),
) -> None:
    """Terminate the tracked RunPod pod."""
    config = _load_config(config_path)
    try:
        terminate_pod(config)
    except RunPodError as exc:
        typer.echo(f"Failed to terminate pod: {exc}")
        raise typer.Exit(code=1)
    console.print("[yellow]Pod terminated.[/yellow]")


@app.command()
def status(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config path."),
) -> None:
    """Show the tracked pod status."""
    config = _load_config(config_path)
    try:
        pod_status = get_pod_status(config)
    except RunPodError as exc:
        typer.echo(f"Failed to fetch status: {exc}")
        raise typer.Exit(code=1)
    table = Table(title="RunPod Status")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Endpoint URL")
    table.add_column("SSH Command")
    table.add_row(
        pod_status.id,
        pod_status.status,
        pod_status.endpoint_url or "pending",
        pod_status.ssh_command or "pending",
    )
    console.print(table)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User message."),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config path."),
    disable_rag: bool = typer.Option(False, help="Disable RAG retrieval."),
) -> None:
    """Send a message to the model with local context and store the chat locally."""
    config = _load_config(config_path)
    record_activity(config.base_dir)
    reply = send_chat(config, prompt, include_rag=not disable_rag)
    typer.echo(reply)


@app.command()
def monitor(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, help="Config path."),
) -> None:
    """Check idle timer and shutdown the pod if inactive."""
    config = _load_config(config_path)
    if should_shutdown(config.base_dir, config.auto_shutdown_minutes):
        console.print(
            f"[yellow]No activity since {config.auto_shutdown_minutes} minutes. Shutting down RunPod.[/yellow]"
        )
        try:
            terminate_pod(config)
        except RunPodError as exc:
            typer.echo(f"Failed to terminate pod: {exc}")
            raise typer.Exit(code=1)
    else:
        last = (config.base_dir / "last_activity.txt")
        timestamp = "unknown"
        if last.exists():
            timestamp = datetime.fromisoformat(last.read_text().strip()).isoformat()
        console.print(f"[green]Agent is active. Last activity: {timestamp}[/green]")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
