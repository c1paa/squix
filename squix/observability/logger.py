"""Structured logging and event emission for Squix."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class SquixLogger:
    """Handles structured logging and rich CLI output."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.show_costs = self.config.get("show_costs", True)
        self.show_agent_status = self.config.get("show_agent_status", True)
        self.log_level = self.config.get("log_level", "INFO")
        self._log_path: Path | None = None

    def configure(self, log_path: Path) -> None:
        """Set the log file path and configure file+console handlers."""
        self._log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path),
            ],
        )

    # ---- High-level Squix events ----

    def system(self, message: str) -> None:
        """Print a system startup/status message."""
        console.print(f"[bold cyan]⚙  {message}[/bold cyan]")

    def task_started(self, task_id: str, text: str) -> None:
        """Print task start banner."""
        console.print(f"\n[bold green]▶  Task {task_id}[/bold green]: {text}")

    def task_completed(self, task_id: str) -> None:
        """Print task completion banner."""
        console.print(f"[bold green]✓  Task {task_id} completed[/bold green]")

    def agent_dispatch(self, agent_id: str, task_text: str) -> None:
        """Print agent dispatch event."""
        if self.show_agent_status:
            console.print(f"  [blue]→ {agent_id}[/blue]: {task_text[:80]}")

    def agent_result(self, agent_id: str, result: str) -> None:
        """Print agent result."""
        if self.show_agent_status:
            preview = result[:120] + "..." if len(result) > 120 else result
            console.print(f"  [dim]← {agent_id}:[/dim] {preview}")

    def event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a structured event to the file logger."""
        logger = logging.getLogger("squix.events")
        logger.info(json.dumps({"type": event_type, "data": data}, default=str))

    def cost(self, model: str, cost: float) -> None:
        """Print cost update."""
        if self.show_costs:
            console.print(f"  [yellow]$  {model}:[/yellow] ${cost:.6f}")

    def error(self, message: str) -> None:
        """Print error."""
        console.print(f"[bold red]✗  ERROR: {message}[/bold red]")

    def warning(self, message: str) -> None:
        """Print warning."""
        console.print(f"[bold yellow]⚠  {message}[/bold yellow]")
