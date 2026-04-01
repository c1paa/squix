"""Structured logging — all logs go to FILE only. CLI output is explicit."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("squix.events")


class SquixLogger:
    """All structured logging → FILE only. No console prints.
    The CLI itself decides what to show the user.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.show_costs = self.config.get("show_costs", True)
        self.show_agent_status = self.config.get("show_agent_status", True)
        self.log_level = self.config.get("log_level", "INFO")
        self._log_path: Path | None = None
        self._configured = False

    def configure(self, log_path: Path) -> None:
        """Set up file-only logging. Nothing goes to stdout."""
        if self._configured:
            return
        self._log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_path),
            ],
            force=True,
        )
        self._configured = True

    # ---- Everything goes to FILE only ----

    def system(self, message: str) -> None:
        logger.info(f"SYSTEM: {message}")

    def task_started(self, task_id: str, text: str) -> None:
        logger.info(f"TASK_START: {task_id} | {text}")

    def task_completed(self, task_id: str) -> None:
        logger.info(f"TASK_DONE: {task_id}")

    def agent_dispatch(self, agent_id: str, task_text: str) -> None:
        logger.info(f"DISPATCH: {agent_id} | {task_text[:200]}")

    def agent_result(self, agent_id: str, result: str) -> None:
        logger.info(f"RESULT: {agent_id} | {result[:300]}")

    def event(self, event_type: str, data: dict[str, Any]) -> None:
        logger.info(json.dumps({"type": event_type, "data": data}, default=str))

    def cost(self, model: str, cost: float) -> None:
        logger.info(f"COST: {model} | ${cost:.6f}")

    def error(self, message: str) -> None:
        logger.error(f"ERROR: {message}")

    def warning(self, message: str) -> None:
        logger.warning(f"WARNING: {message}")
