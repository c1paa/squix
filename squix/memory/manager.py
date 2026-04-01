"""Memory manager — persists and restores session state, agent states, and task history."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from squix.core.session import Session

logger = logging.getLogger("squix.memory")


class MemoryManager:
    """Manages persistence and restoration of Squix state."""

    def __init__(self, config: dict[str, Any], project_dir: Path) -> None:
        self.storage_dir = project_dir / config.get("storage_dir", ".squix")
        self.session_dir = project_dir / config.get("session_dir", ".squix/sessions")
        self.agents_file = self.storage_dir / "agents.json"
        self.history_file = self.storage_dir / "history.json"

    def init(self) -> None:
        """Create storage directories."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    async def create_session(self) -> Session:
        """Create a new session and persist it."""
        self.init()  # Ensure directories exist
        session = Session()
        self._save_session_file(session)
        logger.info("Created new session: %s", session.session_id)
        return session

    async def load_session(self) -> Session | None:
        """Load the most recent session from disk."""
        sessions = sorted(
            self.session_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            return None

        try:
            with open(sessions[0]) as f:
                data = json.load(f)
            session = Session(
                session_id=data["session_id"],
                created_at=data.get("created_at", ""),
                tasks_completed=data.get("tasks_completed", 0),
            )
            # Restore tasks
            for t in data.get("tasks", []):
                from squix.core.session import TaskRecord
                session.tasks.append(TaskRecord(**t))
            return session
        except Exception:
            logger.exception("Failed to load session")
            return None

    async def save_session(self, session: Session) -> None:
        """Persist current session state."""
        self._save_session_file(session)

    def _save_session_file(self, session: Session) -> None:
        path = self.session_dir / f"session_{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "tasks": [
                {
                    "id": t.id,
                    "user_input": t.user_input,
                    "status": t.status,
                    "created_at": t.created_at,
                    "completed_at": t.completed_at,
                    "plan": t.plan,
                    "results": t.results,
                }
                for t in session.tasks
            ],
            "tasks_completed": session.tasks_completed,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    async def save_agent_state(self, agent_id: str, state: dict[str, Any]) -> None:
        """Save an individual agent's internal state."""
        agents_path = self.storage_dir / "agent_states.json"
        data = {}
        if agents_path.exists():
            try:
                with open(agents_path) as f:
                    data = json.load(f)
            except Exception:
                pass
        data[agent_id] = state
        with open(agents_path, "w") as f:
            json.dump(data, f, indent=2)

    async def load_agent_state(self, agent_id: str) -> dict[str, Any] | None:
        """Load an agent's previously saved state."""
        agents_path = self.storage_dir / "agent_states.json"
        if not agents_path.exists():
            return None
        try:
            with open(agents_path) as f:
                data = json.load(f)
            return data.get(agent_id)
        except Exception:
            return None

    async def append_history(self, entry: dict[str, Any]) -> None:
        """Append an entry to the persistent history log."""
        entries = []
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    entries = json.load(f)
            except Exception:
                pass
        entries.append(entry)
        with open(self.history_file, "w") as f:
            json.dump(entries, f, indent=2)
