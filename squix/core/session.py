"""Session management — tracks task IDs, progress, and session state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class TaskRecord:
    id: str
    user_input: str
    status: str = "pending"  # pending, running, done, error
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    plan: str | None = None
    results: list[dict] = field(default_factory=list)


@dataclass
class Session:
    """Represents a single Squix session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tasks: list[TaskRecord] = field(default_factory=list)
    tasks_completed: int = 0

    def next_task_id(self) -> str:
        num = len(self.tasks) + 1
        return f"t{num:03d}"

    def add_task(self, task_id: str, user_input: str) -> TaskRecord:
        t = TaskRecord(id=task_id, user_input=user_input, status="running")
        self.tasks.append(t)
        return t

    def complete_task(self, task_id: str) -> None:
        for t in self.tasks:
            if t.id == task_id:
                t.status = "done"
                t.completed_at = datetime.now(UTC).isoformat()
                self.tasks_completed += 1
                break
