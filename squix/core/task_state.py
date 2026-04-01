"""TaskState — shared execution state for the entire agent pipeline.

Every task gets a TaskState that is updated by every agent that touches it.
This is the central nervous system: who is working, what has been done,
which files were read/written, what decisions were made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Handoff:
    """Record of an agent passing work to another."""
    from_agent: str
    to_agent: str
    summary: str
    ts: float = field(default_factory=time.time)


@dataclass
class TaskState:
    """Mutable state for a single task execution.

    This is updated by EVERY agent that participates in the task.
    """
    task_id: str = ""
    user_input: str = ""
    status: str = "running"  # running, done, error

    # Pipeline
    handoffs: list[Handoff] = field(default_factory=list)
    steps_done: list[str] = field(default_factory=list)
    current_agent: str = ""

    # Workspace
    primary_file: str = ""
    files_read: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)

    # Skill calls (audit trail)
    skill_calls: list[dict[str, Any]] = field(default_factory=list)

    # Notes / decisions
    observations: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Final result
    result_summary: str = ""

    # Timing
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    # ── Mutators ────────────────────────────────────────────────────────

    def record_handoff(self, from_agent: str, to_agent: str, summary: str = "") -> None:
        self.handoffs.append(Handoff(from_agent, to_agent, summary or f"{from_agent} → {to_agent}"))
        self.current_agent = to_agent
        self.steps_done.append(f"HANDOFF: {from_agent} → {to_agent}")

    def record_skill(self, skill: str, params: dict[str, Any], result: Any, agent: str = "") -> None:
        self.skill_calls.append({
            "skill": skill,
            "params": {k: (str(v)[:100] if isinstance(v, str) and len(v) > 100 else v)
                       for k, v in params.items()},
            "agent": agent,
            "ts": time.time(),
        })

    def record_file_read(self, path: str) -> None:
        if path not in self.files_read:
            self.files_read.append(path)

    def record_file_written(self, path: str, created: bool = False) -> None:
        if path not in self.files_modified:
            self.files_modified.append(path)
        if created and path not in self.files_created:
            self.files_created.append(path)

    def record_observation(self, text: str) -> None:
        self.observations.append(text)

    def record_error(self, text: str) -> None:
        self.errors.append(text)

    def finish(self, summary: str = "") -> None:
        self.result_summary = summary or self.result_summary
        self.status = "done"
        self.finished_at = time.time()

    def fail(self, reason: str) -> None:
        self.errors.append(reason)
        self.status = "error"
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logging."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "current_agent": self.current_agent,
            "primary_file": self.primary_file,
            "handoffs": [f"{h.from_agent}→{h.to_agent}" for h in self.handoffs],
            "files_read": self.files_read[:10],
            "files_modified": self.files_modified[:10],
            "skill_calls": [f"{s['agent'] or '?'}:{s['skill']}" for s in self.skill_calls[-10:]],
            "errors": self.errors[-5:],
            "summary": self.result_summary[:300],
            "elapsed": round(self.finished_at - self.started_at, 1) if self.finished_at else None,
        }
