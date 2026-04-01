"""Session context — shared memory between all agents within a session.

This is the "glue" that makes agents work as a team instead of isolated LLMs.
Every agent sees what happened before, what files were touched, and what the
user is currently working on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Exchange:
    """A single user<->system exchange."""

    user: str
    response: str
    agent_chain: list[str] = field(default_factory=list)  # e.g. ["talk", "orch", "build"]
    files: list[str] = field(default_factory=list)


class SessionContext:
    """Accumulates context across the entire session.

    Passed to every agent so they know:
    - What the user asked before and what was answered
    - Which files have been worked on
    - What the project is about
    - What tasks were completed
    """

    def __init__(self) -> None:
        self.exchanges: list[Exchange] = []
        self.files_touched: list[str] = []
        self.project_summary: str = ""
        self.active_task_description: str = ""

    # -- Mutation --

    def add_exchange(
        self,
        user_input: str,
        response: str,
        agent_chain: list[str] | None = None,
        files: list[str] | None = None,
    ) -> None:
        ex = Exchange(
            user=user_input,
            response=response[:500],
            agent_chain=agent_chain or [],
            files=files or [],
        )
        self.exchanges.append(ex)
        # Keep last 10 exchanges
        if len(self.exchanges) > 10:
            self.exchanges = self.exchanges[-10:]
        # Track files
        for f in (files or []):
            self.add_file(f)

    def add_file(self, path: str) -> None:
        if path and path not in self.files_touched:
            self.files_touched.append(path)
        # Keep last 20 files
        if len(self.files_touched) > 20:
            self.files_touched = self.files_touched[-20:]

    def set_project_summary(self, summary: str) -> None:
        self.project_summary = summary

    # -- Formatting for agent prompts --

    def format_for_talk(self) -> str:
        """Compact context for the Talk agent (classifier)."""
        parts: list[str] = []

        if self.exchanges:
            parts.append("=== Recent conversation ===")
            for ex in self.exchanges[-5:]:
                parts.append(f"User: {ex.user[:150]}")
                resp_short = ex.response[:150]
                if ex.agent_chain:
                    chain = "->".join(ex.agent_chain)
                    parts.append(f"[{chain}]: {resp_short}")
                else:
                    parts.append(f"Assistant: {resp_short}")
                if ex.files:
                    parts.append(f"  Files: {', '.join(ex.files)}")

        if self.files_touched:
            parts.append(f"\n=== Files worked on ===\n{', '.join(self.files_touched[-10:])}")

        if self.project_summary:
            parts.append(f"\n=== Project ===\n{self.project_summary[:300]}")

        return "\n".join(parts) if parts else ""

    def format_for_worker(self) -> str:
        """Context for worker agents (build, debug, etc.)."""
        parts: list[str] = []

        # Last few exchanges so the worker knows what happened
        if self.exchanges:
            parts.append("=== Session context ===")
            for ex in self.exchanges[-3:]:
                parts.append(f"User asked: {ex.user[:200]}")
                if ex.files:
                    parts.append(f"Files involved: {', '.join(ex.files)}")

        if self.files_touched:
            recent = self.files_touched[-5:]
            parts.append(f"\nRecently touched files: {', '.join(recent)}")

        if self.project_summary:
            parts.append(f"\nProject: {self.project_summary[:200]}")

        return "\n".join(parts) if parts else ""
