"""Debugger agent — finds bugs and fixes errors."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class DebuggerAgent(BaseAgent):
    """Finds bugs, fixes errors, and improves code quality."""

    agent_id = "debug"
    role = (
        "Debugger — you analyze errors, find bugs, and suggest fixes. "
        "When given code or an error message, identify the root cause and "
        "propose a concrete fix. Keep explanations clear and brief."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[debug] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
