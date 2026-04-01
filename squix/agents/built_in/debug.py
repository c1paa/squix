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
        self.progress = f"Debugging: {msg.content[:50]}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        self.progress = "Debug complete"
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=response.text,
            task_id=msg.task_id,
            metadata={"type": "work"},
        )
