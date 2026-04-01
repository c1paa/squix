"""Builder agent — writes code and implements solutions."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class BuilderAgent(BaseAgent):
    """Writes code, implements features, creates files based on tasks."""

    agent_id = "build"
    role = (
        "Builder — you write code, implement solutions, and create artifacts. "
        "Focus on correctness and clarity. When asked to write code, produce "
        "complete, runnable code. Explain your approach briefly."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        self.progress = f"Building: {msg.content[:50]}"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        response = await self.invoke_llm(messages)
        self.progress = "Build complete"
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=response.text,
            task_id=msg.task_id,
            metadata={"type": "work"},
        )
