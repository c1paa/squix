"""README/Documentation agent — writes docs and guides."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class ReadmeAgent(BaseAgent):
    """Writes README, guides, and user documentation."""

    agent_id = "README"
    role = (
        "Documentation Writer — you write README files, user guides, API docs, "
        "and help documentation. Produce clear, well-structured markdown."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[readme] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
