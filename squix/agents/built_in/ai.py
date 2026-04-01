"""AI Specialist agent — handles ML, AI, model-related tasks."""

from __future__ import annotations

from squix.agents.base import AgentMessage, BaseAgent


class AISpecialistAgent(BaseAgent):
    """Handles ML, AI, model-selection, and neural network related tasks."""

    agent_id = "AI"
    role = (
        "AI Specialist — you handle machine learning, AI, model architecture, "
        "and related technical questions. Provide expert-level advice on AI topics."
    )

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[ai] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
