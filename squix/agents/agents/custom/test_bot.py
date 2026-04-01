from squix.agents.base import BaseAgent, AgentMessage


class TestbotAgent(BaseAgent):
    agent_id = "test_bot"
    role = "Test bot"

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": msg.content},
        ]
        return AgentMessage(
            sender=self.agent_id,
            recipient="orch",
            content=f"[test_bot] {msg.content}",
            task_id=msg.task_id,
            metadata={"type": "work", "llm_messages": messages},
        )
