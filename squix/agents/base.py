"""Base agent class — all agents inherit from this."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("squix.agent")


class AgentState(StrEnum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    DONE = "done"


@dataclass
class AgentMessage:
    """A message sent between agents."""

    sender: str
    recipient: str
    content: str
    task_id: str = ""
    reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all Squix agents.

    Subclasses must implement :meth:`handle` to define the agent's behavior.
    Each agent has a role, an optional model preference list, and knowledge
    of which other agents it can communicate with.
    """

    agent_id: str = "base"
    role: str = ""

    def __init__(
        self,
        agent_id: str | None = None,
        role: str | None = None,
        model_prefers: list[str] | None = None,
        neighbors: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.agent_id = agent_id or self.agent_id
        self.role = role or self.role
        self.model_prefers = model_prefers or []
        self.neighbors = neighbors or []
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.state: AgentState = AgentState.IDLE
        self.progress: str = ""
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()

    async def put_message(self, msg: AgentMessage) -> None:
        """Place a message in this agent's inbox for processing."""
        await self._inbox.put(msg)

    async def run(self) -> None:
        """Main running loop — processes messages from the inbox."""
        self.state = AgentState.WORKING
        while True:
            msg: AgentMessage = await self._inbox.get()
            try:
                self.progress = f"Processing message from {msg.sender}"
                response = await self.handle(msg)
                if response:
                    await self._on_response(response)
                self.progress = f"Done: {msg.content[:60]}"
                self.state = AgentState.IDLE
            except Exception as e:
                logger.exception("Agent %s error", self.agent_id)
                self.progress = f"ERROR: {e}"
                self.state = AgentState.ERROR

    @abstractmethod
    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        """Process an incoming message and optionally return a response."""
        ...

    async def _on_response(self, response: AgentMessage) -> None:
        """Hook for subclasses to react after producing a response."""
        raise NotImplementedError

    def _default_system_prompt(self) -> str:
        return (
            f"You are {self.agent_id} in the Squix AI system. "
            f"Your role: {self.role}. "
            f"Stay focused on your role. Be concise."
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent state for memory persistence."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model_prefers": self.model_prefers,
            "neighbors": self.neighbors,
            "state": self.state.value,
            "progress": self.progress,
            "system_prompt": self.system_prompt,
        }
