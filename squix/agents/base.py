"""Base agent class — all agents inherit from this."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from squix.models.base import ModelResponse

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


# Type alias for send function
SendFn = Callable[[AgentMessage], Awaitable[None]]


class BaseAgent(ABC):
    """Base class for all Squix agents.

    Subclasses must implement :meth:`handle` to define the agent's behavior.
    Each agent has a role, an optional model preference list, and knowledge
    of which other agents it can communicate with.
    """

    agent_id: str = "base"
    role: str = ""

    # Type alias for progress callback: (agent_id, text, task_id) -> None
    ProgressCb = Callable[[str, str, str], Awaitable[None]]

    def __init__(
        self,
        agent_id: str | None = None,
        role: str | None = None,
        model_prefers: list[str] | None = None,
        neighbors: list[str] | None = None,
        system_prompt: str | None = None,
        registry: Any = None,
        send_fn: SendFn | None = None,
        result_queue: asyncio.Queue | None = None,
        cost_tracker: Any = None,
        policy: Any = None,
        workspace_manager: Any = None,
        skills: Any = None,
        task_state: Any = None,
        primary_tracker: Any = None,
        progress_cb: ProgressCb | None = None,
    ) -> None:
        self.agent_id = agent_id or self.agent_id
        self.role = role or self.role
        self.model_prefers = model_prefers or []
        self.neighbors = neighbors or []
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.state: AgentState = AgentState.IDLE
        self.progress: str = ""
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._registry = registry
        self._send_fn = send_fn
        self._result_queue = result_queue
        self._cost_tracker = cost_tracker
        self._policy = policy
        self._workspace = workspace_manager
        self._skills = skills
        self._task_state = task_state
        self._primary = primary_tracker
        self._progress_cb = progress_cb
        # Conversation history for current task context
        self._conversation: list[dict[str, str]] = []
        self._current_task_id: str = ""

    async def set_progress(self, text: str, **detail: Any) -> None:
        """Set progress text and optionally emit to UI callback.

        Call this instead of ``self.progress = text`` when you want
        the user to see live progress during task execution.
        """
        self.progress = text
        if self._progress_cb and self._current_task_id:
            try:
                await self._progress_cb(
                    self.agent_id, text, self._current_task_id,
                )
            except Exception as e:
                logger.debug("Progress callback error: %s", e)

    def set_progress_callback(self, cb: Any) -> None:
        """Set the progress callback from the engine.

        Engine calls this during startup to wire live progress updates.
        Callback signature: (agent_id, text, task_id) -> None (can be async).
        """
        self._progress_cb = cb

    async def invoke_skill(
        self,
        skill_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a skill through the skill registry.

        Think → Act → Observe loop: agents call this instead of
        "telling the user to do something".
        """
        if not self._skills:
            return {"status": "error", "error": "Skills not available"}

        params = params or {}
        try:
            result = await self._skills.call(skill_name, params, agent_id=self.agent_id)
            # Record in task state if available
            if self._task_state:
                self._task_state.record_skill(skill_name, params, result, self.agent_id)
            # Emit skill event to result queue for real-time UI
            if self._result_queue:
                skill_meta: dict[str, Any] = {
                    "type": "skill",
                    "skill": skill_name,
                    "agent_id": self.agent_id,
                }
                # Include file path for file operations
                if "path" in params:
                    skill_meta["path"] = params["path"]
                if isinstance(result, dict):
                    skill_meta["result_status"] = result.get("status", "ok")
                    if "content" in result and skill_name == "read_file":
                        skill_meta["lines"] = len(
                            result["content"].splitlines()
                        ) if isinstance(result.get("content"), str) else 0
                await self._result_queue.put(AgentMessage(
                    sender=self.agent_id,
                    recipient="user",
                    content=f"skill:{skill_name}",
                    task_id=self._current_task_id,
                    metadata=skill_meta,
                ))
            return result
        except PermissionError as e:
            logger.warning("Agent %s denied skill %s: %s", self.agent_id, skill_name, e)
            return {"status": "permission_denied", "error": str(e)}
        except Exception as e:
            logger.exception("Agent %s skill %s error", self.agent_id, skill_name)
            return {"status": "error", "error": str(e)}

    async def put_message(self, msg: AgentMessage) -> None:
        """Place a message in this agent's inbox for processing."""
        await self._inbox.put(msg)

    async def run(self) -> None:
        """Main running loop — processes messages from the inbox."""
        self.state = AgentState.IDLE
        while True:
            msg: AgentMessage = await self._inbox.get()
            try:
                self.state = AgentState.WORKING
                self.progress = f"Processing from {msg.sender}"
                # Reset conversation if new task
                if msg.task_id != self._current_task_id:
                    self._conversation = []
                    self._current_task_id = msg.task_id
                response = await self.handle(msg)
                if response:
                    await self._on_response(response)
                self.progress = f"Done: {msg.content[:60]}"
                self.state = AgentState.IDLE
            except Exception as e:
                logger.exception("Agent %s error", self.agent_id)
                self.progress = f"ERROR: {e}"
                self.state = AgentState.ERROR
                # Send error to result queue so engine doesn't hang
                if self._result_queue:
                    error_msg = AgentMessage(
                        sender=self.agent_id,
                        recipient="user",
                        content=f"[{self.agent_id} error] {e}",
                        task_id=msg.task_id,
                        metadata={"type": "error"},
                    )
                    await self._result_queue.put(error_msg)

    @abstractmethod
    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        """Process an incoming message and optionally return a response."""
        ...

    async def _on_response(self, response: AgentMessage) -> None:
        """Route the response to the appropriate destination."""
        if response.recipient == "user":
            # Final result — put in result queue for engine to collect
            if self._result_queue:
                await self._result_queue.put(response)
        elif self._send_fn:
            # Send to another agent via the send function
            await self._send_fn(response)
        elif self._result_queue:
            # Fallback: if no send_fn, put in result queue
            await self._result_queue.put(response)

    async def send_to(self, recipient: str, content: str, task_id: str = "",
                      metadata: dict[str, Any] | None = None) -> None:
        """Helper to send a message to another agent."""
        msg = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            task_id=task_id or self._current_task_id,
            metadata=metadata or {},
        )
        if self._send_fn:
            await self._send_fn(msg)

    async def send_result(self, content: str, task_id: str = "",
                          metadata: dict[str, Any] | None = None) -> None:
        """Helper to send a result back to the user/engine."""
        msg = AgentMessage(
            sender=self.agent_id,
            recipient="user",
            content=content,
            task_id=task_id or self._current_task_id,
            metadata=metadata or {"type": "result"},
        )
        if self._result_queue:
            await self._result_queue.put(msg)

    def _default_system_prompt(self) -> str:
        return (
            f"You are {self.agent_id} in the Squix AI system. "
            f"Your role: {self.role}. "
            f"Stay focused on your role. Be concise."
        )

    async def invoke_llm(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Call the LLM through the registry using model preferences with policy + cost tracking."""
        # Select model via policy or use first preference
        model_id = None
        if self._policy and self._registry:
            model_id = self._policy.select_model_for_agent(
                self.model_prefers, self._registry
            )
        if not model_id:
            model_id = self.model_prefers[0] if self.model_prefers else "default"

        adapter = (
            self._registry.get_adapter(model_id) if self._registry else None
        )
        if adapter is None:
            return ModelResponse(
                text="[no model available]",
                input_tokens=0,
                output_tokens=0,
                model_id=model_id,
                cost=0.0,
            )

        # Try with fallback
        attempt = 0
        last_error = None
        while True:
            try:
                response = await adapter.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Record cost
                if self._cost_tracker:
                    self._cost_tracker.record(
                        model_id=response.model_id,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cost=response.cost,
                        agent_id=self.agent_id,
                    )
                return response
            except Exception as e:
                last_error = e
                attempt += 1
                logger.warning(
                    "Agent %s LLM call failed (attempt %d): %s",
                    self.agent_id, attempt, e,
                )
                # Try fallback
                if self._policy and self._policy.should_fallback(str(e), attempt):
                    next_model = self._policy.next_fallback_model(
                        model_id, self.model_prefers, self._registry
                    )
                    if next_model:
                        model_id = next_model
                        adapter = self._registry.get_adapter(model_id)
                        if adapter:
                            continue
                # No more fallbacks
                return ModelResponse(
                    text=f"[LLM error: {last_error}]",
                    input_tokens=0,
                    output_tokens=0,
                    model_id=model_id,
                    cost=0.0,
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
