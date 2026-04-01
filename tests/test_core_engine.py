"""Tests for SquixEngine — routing, chat_only, plan_only, process_input."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from squix.agents.base import AgentMessage, BaseAgent
from squix.core.session import Session
from squix.memory.manager import MemoryManager
from squix.models.base import ModelResponse
from squix.observability.cost_tracker import CostTracker
from squix.policy.engine import PolicyEngine


def _make_engine(tmp_path):
    """Build engine mocks for testing."""
    from squix.core.engine import SquixEngine

    # Mock registry with a working async chat method
    mock_adapter = MagicMock()
    async def fake_chat(messages, *, temperature=0.7, max_tokens=None):
        return ModelResponse(
            text="mock response",
            input_tokens=10, output_tokens=5,
            model_id="mock", cost=0.001,
        )
    mock_adapter.chat = fake_chat

    mock_registry = MagicMock()
    mock_registry.get_adapter.return_value = mock_adapter
    mock_registry.get_model_ids.return_value = ["mock"]

    eng = SquixEngine(project_dir=tmp_path)
    eng.registry = mock_registry
    eng.cost_tracker = CostTracker()
    eng.policy = PolicyEngine({})
    eng.memory = MemoryManager(
        {"storage_dir": ".squix", "sessions_dir": "sessions"}, tmp_path,
    )
    eng.session = Session(session_id="test1")
    eng.logger = MagicMock()

    # Create the sessions dir so memory.save_session() works
    sessions_dir = tmp_path / ".squix" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    return eng, mock_registry, mock_adapter


class TestEngineChatOnly:
    def test_chat_only_returns_llm_response(self, tmp_path):
        """chat_only calls registry adapter directly and returns result."""
        eng, _, _ = _make_engine(tmp_path)
        eng.agents = {}  # no agents needed for chat_only

        async def _run():
            return await eng.chat_only("hello")

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert len(results) == 1
        assert results[0].metadata["type"] == "chat"
        assert results[0].sender == "squix"

    def test_chat_no_model_returns_error(self, tmp_path):
        """chat_only returns error if no adapter available."""
        eng, reg, _ = _make_engine(tmp_path)
        reg.get_adapter.return_value = None

        async def _run():
            return await eng.chat_only("hi")

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results[0].metadata["type"] == "error"


class TestEnginePlanOnly:
    def test_plan_only_shows_delegation(self, tmp_path):
        """When Talk wants to delegate, plan_only shows a 'would delegate' note."""
        eng, _, _ = _make_engine(tmp_path)

        # Build a talk agent that ALWAYS delegates to "build"
        class DelegateTalk(BaseAgent):
            agent_id = "talk"
            async def handle(self, msg):
                return AgentMessage(
                    sender="talk", recipient="build",
                    content=f"plan: {msg.content}",
                    task_id=msg.task_id,
                    metadata={"type": "delegate"},
                )

        eng.agents = {"talk": DelegateTalk(result_queue=eng._result_queue)}

        async def _run():
            task = asyncio.create_task(eng.agents["talk"].run())
            try:
                return await eng.plan_only("write fib")
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert len(results) >= 1
        assert results[0].metadata.get("plan_only") is True
        assert "build" in results[0].content

    def test_plan_only_passes_through_chat(self, tmp_path):
        """When Talk responds directly, plan_only passes the response through."""
        eng, _, _ = _make_engine(tmp_path)

        class SimpleTalk(BaseAgent):
            agent_id = "talk"
            async def handle(self, msg):
                return AgentMessage(
                    sender="talk", recipient="user",
                    content=f"echo: {msg.content}",
                    task_id=msg.task_id,
                    metadata={"type": "chat"},
                )

        eng.agents = {"talk": SimpleTalk(result_queue=eng._result_queue)}

        async def _run():
            task = asyncio.create_task(eng.agents["talk"].run())
            try:
                return await eng.plan_only("hello")
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert results[0].metadata.get("type") == "chat"


class TestEngineProcessInput:
    def test_process_input_no_talk_returns_error(self, tmp_path):
        """If talk agent is missing, return an error result."""
        eng, _, _ = _make_engine(tmp_path)
        eng.agents = {}  # no agents

        async def _run():
            return await eng.process_input("hi")

        results = asyncio.get_event_loop().run_until_complete(_run())
        assert len(results) == 1
        assert results[0].metadata.get("type") == "error"


class TestEngineLifecycle:
    def test_engine_has_session(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)
        assert eng.session is not None
        assert isinstance(eng.session, Session)

    def test_engine_has_cost_tracker(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)
        assert eng.cost_tracker is not None

    def test_engine_has_result_queue(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)
        assert isinstance(eng._result_queue, asyncio.Queue)

    def test_project_dir_defaults_to_cwd(self):
        from squix.core.engine import SquixEngine
        eng = SquixEngine()
        assert eng.project_dir is not None


class TestEngineRouting:
    def test_route_to_user_puts_in_result_queue(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)

        async def _run():
            msg = AgentMessage(sender="talk", recipient="user",
                               content="hello", task_id="t001",
                               metadata={"type": "chat"})
            await eng._route_message(msg)
            got = await asyncio.wait_for(eng._result_queue.get(), timeout=1.0)
            return got

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.sender == "talk"
        assert result.metadata["type"] == "chat"

    def test_route_to_unknown_agent(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)
        eng.agents = {}

        async def _run():
            msg = AgentMessage(sender="a", recipient="nobody",
                               content="test", task_id="t001")
            await eng._route_message(msg)
            got = await asyncio.wait_for(eng._result_queue.get(), timeout=1.0)
            return got

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.metadata.get("routing_error") is not None

    def test_route_to_agent_sends_message(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)

        class TestAgent(BaseAgent):
            agent_id = "talk"
            async def handle(self, msg):
                return None

        eng.agents = {"talk": TestAgent()}

        async def _run():
            msg = AgentMessage(sender="user", recipient="talk",
                               content="hi", task_id="t001")
            await eng._route_message(msg)
            agent = eng.agents["talk"]
            got = await asyncio.wait_for(agent._inbox.get(), timeout=1.0)
            return got

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result.content == "hi"

    def test_drain_removes_stale_results(self, tmp_path):
        eng, _, _ = _make_engine(tmp_path)

        async def _run():
            # Put some stale results
            await eng._result_queue.put(AgentMessage(
                sender="x", recipient="user",
                content="old", task_id="old1",
                metadata={"type": "chat"},
            ))
            await eng._result_queue.put(AgentMessage(
                sender="x", recipient="user",
                content="older", task_id="old2",
                metadata={"type": "chat"},
            ))
            assert eng._result_queue.qsize() == 2
            # Drain
            while not eng._result_queue.empty():
                try:
                    eng._result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            return eng._result_queue.qsize()

        size = asyncio.get_event_loop().run_until_complete(_run())
        assert size == 0
