"""Tests for AgentMessage, AgentState, and BaseAgent basics."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from squix.agents.base import AgentMessage, AgentState, BaseAgent


class TestAgentMessage:
    def test_create_simple(self):
        """AgentMessage with minimal fields."""
        msg = AgentMessage(sender="talk", recipient="build", content="hello")
        assert msg.sender == "talk"
        assert msg.recipient == "build"
        assert msg.content == "hello"
        assert msg.task_id == ""
        assert msg.reply_to is None
        assert msg.metadata == {}

    def test_create_with_task_id(self):
        msg = AgentMessage(
            sender="orch", recipient="user", content="done",
            task_id="t001", metadata={"type": "result"},
        )
        assert msg.task_id == "t001"
        assert msg.metadata["type"] == "result"


class TestAgentState:
    def test_states_are_strings(self):
        assert AgentState.IDLE == "idle"
        assert AgentState.WORKING == "working"
        assert AgentState.WAITING == "waiting"
        assert AgentState.ERROR == "error"
        assert AgentState.DONE == "done"


class TestBaseAgent:
    """BaseAgent is abstract — use a concrete subclass for testing."""

    class ConcreteAgent(BaseAgent):
        async def handle(self, msg):
            return None

    def _make_agent(self, **kwargs):
        return self.ConcreteAgent(**kwargs)

    def test_default_values(self):
        """Base agent defaults."""
        agent = self._make_agent()
        assert agent.agent_id == "base"
        assert agent.role == ""
        assert agent.state == AgentState.IDLE
        assert agent.model_prefers == []
        assert agent.neighbors == []
        assert agent._conversation == []

    def test_set_id_and_role(self):
        agent = self._make_agent(agent_id="myagent", role="My role")
        assert agent.agent_id == "myagent"
        assert agent.role == "My role"

    def test_put_message(self):
        """Agent can receive a message."""
        import asyncio
        loop = asyncio.get_event_loop()

        async def _run():
            agent = self._make_agent()
            msg = AgentMessage(sender="user", recipient="test",
                               content="hi", task_id="t001")
            await agent.put_message(msg)
            got = await asyncio.wait_for(agent._inbox.get(), timeout=1.0)
            return got

        result = loop.run_until_complete(_run())
        assert result.content == "hi"

    def test_send_to_puts_in_inbox(self):
        """send_to calls the send_fn with a new message."""
        import asyncio
        loop = asyncio.get_event_loop()

        async def _run():
            sent_msgs = []

            async def capture(msg):
                sent_msgs.append(msg)

            agent = self._make_agent(agent_id="alice", send_fn=capture)
            await agent.send_to("bob", "hello bob", task_id="t001")
            return sent_msgs

        results = loop.run_until_complete(_run())
        assert len(results) == 1
        assert results[0].sender == "alice"
        assert results[0].recipient == "bob"

    def test_send_result_puts_in_queue(self):
        """send_result puts the message in the result queue."""
        import asyncio
        loop = asyncio.get_event_loop()

        async def _run():
            q = asyncio.Queue()
            agent = self._make_agent(result_queue=q)
            await agent.send_result("done!", task_id="t001")
            got = await asyncio.wait_for(q.get(), timeout=1.0)
            return got

        result = loop.run_until_complete(_run())
        assert result.recipient == "user"
        assert result.content == "done!"

    def test_to_dict(self):
        """Serializable dict includes all state fields."""
        agent = self._make_agent(agent_id="x", role="R", model_prefers=["m1"])
        d = agent.to_dict()
        assert d["agent_id"] == "x"
        assert d["role"] == "R"
        assert d["model_prefers"] == ["m1"]
        assert d["state"] == "idle"

    def test_default_system_prompt(self):
        agent = self._make_agent(agent_id="foo", role="Bar")
        prompt = agent._default_system_prompt()
        assert "foo" in prompt
        assert "Bar" in prompt

    def test_run_loop_processes_messages(self):
        """The run loop processes messages and changes state correctly."""
        import asyncio
        loop = asyncio.get_event_loop()

        class SimpleAgent(self.ConcreteAgent):
            async def handle(self, msg):
                return AgentMessage(
                    sender=self.agent_id, recipient="user",
                    content=f"echo: {msg.content}",
                    task_id=msg.task_id,
                    metadata={"type": "result"},
                )

        async def _run():
            q = asyncio.Queue()
            agent = SimpleAgent(
                agent_id="simple", result_queue=q,
            )

            # Start the run loop as a task
            task = asyncio.create_task(agent.run())

            try:
                # Send two messages
                await agent.put_message(
                    AgentMessage(sender="user", recipient="simple",
                                 content="first", task_id="t001")
                )
                await agent.put_message(
                    AgentMessage(sender="user", recipient="simple",
                                 content="second", task_id="t002")
                )

                # Get results
                r1 = await asyncio.wait_for(q.get(), timeout=2.0)
                r2 = await asyncio.wait_for(q.get(), timeout=2.0)
                return r1, r2
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        r1, r2 = loop.run_until_complete(_run())
        assert r1.content == "echo: first"
        assert r2.content == "echo: second"

    def test_run_loop_error_sets_error_state(self):
        """If handle() raises, agent goes to ERROR state."""
        import asyncio
        loop = asyncio.get_event_loop()

        class ErrorAgent(self.ConcreteAgent):
            async def handle(self, msg):
                raise RuntimeError("boom!")

        async def _run():
            q = asyncio.Queue()
            agent = ErrorAgent(agent_id="err", result_queue=q)
            task = asyncio.create_task(agent.run())

            try:
                await agent.put_message(
                    AgentMessage(sender="user", recipient="err",
                                 content="trigger error", task_id="t001")
                )
                # Wait for the error message to appear in the result queue
                err_msg = await asyncio.wait_for(q.get(), timeout=2.0)
                assert err_msg.metadata.get("type") == "error"
                assert agent.state == AgentState.ERROR
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        loop.run_until_complete(_run())
