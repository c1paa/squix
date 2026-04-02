"""Tests for the agentic loop in BaseAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from squix.agents.base import BaseAgent, AgentMessage
from squix.models.base import ModelResponse


class ConcreteAgent(BaseAgent):
    """Concrete agent for testing (BaseAgent is abstract)."""
    agent_id = "test_agent"
    role = "test"

    async def handle(self, msg: AgentMessage) -> AgentMessage | None:
        return None


class TestParseToolCalls:
    """_parse_tool_calls extracts tool calls from LLM text."""

    def test_json_code_block(self):
        text = '```json\n[{"tool": "read_file", "params": {"path": "main.py"}}]\n```'
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "read_file"

    def test_bare_json_array(self):
        text = 'Let me read the file.\n[{"tool": "read_file", "params": {"path": "x.py"}}]'
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["params"]["path"] == "x.py"

    def test_single_object(self):
        text = 'I\'ll read: {"tool": "list_files", "params": {"path": "."}}'
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "list_files"

    def test_no_tool_calls(self):
        text = "I've completed the task. The file has been updated."
        calls = BaseAgent._parse_tool_calls(text)
        assert calls == []

    def test_multiple_tools(self):
        text = '```json\n[{"tool": "read_file", "params": {"path": "a.py"}}, {"tool": "read_file", "params": {"path": "b.py"}}]\n```'
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 2

    def test_invalid_json_returns_empty(self):
        text = '```json\n[{"tool": "broken\n```'
        calls = BaseAgent._parse_tool_calls(text)
        assert calls == []

    def test_json_without_tool_key_ignored(self):
        text = '[{"name": "foo", "value": 42}]'
        calls = BaseAgent._parse_tool_calls(text)
        assert calls == []

    def test_large_nested_content_with_braces(self):
        """Tool call with code containing { } in content param must be parsed."""
        code = (
            "import pygame\\n"
            "def main():\\n"
            "    screen = pygame.display.set_mode((600, 600))\\n"
            "    while True:\\n"
            "        for event in pygame.event.get():\\n"
            "            if event.type == pygame.QUIT:\\n"
            "                return\\n"
            "        data = {\\\"key\\\": \\\"value\\\"}\\n"
            "if __name__ == '__main__':\\n"
            "    main()\\n"
        )
        text = f'[{{"tool": "write_file", "params": {{"path": "main.py", "content": "{code}"}}}}]'
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "write_file"
        assert calls[0]["params"]["path"] == "main.py"

    def test_tool_call_in_code_block_with_nested_json(self):
        """Tool call inside code block with nested braces."""
        text = (
            "I'll write the file now.\n"
            '```json\n'
            '[{"tool": "write_file", "params": {"path": "app.py", '
            '"content": "data = {\\"a\\": 1}\\nprint(data)"}}]\n'
            '```'
        )
        calls = BaseAgent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "write_file"


class TestFormatToolsForPrompt:

    def test_format_skills(self):
        from squix.skills.definitions import SkillDef, SkillParam
        skills = [
            SkillDef(
                name="read_file",
                description="Read a file",
                allowed_agents=["test"],
                params=[SkillParam("path", description="File path")],
            ),
            SkillDef(
                name="write_file",
                description="Write a file",
                allowed_agents=["test"],
                params=[SkillParam("path"), SkillParam("content")],
                is_dangerous=True,
            ),
        ]
        result = BaseAgent._format_tools_for_prompt(skills)
        assert "read_file" in result
        assert "write_file" in result
        assert "⚠️" in result  # dangerous marker

    def test_empty_skills(self):
        result = BaseAgent._format_tools_for_prompt([])
        assert "No tools" in result


class TestAgenticLoop:
    """run_agentic_loop executes the LLM → tools → repeat cycle."""

    @pytest.fixture
    def agent(self):
        agent = ConcreteAgent()
        agent._skills = MagicMock()
        agent._skills.list_allowed = MagicMock(return_value=[])
        agent._skills.call = AsyncMock(return_value={"status": "success", "content": "hello"})
        agent._progress_cb = None
        agent._current_task_id = "t1"
        agent._result_queue = None
        return agent

    @pytest.mark.asyncio
    async def test_no_tools_returns_final_text(self, agent):
        """If LLM returns no tool calls, return its text as final answer."""
        agent.invoke_llm = AsyncMock(return_value=ModelResponse(
            text="The task is complete.",
            input_tokens=10, output_tokens=5, model_id="test", cost=0.0,
        ))
        final, log = await agent.run_agentic_loop(
            task="do something", system_prompt="you are a test",
        )
        assert final == "The task is complete."
        assert log == []

    @pytest.mark.asyncio
    async def test_single_tool_then_final(self, agent):
        """LLM calls one tool, then returns final text."""
        responses = [
            ModelResponse(
                text='[{"tool": "read_file", "params": {"path": "x.py"}}]',
                input_tokens=10, output_tokens=20, model_id="test", cost=0.0,
            ),
            ModelResponse(
                text="Done. I read the file.",
                input_tokens=30, output_tokens=10, model_id="test", cost=0.0,
            ),
        ]
        agent.invoke_llm = AsyncMock(side_effect=responses)
        agent.invoke_skill = AsyncMock(return_value={"status": "success", "content": "code"})

        final, log = await agent.run_agentic_loop(
            task="read x.py", system_prompt="test",
        )
        assert final == "Done. I read the file."
        assert len(log) == 1
        assert log[0]["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_max_iterations_stops(self, agent):
        """Loop stops after max_iterations even if LLM keeps calling tools."""
        # Each call returns a different tool call to avoid loop detection
        call_count = 0
        async def varying_llm(messages, **kw):
            nonlocal call_count
            call_count += 1
            return ModelResponse(
                text=f'[{{"tool": "list_files", "params": {{"path": "dir{call_count}"}}}}]',
                input_tokens=10, output_tokens=10, model_id="test", cost=0.0,
            )
        agent.invoke_llm = varying_llm
        agent.invoke_skill = AsyncMock(return_value={"status": "success", "files": []})

        final, log = await agent.run_agentic_loop(
            task="list files forever",
            system_prompt="test",
            max_iterations=3,
        )
        assert len(log) == 3

    @pytest.mark.asyncio
    async def test_loop_detection_breaks(self, agent):
        """If LLM repeats exact same tool calls, loop breaks."""
        agent.invoke_llm = AsyncMock(return_value=ModelResponse(
            text='[{"tool": "read_file", "params": {"path": "same.py"}}]',
            input_tokens=10, output_tokens=10, model_id="test", cost=0.0,
        ))
        agent.invoke_skill = AsyncMock(return_value={"status": "success", "content": "x"})

        final, log = await agent.run_agentic_loop(
            task="stuck task", system_prompt="test", max_iterations=10,
        )
        # First iteration executes, second detects duplicate and breaks
        assert len(log) == 1
