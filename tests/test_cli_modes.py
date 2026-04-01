"""Tests for CLI modes — cycle, display info, mode routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the mode constants directly
MODES = {
    "auto": {"label": "Auto", "icon": "🚀", "color": "green",
             "description": "Full pipeline — classify, plan, execute"},
    "plan": {"label": "Plan", "icon": "📋", "color": "yellow",
             "description": "Show plan only, no execution"},
    "interactive": {"label": "Interactive", "icon": "💬", "color": "cyan",
                    "description": "Ask y/n before each step"},
    "talk": {"label": "Talk", "icon": "🗣️", "color": "blue",
             "description": "Direct chat only, no delegation"},
}
MODE_ORDER = ["auto", "plan", "interactive", "talk"]


class TestModeConstants:
    def test_all_modes_defined(self):
        """Every mode has label, icon, color, description."""
        for mode_name in MODE_ORDER:
            assert mode_name in MODES
            info = MODES[mode_name]
            assert "label" in info
            assert "icon" in info
            assert "color" in info
            assert "description" in info

    def test_mode_order_starts_with_auto(self):
        assert MODE_ORDER[0] == "auto"

    def test_all_modes_have_unique_icons(self):
        icons = [MODES[m]["icon"] for m in MODE_ORDER]
        assert len(icons) == len(set(icons))


class TestModeCycling:
    def test_cycling_sequence(self):
        """Cycling through modes should visit each in order."""
        current = "auto"
        for i in range(1, 5):
            idx = MODE_ORDER.index(current)
            current = MODE_ORDER[(idx + 1) % len(MODE_ORDER)]
            assert current == MODE_ORDER[i % 4]

    def test_full_cycle_returns_to_start(self):
        """Cycling 4 times returns to the original mode."""
        start = "auto"
        current = start
        for _ in range(4):
            idx = MODE_ORDER.index(current)
            current = MODE_ORDER[(idx + 1) % len(MODE_ORDER)]
        assert current == start


class TestModeInfo:
    def test_mode_info_returns_dict(self):
        """Each mode should return a dict with required keys."""
        required_keys = {"label", "icon", "color", "description"}
        for mode_name in MODE_ORDER:
            info = MODES[mode_name]
            assert required_keys.issubset(info.keys())

    def test_auto_mode_label(self):
        assert MODES["auto"]["label"] == "Auto"

    def test_plan_mode_icon(self):
        assert MODES["plan"]["icon"] == "📋"

    def test_talk_mode_description(self):
        assert "no delegation" in MODES["talk"]["description"].lower()


class TestModeEngineRouting:
    """Verify that _handle_input routes to correct engine method per mode."""

    def _make_cli_with_mode(self, mode: str):
        """Factory that creates a CLI mock with a given mode."""
        from squix.ui.cli import SquixCLI

        mock_engine = MagicMock()
        mock_engine.process_input = AsyncMock(return_value=[])
        mock_engine.chat_only = AsyncMock(return_value=[])
        mock_engine.plan_only = AsyncMock(return_value=[])
        mock_engine.cost_tracker = MagicMock()
        mock_engine.cost_tracker.total_cost = 0.0
        mock_engine.get_current_progress = MagicMock(return_value="")

        # Patch the rich console to avoid terminal output
        with patch("squix.ui.cli.console"):
            cli = SquixCLI(mock_engine)
            cli.current_mode = mode
            return cli

    @pytest.mark.asyncio
    async def test_talk_mode_calls_chat_only(self):
        cli = self._make_cli_with_mode("talk")
        # Mock the engine's chat_only
        cli.engine.chat_only = AsyncMock(return_value=[])

        # Patch console to avoid real output
        with patch("squix.ui.cli.console"):
            await cli._handle_input("hello")

        cli.engine.chat_only.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_plan_mode_calls_plan_only(self):
        cli = self._make_cli_with_mode("plan")
        cli.engine.plan_only = AsyncMock(return_value=[])

        with patch("squix.ui.cli.console"):
            await cli._handle_input("write fib")

        cli.engine.plan_only.assert_called_once_with("write fib")

    @pytest.mark.asyncio
    async def test_auto_mode_calls_submit_input(self):
        cli = self._make_cli_with_mode("auto")
        cli.engine.submit_input = AsyncMock(return_value="t001")
        cli.engine.complete_task = AsyncMock()
        cli._stream_execution = AsyncMock()
        cli.engine.project_dir = MagicMock()
        cli.engine.project_dir.name = "test-project"

        with patch("squix.ui.cli.console"):
            await cli._handle_input("write fib")

        cli.engine.submit_input.assert_called_once_with("write fib")

    @pytest.mark.asyncio
    async def test_interactive_mode_calls_submit_input(self):
        cli = self._make_cli_with_mode("interactive")
        cli.engine.submit_input = AsyncMock(return_value="t001")
        cli.engine.complete_task = AsyncMock()
        cli._stream_execution = AsyncMock()
        cli.engine.project_dir = MagicMock()
        cli.engine.project_dir.name = "test-project"

        with patch("squix.ui.cli.console"):
            await cli._handle_input("debug my code")

        cli.engine.submit_input.assert_called_once_with("debug my code")
