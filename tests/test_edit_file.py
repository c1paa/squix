"""Tests for edit_file skill — the new old_string/new_string editing mechanism."""

from __future__ import annotations

import pytest

from squix.skills.registry import SkillRegistry
from squix.workspace.manager import WorkspaceManager


@pytest.fixture
def workspace(tmp_path):
    ws = WorkspaceManager(project_dir=tmp_path)
    ws.init()
    return ws


@pytest.fixture
def registry(workspace):
    return SkillRegistry(workspace=workspace)


class TestEditFile:
    """edit_file skill: old_string/new_string replacement."""

    @pytest.mark.asyncio
    async def test_basic_replacement(self, registry, workspace, tmp_path):
        """Replace a unique string in a file."""
        (tmp_path / "hello.py").write_text("print('hello')\n")
        # Must read first
        workspace.read_file("hello.py")
        result = await registry.call(
            "edit_file",
            {"path": "hello.py", "old_string": "hello", "new_string": "world"},
            agent_id="build",
        )
        assert result["status"] == "success"
        assert result["replacements"] == 1
        content = (tmp_path / "hello.py").read_text()
        assert "world" in content
        assert "hello" not in content

    @pytest.mark.asyncio
    async def test_no_match_returns_error(self, registry, workspace, tmp_path):
        """If old_string not found, return error with hint."""
        (tmp_path / "hello.py").write_text("print('hello')\n")
        workspace.read_file("hello.py")
        result = await registry.call(
            "edit_file",
            {"path": "hello.py", "old_string": "nonexistent_text", "new_string": "x"},
            agent_id="build",
        )
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_multiple_occurrences_error(self, registry, workspace, tmp_path):
        """Multiple occurrences without replace_all should error."""
        (tmp_path / "dup.py").write_text("x = 1\nx = 2\n")
        workspace.read_file("dup.py")
        result = await registry.call(
            "edit_file",
            {"path": "dup.py", "old_string": "x = ", "new_string": "y = "},
            agent_id="build",
        )
        assert result["status"] == "error"
        assert "2 times" in result["error"]

    @pytest.mark.asyncio
    async def test_replace_all(self, registry, workspace, tmp_path):
        """replace_all=true replaces all occurrences."""
        (tmp_path / "dup.py").write_text("x = 1\nx = 2\n")
        workspace.read_file("dup.py")
        result = await registry.call(
            "edit_file",
            {
                "path": "dup.py",
                "old_string": "x = ",
                "new_string": "y = ",
                "replace_all": True,
            },
            agent_id="build",
        )
        assert result["status"] == "success"
        assert result["replacements"] == 2
        content = (tmp_path / "dup.py").read_text()
        assert content.count("y = ") == 2

    @pytest.mark.asyncio
    async def test_must_read_first(self, registry, tmp_path):
        """edit_file without prior read_file should error."""
        (tmp_path / "unread.py").write_text("pass\n")
        result = await registry.call(
            "edit_file",
            {"path": "unread.py", "old_string": "pass", "new_string": "return"},
            agent_id="build",
        )
        assert result["status"] == "error"
        assert "read_file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_syntax_check_on_python(self, registry, workspace, tmp_path):
        """After editing a .py file, syntax_check should be present."""
        (tmp_path / "syn.py").write_text("x = 1\n")
        workspace.read_file("syn.py")
        result = await registry.call(
            "edit_file",
            {"path": "syn.py", "old_string": "x = 1", "new_string": "x = 2"},
            agent_id="build",
        )
        assert result["status"] == "success"
        assert "syntax_check" in result
        assert result["syntax_check"] == "ok"

    @pytest.mark.asyncio
    async def test_file_not_found(self, registry, workspace):
        """Editing a nonexistent file should error."""
        result = await registry.call(
            "edit_file",
            {"path": "nope.py", "old_string": "x", "new_string": "y"},
            agent_id="build",
        )
        assert result["status"] == "error"


class TestFileStateTracking:
    """WorkspaceManager file state tracking."""

    def test_read_creates_state(self, workspace, tmp_path):
        (tmp_path / "a.py").write_text("code")
        workspace.read_file("a.py")
        state = workspace.get_file_state("a.py")
        assert state is not None
        assert "content_hash" in state
        assert "mtime" in state

    def test_staleness_not_read(self, workspace, tmp_path):
        (tmp_path / "b.py").write_text("code")
        is_stale, reason = workspace.check_staleness("b.py")
        assert is_stale
        assert "not read" in reason

    def test_staleness_after_read_ok(self, workspace, tmp_path):
        (tmp_path / "c.py").write_text("code")
        workspace.read_file("c.py")
        is_stale, reason = workspace.check_staleness("c.py")
        assert not is_stale

    def test_staleness_after_external_write(self, workspace, tmp_path):
        (tmp_path / "d.py").write_text("code")
        workspace.read_file("d.py")
        # Simulate external modification
        import time
        time.sleep(0.05)
        (tmp_path / "d.py").write_text("modified")
        is_stale, reason = workspace.check_staleness("d.py")
        assert is_stale
        assert "modified" in reason

    def test_write_updates_state(self, workspace, tmp_path):
        workspace.write_file("e.py", "new content")
        state = workspace.get_file_state("e.py")
        assert state is not None
        is_stale, _ = workspace.check_staleness("e.py")
        assert not is_stale
