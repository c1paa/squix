"""Tests for Session and TaskRecord."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from squix.core.session import Session, TaskRecord


class TestTaskRecord:
    def test_defaults(self):
        t = TaskRecord(id="t001", user_input="hi")
        assert t.status == "pending"
        assert t.created_at is not None
        assert t.completed_at is None
        assert t.plan is None
        assert t.results == []

    def test_with_all_fields(self):
        t = TaskRecord(
            id="t002",
            user_input="write fib",
            status="running",
            created_at="2020-01-01T00:00:00Z",
            completed_at="2020-01-01T00:00:01Z",
            plan="step1, step2",
            results=[{"step": 1, "status": "done"}],
        )
        assert t.status == "running"
        assert t.plan == "step1, step2"
        assert len(t.results) == 1


class TestSession:
    def test_default_session_id(self):
        s = Session()
        assert len(s.session_id) == 8
        assert isinstance(s.session_id, str)

    def test_custom_session_id(self):
        s = Session(session_id="abc123")
        assert s.session_id == "abc123"

    def test_next_task_id_starts_at_1(self):
        s = Session()
        assert s.next_task_id() == "t001"

    def test_next_task_id_increments(self):
        s = Session()
        # Simulate having 2 tasks already
        s.tasks.append(TaskRecord(id="t001", user_input="a"))
        s.tasks.append(TaskRecord(id="t002", user_input="b"))
        assert s.next_task_id() == "t003"

    def test_add_task_appends_record(self):
        s = Session()
        t = s.add_task("t001", "hello world")
        assert len(s.tasks) == 1
        assert t.status == "running"
        assert t.user_input == "hello world"
        assert s.tasks[0] is t

    def test_complete_task_marks_done(self):
        s = Session()
        s.add_task("t001", "test")
        assert s.tasks_completed == 0
        assert s.tasks[0].status == "running"

        s.complete_task("t001")
        assert s.tasks_completed == 1
        assert s.tasks[0].status == "done"
        assert s.tasks[0].completed_at is not None

    def test_complete_task_unknown_id(self):
        s = Session()
        s.add_task("t001", "test")
        # Should not raise — just no-ops for unknown IDs
        s.complete_task("t999")
        assert s.tasks_completed == 0  # nothing completed

    def test_created_at_is_iso(self):
        s = Session()
        # Should be parseable as ISO (no exception)
        datetime.fromisoformat(s.created_at)

    def test_tasks_completed_counter(self):
        s = Session()
        s.add_task("t001", "task1")
        s.add_task("t002", "task2")
        s.complete_task("t001")
        s.complete_task("t002")
        assert s.tasks_completed == 2
