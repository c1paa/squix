"""Tests for PolicyEngine — model selection, fallback, escalation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from squix.policy.engine import PolicyEngine


class TestPolicySelection:
    def test_select_model_returns_first_available(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_model_ids.return_value = ["m1", "m2"]
        reg.get_adapter.side_effect = lambda mid: adapter if mid == "m1" else None

        result = policy.select_model("any", reg)
        assert result == "m1"

    def test_select_model_all_available(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_model_ids.return_value = ["m1", "m2"]
        reg.get_adapter.return_value = adapter

        result = policy.select_model("any", reg)
        assert result == "m1"

    def test_select_model_none_available(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        reg.get_model_ids.return_value = ["m1"]
        reg.get_adapter.return_value = None  # nothing works

        result = policy.select_model("any", reg)
        assert result is None

    def test_select_model_escalation_uses_escalation_models(self):
        policy = PolicyEngine({"escalation_enabled": True, "escalation_models": ["big-model"]})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_adapter.return_value = adapter

        result = policy.select_model("any", reg, escalated=True)
        assert result == "big-model"

    def test_select_model_for_agent_returns_first_in_prefer_list(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_adapter.side_effect = lambda mid: adapter if mid == "m2" else None

        result = policy.select_model_for_agent(["m1", "m2"], reg)
        assert result == "m2"

    def test_select_model_for_agent_falls_back(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_model_ids.return_value = ["fb"]
        # Return adapter only for "fb", None for anything else
        reg.get_adapter.side_effect = lambda mid: adapter if mid == "fb" else None

        result = policy.select_model_for_agent(["nonexistent"], reg)
        assert result == "fb"


class TestFallback:
    def test_should_fallback_true(self):
        policy = PolicyEngine({"fallback_enabled": True, "max_retries": 3})
        assert policy.should_fallback("error", 1) is True
        assert policy.should_fallback("error", 2) is True
        assert policy.should_fallback("error", 3) is False  # at max

    def test_should_fallback_disabled(self):
        policy = PolicyEngine({"fallback_enabled": False})
        assert policy.should_fallback("error", 1) is False

    def test_next_fallback_skips_current(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        adapter = MagicMock()
        reg.get_model_ids.return_value = ["m1", "m2", "m3"]
        reg.get_adapter.side_effect = lambda mid: adapter if mid != "m1" else None

        result = policy.next_fallback_model("m2", ["m1", "m2", "m3"], reg)
        assert result == "m3"  # skips m2 (current), m1 not available

    def test_next_fallback_no_more_models(self):
        policy = PolicyEngine({})
        reg = MagicMock()
        reg.get_model_ids.return_value = ["m1"]
        reg.get_adapter.return_value = None
        result = policy.next_fallback_model("m1", ["m1"], reg)
        assert result is None


class TestEscalation:
    def test_needs_escalation_low_confidence(self):
        policy = PolicyEngine({"escalation_enabled": True})
        assert policy.needs_escalation(0.1) is True
        assert policy.needs_escalation(0.5) is False

    def test_needs_escalation_disabled(self):
        policy = PolicyEngine({"escalation_enabled": False})
        assert policy.needs_escalation(0.0) is False

    def test_needs_escalation_none_is_false(self):
        policy = PolicyEngine({"escalation_enabled": True})
        assert policy.needs_escalation(None) is False

    def test_get_escalation_model(self):
        policy = PolicyEngine({"escalation_models": ["gpt5", "claude-opus"]})
        assert policy.get_escalation_model() == "gpt5"

    def test_get_escalation_model_empty(self):
        policy = PolicyEngine({})
        assert policy.get_escalation_model() is None
