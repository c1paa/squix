"""Tests for cost tracker."""

from __future__ import annotations

import pytest

from squix.observability.cost_tracker import CostTracker


class TestCostTrackerBasics:
    def test_record_single_model(self):
        ct = CostTracker()
        ct.record(model_id="m1", input_tokens=100, output_tokens=50, cost=0.005)
        assert ct.total_calls == 1
        assert ct.total_cost == pytest.approx(0.005)
        assert ct.total_input_tokens == 100
        assert ct.total_output_tokens == 50

    def test_record_multiple_models(self):
        ct = CostTracker()
        ct.record(model_id="m1", input_tokens=100, output_tokens=50, cost=0.01)
        ct.record(model_id="m2", input_tokens=200, output_tokens=100, cost=0.02)
        assert ct.total_calls == 2
        assert ct.total_cost == pytest.approx(0.03)

    def test_accumulates_costs(self):
        ct = CostTracker()
        for i in range(5):
            ct.record(model_id="m", input_tokens=10, output_tokens=5, cost=0.001)
        assert ct.total_calls == 5
        assert ct.total_cost == pytest.approx(0.005)

    def test_record_with_agent(self):
        ct = CostTracker()
        ct.record(model_id="m1", input_tokens=10, output_tokens=5, cost=0.01, agent_id="build")
        summary = ct.summary()
        assert summary["by_agent"]["build"] == pytest.approx(0.01)


class TestCostTrackerLimits:
    def test_is_over_limit(self):
        ct = CostTracker(limits={"daily_limit": 1.0})
        assert ct.is_over_limit() is False
        ct.record(model_id="m", input_tokens=100, output_tokens=50, cost=1.1)
        assert ct.is_over_limit() is True

    def test_is_over_limit_no_limit_set(self):
        ct = CostTracker()
        # Should never be over limit if no limit is set
        ct.record(model_id="m", input_tokens=1000, output_tokens=500, cost=999.0)
        assert ct.is_over_limit() is False

    def test_warning_issued_at_threshold(self):
        ct = CostTracker(limits={"daily_limit": 1.0, "warn_threshold": 0.8})
        # 0.8 is the warns threshold, so 0.79 should not trigger
        ct.record(model_id="m", input_tokens=100, output_tokens=50, cost=0.79)
        assert ct._warning_issued is False
        # Now cross the threshold
        ct.record(model_id="m", input_tokens=100, output_tokens=50, cost=0.05)
        assert ct.total_cost >= 0.8  # should have triggered
        assert ct._warning_issued is True


class TestCostTrackerSummary:
    def test_summary_with_data(self):
        ct = CostTracker()
        ct.record(model_id="m1", input_tokens=100, output_tokens=50, cost=0.01, agent_id="a1")
        ct.record(model_id="m1", input_tokens=200, output_tokens=100, cost=0.02, agent_id="a2")
        ct.record(model_id="m2", input_tokens=300, output_tokens=150, cost=0.03, agent_id="a1")

        s = ct.summary()
        assert s["total_cost"] == pytest.approx(0.06)
        assert s["total_calls"] == 3
        assert s["by_model"]["m1"]["calls"] == 2
        assert s["by_model"]["m2"]["calls"] == 1
        assert s["by_agent"]["a1"] == pytest.approx(0.04)
        assert s["by_agent"]["a2"] == pytest.approx(0.02)

    def test_empty_summary(self):
        ct = CostTracker()
        s = ct.summary()
        assert s["total_cost"] == 0.0
        assert s["total_calls"] == 0
        assert s["by_model"] == {}
        assert s["by_agent"] == {}
