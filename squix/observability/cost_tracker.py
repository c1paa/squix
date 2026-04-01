"""Cost tracker — accumulates and reports token costs across the session."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger("squix.cost")


class CostTracker:
    """Tracks token usage and costs per model and per agent.

    Supports daily/monthly limits with warnings and auto-switch.
    """

    def __init__(self, limits: dict[str, Any] | None = None) -> None:
        self._costs: dict[str, float] = defaultdict(float)
        self._tokens_input: dict[str, int] = defaultdict(int)
        self._tokens_output: dict[str, int] = defaultdict(int)
        self._agent_costs: dict[str, float] = defaultdict(float)
        self._call_count: dict[str, int] = defaultdict(int)

        # Limits
        limits = limits or {}
        self.daily_limit: float | None = limits.get("daily_limit")
        self.monthly_limit: float | None = limits.get("monthly_limit")
        self.warn_threshold: float = limits.get("warn_threshold", 0.8)
        self._warning_issued = False

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        *,
        agent_id: str | None = None,
    ) -> None:
        """Record a model call."""
        self._costs[model_id] += cost
        self._tokens_input[model_id] += input_tokens
        self._tokens_output[model_id] += output_tokens
        self._call_count[model_id] += 1
        if agent_id:
            self._agent_costs[agent_id] += cost

        # Check limits
        if self.daily_limit and not self._warning_issued:
            if self.total_cost >= self.daily_limit * self.warn_threshold:
                logger.warning(
                    "Cost warning: $%.4f / $%.4f daily limit (%.0f%%)",
                    self.total_cost, self.daily_limit,
                    (self.total_cost / self.daily_limit) * 100,
                )
                self._warning_issued = True

    def is_over_limit(self) -> bool:
        """Check if we've exceeded the daily limit."""
        if self.daily_limit and self.total_cost >= self.daily_limit:
            return True
        return False

    @property
    def total_cost(self) -> float:
        return sum(self._costs.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(self._tokens_input.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self._tokens_output.values())

    @property
    def total_calls(self) -> int:
        return sum(self._call_count.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "by_model": {
                mid: {
                    "cost": round(c, 6),
                    "input_tokens": self._tokens_input[mid],
                    "output_tokens": self._tokens_output[mid],
                    "calls": self._call_count[mid],
                }
                for mid, c in self._costs.items()
            },
            "by_agent": {
                aid: round(c, 6) for aid, c in self._agent_costs.items()
            },
        }
