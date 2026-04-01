"""Cost tracker — accumulates and reports token costs across the session."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class CostTracker:
    """Tracks token usage and costs per model and per agent."""

    def __init__(self) -> None:
        self._costs: dict[str, float] = defaultdict(float)
        self._tokens_input: dict[str, int] = defaultdict(int)
        self._tokens_output: dict[str, int] = defaultdict(int)
        self._agent_costs: dict[str, float] = defaultdict(float)

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
        if agent_id:
            self._agent_costs[agent_id] += cost

    @property
    def total_cost(self) -> float:
        return sum(self._costs.values())

    @property
    def total_input_tokens(self) -> int:
        return sum(self._tokens_input.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(self._tokens_output.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "by_model": {
                mid: {
                    "cost": round(c, 6),
                    "input_tokens": self._tokens_input[mid],
                    "output_tokens": self._tokens_output[mid],
                }
                for mid, c in self._costs.items()
            },
            "by_agent": {
                aid: round(c, 6) for aid, c in self._agent_costs.items()
            },
        }
