from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Protocol

from core.types import ClinicalState, OutcomeBranch


class OutcomeProvider(Protocol):
    def predict(self, state: ClinicalState, action: str, branch_count: int,
                expected_result_schema: Sequence[Mapping[str, object]] = ()) -> Sequence[OutcomeBranch]: ...


class PlanningResultPredictor:
    """Generate normalized hypothetical branches, never actual execution results."""

    def __init__(self, provider: OutcomeProvider, branch_count: int, action_schemas: Mapping[str, Sequence[Mapping[str, object]]] | None = None) -> None:
        self.provider, self.branch_count = provider, branch_count
        self.action_schemas = action_schemas or {}
        # A search revisits the same simulated state through several value
        # calculations.  Reusing its branches both makes the tree internally
        # consistent and prevents duplicate provider calls.
        self._cache: dict[tuple[str, str, int], tuple[OutcomeBranch, ...]] = {}

    @staticmethod
    def _state_key(state: ClinicalState) -> str:
        return json.dumps({"case_id": state.case_id, "initial_information": state.initial_information,
                           "history": [{"action": item.action, "value": item.value, "source": item.source}
                                       for item in state.history]},
                          ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

    def clear_cache(self) -> None:
        """Start a new decision tree; never reuse hypothetical outcomes across turns."""
        self._cache.clear()

    def branches(self, state: ClinicalState, action: str) -> tuple[OutcomeBranch, ...]:
        key = (self._state_key(state), action, self.branch_count)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        schema = self.action_schemas.get(action, ())
        try:
            candidates = tuple(self.provider.predict(state, action, self.branch_count, schema))
        except TypeError as error:
            # Keep small test/dummy providers and external adapters compatible
            # with the pre-schema three-argument contract.
            if "positional" not in str(error) and "argument" not in str(error):
                raise
            candidates = tuple(self.provider.predict(state, action, self.branch_count))
        if not candidates:
            raise ValueError("result prediction Provider returned no branches")
        total = sum(max(0.0, branch.probability) for branch in candidates)
        if total <= 0:
            raise ValueError("result branches contain no positive probability")
        normalized = tuple(OutcomeBranch(branch.observation, max(0.0, branch.probability) / total) for branch in candidates)
        self._cache[key] = normalized
        return normalized
