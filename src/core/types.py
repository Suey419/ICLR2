from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Observation:
    action: str
    value: Any
    source: str  # Only "real" or "simulated" is allowed.


@dataclass(frozen=True)
class OutcomeBranch:
    observation: Observation
    probability: float


@dataclass(frozen=True)
class ClinicalState:
    """Approximate POMDP state; history must never contain hidden information."""

    case_id: str
    initial_information: Mapping[str, Any]
    history: tuple[Observation, ...] = ()
    belief: Mapping[str, float] = field(default_factory=dict)

    @property
    def step(self) -> int:
        return len(self.history)

    def with_observation(self, observation: Observation, belief: Mapping[str, float]) -> "ClinicalState":
        return ClinicalState(self.case_id, self.initial_information, self.history + (observation,), dict(belief))
