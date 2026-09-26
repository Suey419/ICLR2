from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from core.types import Observation


class SimulationProvider(Protocol):
    def simulate(self, action: str, hidden_context: Mapping[str, Any], seed: int) -> Observation: ...


class ResultEnvironment:
    """Prefer real results; freeze simulated results by case, action, and seed."""

    def __init__(self, simulator: SimulationProvider, cache_root: str | Path) -> None:
        self.simulator, self.cache_root = simulator, Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def execute(self, case_id: str, action: str, seed: int, real_results: Mapping[str, Any], hidden_context: Mapping[str, Any], mode: str = "mixed", env_sens_mode: str | None = None) -> Observation:
        if mode not in {"mixed", "simulated", "real_only"}:
            raise ValueError("mode mustYes mixed、simulated or real_only")
        if env_sens_mode not in {None, "actual_or_unknown", "simulated_always"}:
            raise ValueError("env_sens_mode mustYes actual_or_unknown or simulated_always")
        if env_sens_mode == "actual_or_unknown":
            if action in real_results:
                return Observation(action, real_results[action], "real")
            return Observation(action, "This result is unknown，cannot be inferred to be normal or abnormal。", "unknown")
        if env_sens_mode == "simulated_always":
            mode = "simulated"
        if mode != "simulated" and action in real_results:
            return Observation(action, real_results[action], "real")
        if mode == "real_only":
            raise KeyError(f"Examination {action} No real result is available")
        path = self.cache_root / f"{case_id}__{action}__{seed}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Observation(data["action"], data["value"], "simulated")
        observation = self.simulator.simulate(action, hidden_context, seed)
        if observation.action != action:
            raise ValueError("Simulator returned an examination result inconsistent with the current action")
        # The run directory may be created/resumed concurrently and an older
        # checkpoint can reference a cache directory that is missing.  Make
        # the parent durable at the point of writing as well as in __init__;
        # otherwise every retry repeats the same FileNotFoundError.
        self.cache_root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"action": action, "value": observation.value}, ensure_ascii=False), encoding="utf-8")
        return Observation(action, observation.value, "simulated")
