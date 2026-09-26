from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Callable, Mapping, Sequence

from core.types import ClinicalState
from diagnosis.belief import DiagnosisBelief
from result_prediction.predictor import PlanningResultPredictor


@dataclass(frozen=True)
class PlannerConfig:
    max_steps: int = 5
    search_depth: int = 2
    max_actions: int = 5
    belief_threshold: float = 0.9
    critical_loss_threshold: float = 0.0
    diagnostic_error_cost: float = 10_000.0
    test_costs: Mapping[str, float] = field(default_factory=dict)
    critical_extra_loss: Mapping[str, float] = field(default_factory=dict)
    critical_default_extra_loss: float = 0.0


class RollingAndOrPlanner:
    """Finite-depth AND-OR planner; execute only the returned root action per round."""

    STOP = "__STOP__"

    def __init__(self, diagnosis: DiagnosisBelief, predictor: PlanningResultPredictor, diseases: Sequence[str], critical_diseases: set[str], config: PlannerConfig,
                 is_critical: Callable[[str], bool] | None = None,
                 candidate_selector: Callable[[ClinicalState, Mapping[str, float]], Sequence[str]] | None = None,
                 candidate_batch_selector: Callable[[Sequence[tuple[str, ClinicalState, Mapping[str, float]]]], Mapping[str, Sequence[str]]] | None = None) -> None:
        self.diagnosis, self.predictor = diagnosis, predictor
        self.diseases, self.critical, self.config = tuple(diseases), set(critical_diseases), config
        self.is_critical = is_critical or (lambda disease: disease in self.critical)
        self.candidate_selector = candidate_selector
        self.candidate_batch_selector = candidate_batch_selector
        self._belief_cache: dict[str, dict[str, float]] = {}
        self._candidate_cache: dict[str, tuple[str, ...]] = {}
        self._root_state_key = ""
        self.last_trace: dict[str, object] = {}

    @staticmethod
    def _state_key(state: ClinicalState) -> str:
        return json.dumps({"case_id": state.case_id, "initial_information": state.initial_information,
                           "history": [{"action": item.action, "value": item.value, "source": item.source}
                                       for item in state.history]},
                          ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))

    def _critical_loss(self, belief: Mapping[str, float], preferred: str | None = None) -> float:
        return sum(probability * self.config.critical_extra_loss.get(disease, self.config.critical_default_extra_loss)
                   for disease, probability in belief.items()
                   if self.is_critical(disease) and disease != preferred)

    def _belief(self, state: ClinicalState) -> dict[str, float]:
        if state.belief:
            return dict(state.belief)
        key = self._state_key(state)
        cached = self._belief_cache.get(key)
        if cached is None:
            cached = dict(self.diagnosis.infer(state, self.diseases))
            self._belief_cache[key] = cached
        return dict(cached)

    def _terminal_loss(self, state: ClinicalState) -> float:
        belief = self._belief(state)
        diagnosis, probability = self.diagnosis.preferred(belief)
        general = self.config.diagnostic_error_cost * (1.0 - probability)
        critical = self._critical_loss(belief, diagnosis)
        return general + critical

    def _safe(self, state: ClinicalState) -> bool:
        belief = self._belief(state)
        _, confidence = self.diagnosis.preferred(belief)
        critical_loss = self._critical_loss(belief)
        return confidence >= self.config.belief_threshold and critical_loss <= self.config.critical_loss_threshold

    def _candidate_actions(self, state: ClinicalState, root_actions: Sequence[str]) -> tuple[str, ...]:
        key = self._state_key(state)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        if key == self._root_state_key or self.candidate_selector is None:
            selected = tuple(root_actions)
        else:
            selected = tuple(self.candidate_selector(state, self._belief(state)))
        self._candidate_cache[key] = selected
        return selected

    def _prefetch_candidates(self, states: Sequence[ClinicalState]) -> None:
        if self.candidate_batch_selector is None:
            return
        requests = []
        for state in states:
            key = self._state_key(state)
            if key not in self._candidate_cache:
                requests.append((key, state, self._belief(state)))
        if not requests:
            return
        selected = self.candidate_batch_selector(requests)
        for key, _, _ in requests:
            if key not in selected:
                raise ValueError(f"batch candidateEnglish textmissingEnglish text {key}")
            self._candidate_cache[key] = tuple(selected[key])

    def allowed_actions(self, state: ClinicalState, actions: Sequence[str]) -> tuple[str, ...]:
        candidates = self._candidate_actions(state, actions)
        remaining = tuple(a for a in candidates if a not in {item.action for item in state.history})[: self.config.max_actions]
        if not remaining or state.step >= self.config.max_steps:
            return (self.STOP,)
        # The shared FISC comparison contract makes the confidence threshold a
        # stopping rule, not merely another low-cost option in the search.
        # Full Planning additionally requires `_safe`, which includes the
        # critical-disease loss constraint.
        if state.step > 0 and self._safe(state):
            return (self.STOP,)
        return remaining

    def _value(self, state: ClinicalState, actions: Sequence[str], depth: int) -> float:
        # A depth limit ends *planning*, not the clinical episode.  The
        # documented leaf approximation is the loss of stopping at this
        # simulated state; it must not charge an extra unobserved test.
        if depth <= 0:
            return self._terminal_loss(state)
        allowed = self.allowed_actions(state, actions)
        if self.STOP in allowed and len(allowed) == 1:
            return self._terminal_loss(state)
        return min(self._q_value(state, action, actions, depth) for action in allowed)

    def _q_value(self, state: ClinicalState, action: str, actions: Sequence[str], depth: int) -> float:
        if action == self.STOP:
            return self._terminal_loss(state)
        future = 0.0
        for branch in self.predictor.branches(state, action):
            next_state = ClinicalState(state.case_id, state.initial_information, state.history + (branch.observation,), {})
            future += branch.probability * self._value(next_state, actions, depth - 1)
        return self.config.test_costs.get(action, 0.0) + future

    def _value_trace(self, state: ClinicalState, actions: Sequence[str], depth: int) -> tuple[float, dict[str, object]]:
        if depth <= 0:
            loss = self._terminal_loss(state)
            return loss, {"kind": "terminal", "depth": depth, "loss": loss, "belief": self._belief(state)}
        candidates = self._candidate_actions(state, actions)
        allowed = self.allowed_actions(state, actions)
        if self.STOP in allowed and len(allowed) == 1:
            loss = self._terminal_loss(state)
            return loss, {"kind": "stop", "depth": depth, "loss": loss, "belief": self._belief(state)}
        if depth > 1:
            child_states = []
            for action in allowed:
                if action != self.STOP:
                    child_states.extend(ClinicalState(state.case_id, state.initial_information,
                                                      state.history + (branch.observation,), {})
                                       for branch in self.predictor.branches(state, action))
            self._prefetch_candidates(child_states)
        action_nodes = []
        for action in allowed:
            value, node = self._q_value_trace(state, action, actions, depth)
            action_nodes.append((value, node))
        best_value, best_node = min(action_nodes, key=lambda item: item[0])
        return best_value, {
            "kind": "choice",
            "depth": depth,
            "candidate_actions": list(candidates),
            "allowed_actions": list(allowed),
            "selected_action": best_node["action"],
            "selected_value": best_value,
            "actions": [node for _, node in action_nodes],
        }

    def _q_value_trace(self, state: ClinicalState, action: str, actions: Sequence[str], depth: int) -> tuple[float, dict[str, object]]:
        if action == self.STOP:
            loss = self._terminal_loss(state)
            return loss, {"action": self.STOP, "depth": depth, "q_value": loss, "terminal_loss": loss}
        cost = self.config.test_costs.get(action, 0.0)
        branches = []
        future = 0.0
        for branch in self.predictor.branches(state, action):
            next_state = ClinicalState(state.case_id, state.initial_information, state.history + (branch.observation,), {})
            child_value, child = self._value_trace(next_state, actions, depth - 1)
            weighted_value = branch.probability * child_value
            future += weighted_value
            branches.append({
                "record_type": "predicted_result_branch",
                "probability": branch.probability,
                "predicted_result": {
                    "action": branch.observation.action,
                    "value": branch.observation.value,
                    "source": branch.observation.source,
                },
                "child_value": child_value,
                "weighted_value": weighted_value,
                "reward": -weighted_value,
                "child": child,
            })
        q_value = cost + future
        return q_value, {
            "record_type": "planned_action_path",
            "action": action,
            "depth": depth,
            "test_cost": cost,
            "expected_future_value": future,
            "q_value": q_value,
            "reward": -q_value,
            "branches": branches,
        }

    def choose(self, state: ClinicalState, actions: Sequence[str]) -> tuple[str, dict[str, float]]:
        # Caches are intentionally scoped to one root decision.  A new real
        # observation creates a new decision problem and must not inherit
        # hypothetical inferences from the previous one.
        self._belief_cache.clear()
        self.predictor.clear_cache()
        self._candidate_cache.clear()
        self._root_state_key = self._state_key(state)
        allowed = self.allowed_actions(state, actions)
        action_nodes = []
        for action in allowed:
            value, node = self._q_value_trace(state, action, actions, self.config.search_depth)
            action_nodes.append((value, node))
        values = {str(node["action"]): value for value, node in action_nodes}
        selected = min(values, key=values.get, default=self.STOP)
        self.last_trace = {
            "trace_type": "and_or_planning_report",
            "trace_version": 2,
            "search_depth": self.config.search_depth,
            "candidate_action_count": self.config.max_actions,
            "branches_per_action": self.predictor.branch_count,
            "state_step": state.step,
            "allowed_actions": list(allowed),
            "values": values,
            "selected_action": selected,
            "selected_reward": -values.get(selected, 0.0) if selected != self.STOP else 0.0,
            "root_candidates": list(actions),
            "actions": [node for _, node in action_nodes],
        }
        return selected, values
