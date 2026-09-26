"""Run the Direct_LLM baseline: recommend, execute, then reveal one result."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .self_depending import SelfDependingProvider
from .icd10 import DEFAULT_CODEBOOK, icd3, require_icd10_cn_v2
from .simulate_missing_catalog_test import _bundle
from catalog.actions import build_catalog
from core.run_store import RunStore
from core.types import ClinicalState, Observation
from data_ingestion.fhir_bundle import discover_cases, load_case, real_results_by_loinc
from execution_environment.environment import ResultEnvironment
from providers.llm import LLMClinicalSummaryProvider, LLMSimulationProvider, load_llm_config, load_simulation_llm_config


def _resume(case, checkpoint: dict) -> tuple[ClinicalState, list[dict]]:
    history = tuple(Observation(x["action"], x["value"], x["source"]) for x in checkpoint.get("history", []))
    differentials = list(checkpoint.get("differential_history", []))
    return ClinicalState(case.case_id, case.initial_information, history), differentials


def _gold_icd10_codes(path: Path) -> tuple[str, ...]:
    """Evaluation-only oracle.  Never included in any LLM request."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        code = resource.get("code", {})
        for coding in code.get("coding", []):
            if "icd-10" not in str(coding.get("system", "")).lower():
                continue
            found.update(part.strip().upper() for part in str(coding.get("code", "")).split(";") if part.strip())
    return tuple(sorted(found))


def _write_audit(directory: Path, name: str, payload: object) -> None:
    """Write an inspectable, atomic per-interaction artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=lambda item: item.__dict__, indent=2), encoding="utf-8")
    temporary.replace(destination)


def _pending_stop(event_path: Path, checkpoint: dict) -> bool:
    """Recover a stop decision written before an interrupted final-diagnosis call."""
    if checkpoint.get("pending_stop") is True:
        return True
    if not event_path.is_file():
        return False
    for line in reversed(event_path.read_text(encoding="utf-8").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "decision":
            return event.get("stop") is True
    return False


class GenerationRetryExhausted(RuntimeError):
    def __init__(self, stage: str, errors: list[dict[str, str]]) -> None:
        super().__init__(stage)
        self.stage, self.errors = stage, errors


def _generate_with_retries(store: RunStore, audit_directory: Path, stage: str, invoke):
    """One initial generation plus at most three retry attempts."""
    errors = []
    for attempt in range(1, 5):
        try:
            return invoke(attempt)
        except Exception as error:
            detail = {"stage": stage, "attempt": attempt, "error_type": type(error).__name__,
                      "error": str(error), "traceback": traceback.format_exc()}
            errors.append({key: str(detail[key]) for key in ("attempt", "error_type", "error")})
            _write_audit(audit_directory, f"{stage}_attempt_{attempt}_error.json", detail)
            store.event("generation_attempt_failed", **detail)
            if "API HTTP 400" in str(error) or "invalid_prompt" in str(error):
                break
    raise GenerationRetryExhausted(stage, errors)


def _save_retry_exhausted(store: RunStore, case, path: Path, state: ClinicalState, differentials: list[dict], pending_stop: bool, exhausted: GenerationRetryExhausted) -> int:
    store.checkpoint({"case_id": case.case_id, "source": str(path), "history": list(state.history),
                      "differential_history": differentials, "pending_stop": pending_stop,
                      "generation_status": {"status": "retry_exhausted", "stage": exhausted.stage, "errors": exhausted.errors},
                      "completed": False})
    store.event("generation_retry_exhausted", stage=exhausted.stage, errors=exhausted.errors)
    print(json.dumps({"case_id": case.case_id, "status": "retry_exhausted", "stage": exhausted.stage}, ensure_ascii=False), file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="FHIR JSON path or a unique patient-id fragment")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[3] / "data/FISC")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[3] / ".env")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).parents[3] / "runs/FISC_TOP3_gpt-6-astra/self_depending")
    parser.add_argument("--run-id")
    parser.add_argument("--mode", choices=("mixed", "simulated", "real_only"), default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum number of tests; not a required count")
    parser.add_argument("--stop-confidence", type=float, default=0.85,
                        help="Require this maximum differential confidence for a model stop")
    parser.add_argument("--icd10-codebook", type=Path, default=DEFAULT_CODEBOOK,
                        help="China Clinical ICD-10 v2.0English textextended code JSON English text")
    args = parser.parse_args()
    if not 0 <= args.stop_confidence <= 1:
        parser.error("--stop-confidence must be between 0 and 1")

    path = Path(args.case)
    if not path.is_file():
        matches = [candidate for candidate in discover_cases(args.data_root) if args.case in candidate.name]
        if len(matches) != 1:
            raise SystemExit("Case path does not exist，or patient id match count is not 1")
        path = matches[0]
    case = load_case(path)
    all_cases = [load_case(candidate) for candidate in discover_cases(args.data_root)]
    catalog = build_catalog(all_cases).actions
    candidates_by_loinc = {}
    for row in catalog.values():
        for code in row.get("loinc", []):
            candidates_by_loinc.setdefault(str(code), {
                "loinc_code": str(code), "display_name": row.get("aliases", [str(code)])[0],
                "price": float(row.get("price", 0.0)),
            })
    candidates = tuple(candidates_by_loinc.values())
    if not candidates:
        raise SystemExit("English textCaseEnglish textYesEnglish text LOINC ofEnglish textExamination")

    store = RunStore(args.run_root, args.run_id or f"self_depending__{case.case_id}")
    saved = store.resume() or {}
    if saved.get("completed"):
        print(json.dumps({"case_id": case.case_id, "status": "already_completed", "run_directory": str(store.directory)}, ensure_ascii=False))
        return 0
    state, differential_history = _resume(case, saved)
    pending_stop = _pending_stop(store.event_path, saved)
    decision_model = SelfDependingProvider(load_llm_config(args.env_file), args.stop_confidence)
    # This is intentionally a distinct call/provider.  It may use PROVIDERS_MODEL.
    simulation_config = load_simulation_llm_config(args.env_file)
    simulation_provider = LLMSimulationProvider(simulation_config)
    summary_provider = LLMClinicalSummaryProvider(simulation_config)
    environment = ResultEnvironment(simulation_provider, store.directory / "simulation_cache")
    real_by_loinc = real_results_by_loinc(case)
    raw_bundle = json.loads(path.read_text(encoding="utf-8"))

    while state.step < args.max_steps and not pending_stop:
        turn = state.step
        audit_directory = store.directory / "interactions"
        try:
            decision = _generate_with_retries(store, audit_directory, f"turn_{turn:02d}_decision", lambda attempt: decision_model.recommend(
            state, [item for code, item in candidates_by_loinc.items() if code not in {x.action for x in state.history}], lambda stage, payload: _write_audit(
                    audit_directory, f"turn_{turn:02d}_decision_attempt_{attempt}_{stage}.json", payload)))
        except GenerationRetryExhausted as exhausted:
            return _save_retry_exhausted(store, case, path, state, differential_history, pending_stop, exhausted)
        differential_record = {"step": state.step, "critical_risk": decision.critical_risk,
                               "differential_diagnoses": list(decision.differential_diagnoses)}
        differential_history.append(differential_record)
        # `request` is the complete visible state sent to Direct_LLM.  It is
        # retained for reproducibility and, by construction, excludes oracle data.
        store.event(
            "decision",
            step=state.step,
            loinc_code=decision.loinc_code,
            stop=decision.stop,
            critical_risk=decision.critical_risk,
            rationale=decision.rationale,
            llm_request=decision.request,
            llm_response=decision.response,
            differential_diagnoses=decision.differential_diagnoses,
        )
        store.checkpoint({"case_id": case.case_id, "source": str(path), "history": list(state.history),
                          "differential_history": differential_history, "pending_stop": decision.stop, "completed": False})
        if decision.stop:
            pending_stop = True
            break
        selected = candidates_by_loinc[decision.loinc_code]
        action = decision.loinc_code
        order = {"resourceType": "ServiceRequest", "status": "active", "intent": "order",
                 "subject": {"reference": f"Patient/{case.case_id}"},
                 "code": {"concept": {"coding": [{"system": "http://loinc.org", "code": action,
                 "display": selected["display_name"]}], "text": selected["display_name"]}}}
        _write_audit(audit_directory, f"turn_{turn:02d}_fhir_order.json", order)
        store.event("fhir_order", step=state.step, loinc_code=decision.loinc_code, service_request=order)
        try:
            def execute(attempt: int):
                simulation_provider.audit = lambda stage, payload: _write_audit(
                    audit_directory, f"turn_{turn:02d}_simulation_attempt_{attempt}_{stage}.json", payload)
                try:
                    return environment.execute(case.case_id, action, args.seed, real_by_loinc,
                                               {"complete_fhir_bundle": raw_bundle, "requested_test": selected}, args.mode)
                finally:
                    simulation_provider.audit = None
            observation = _generate_with_retries(store, audit_directory, f"turn_{turn:02d}_execution", execute)
        except GenerationRetryExhausted as exhausted:
            return _save_retry_exhausted(store, case, path, state, differential_history, False, exhausted)
        _write_audit(audit_directory, f"turn_{turn:02d}_observation.json", {
            "loinc_code": decision.loinc_code, "action": observation.action,
            "value": observation.value, "source": observation.source,
        })
        if observation.source == "simulated":
            simulated_dir = store.directory / "simulated_fhir_results"
            _write_audit(simulated_dir, f"{case.case_id}__{action}__{args.seed}.fhir.json",
                         _bundle(case.case_id, action, selected["display_name"], observation))
        try:
            summary = _generate_with_retries(store, audit_directory, f"turn_{turn:02d}_summary", lambda attempt: summary_provider.summarize(
                selected["display_name"], observation.value, lambda stage, payload: _write_audit(
                    audit_directory, f"turn_{turn:02d}_summary_attempt_{attempt}_{stage}.json", payload)))
        except GenerationRetryExhausted as exhausted:
            return _save_retry_exhausted(store, case, path, state, differential_history, False, exhausted)
        _write_audit(audit_directory, f"turn_{turn:02d}_clinical_summary.json", {
            "loinc_code": decision.loinc_code, "display_name": selected["display_name"], "summary": summary,
        })
        # Only the summary enters the decision state; the complete result stays
        # in the observation/FHIR audit artifacts above.
        visible_observation = Observation(action, summary, observation.source)
        state = state.with_observation(visible_observation, {})
        store.event(
            "observation",
            step=state.step,
            loinc_code=decision.loinc_code,
            action=observation.action,
            value=observation.value,
            clinical_summary=summary,
            source=observation.source,
        )
        store.checkpoint({"case_id": case.case_id, "source": str(path), "history": list(state.history),
                          "differential_history": differential_history, "pending_stop": False, "completed": False})

    try:
        def generate_final(attempt: int):
            final = decision_model.final_diagnosis(
                state, lambda stage, payload: _write_audit(
                    store.directory / "interactions", f"final_diagnosis_attempt_{attempt}_{stage}.json", payload),
                args.icd10_codebook)
            # An invalid/nonexistent ICD is a failed generation and is retried.
            code, name = require_icd10_cn_v2(final.icd10_code, args.icd10_codebook)
            return final, code, name
        final, predicted_code, official_name = _generate_with_retries(
            store, store.directory / "interactions", "final_diagnosis", generate_final)
    except GenerationRetryExhausted as exhausted:
        return _save_retry_exhausted(store, case, path, state, differential_history, pending_stop, exhausted)
    # Gold labels are read only after the model response for offline evaluation.
    gold_codes = _gold_icd10_codes(path)
    exact = predicted_code in gold_codes
    result = {
        "case_id": case.case_id,
        "history": list(state.history),
        "differential_history": differential_history,
        "termination": "max_steps" if state.step >= args.max_steps else "model_confident_stop",
        "policy": "self_depending",
        "execution_config": {"max_steps": args.max_steps, "stop_confidence": args.stop_confidence,
                             "mode": args.mode, "seed": args.seed},
        "total_cost": sum(float(candidates_by_loinc.get(item.action, {}).get("price", 0.0)) for item in state.history),
        "final_diagnosis": final.diagnosis,
        "final_icd10_code": predicted_code,
        "final_icd10_code_system": "China Clinical ICD-10 v2.0",
        "final_icd10_official_name": official_name,
        "final_confidence": final.confidence,
        "final_reason": final.reason,
        "top3_differential_diagnoses": list(final.top_differential_diagnoses),
        "evaluation": {
            "gold_icd10_codes": gold_codes,
            "icd10_exact_correct": exact,
            "icd10_three_character_correct": any(icd3(predicted_code) == icd3(code) for code in gold_codes),
        },
        "completed": True,
    }
    store.event("final_diagnosis", llm_request=final.request, llm_response=final.response,
                diagnosis=final.diagnosis, icd10_code=predicted_code, official_name=official_name,
                confidence=final.confidence, evaluation=result["evaluation"])
    store.checkpoint(result)
    store.event("completed", steps=state.step)
    print(json.dumps(result, ensure_ascii=False, default=lambda item: item.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
