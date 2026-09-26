"""Leave one recorded examination out, simulate it, and print JSONL results.

The simulator receives the patient's other visible information and other real
test results, but never the selected examination's result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data_ingestion.fhir_bundle import discover_cases, load_case
from execution_environment.environment import ResultEnvironment
from providers.llm import LLMConfig, LLMSimulationProvider, load_simulation_llm_config
from prompts import load_prompt


_POTENTIALLY_LEAKY_SECTION_NAMES = {"Ancillary examination", "Laboratory examination", "examination result", "Post-admission examination"}
_NEGATIVE_MARKERS = ("No", "Denies", "Not seen", "Normal", "Negative", "Not found", "English text", "Discomfort")
_POSITIVE_MARKERS = ("English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "English text", "Abnormal", "English text", "English text", "Positive", "English text", "English text", "English text")


def _path_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _result_schema(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose target test structure without exposing any target result value."""
    schema: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in result:
        name = str(item.get("name", ""))
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        raw_value = item.get("value")
        numeric = isinstance(raw_value, dict) and isinstance(raw_value.get("value"), (int, float))
        entry: dict[str, Any] = {
            "name": name,
            "value_kind": "numeric" if numeric else "text",
            "reference_range": item.get("reference_range", []),
        }
        if numeric and raw_value.get("unit"):
            entry["unit"] = raw_value["unit"]
        schema.append(entry)
    return schema


def _positive_clinical_summary(case, target_action: str, summary_root: Path) -> tuple[str, list[str]]:
    """Create and cache a deterministic positive-finding-only summary."""
    cache_path = summary_root / f"{case.case_id}__{target_action}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["summary"], cached["excluded_sections"]
    sections = case.initial_information.get("clinical_sections", {})
    excluded_sections = [
        name for name, value in sections.items()
        if name in _POTENTIALLY_LEAKY_SECTION_NAMES or target_action in value
    ]
    findings: list[str] = []
    for name, text in sections.items():
        if name in excluded_sections:
            continue
        for sentence in text.replace("\n", "").replace("；", "。").split("。"):
            sentence = sentence.strip()
            if (sentence and not any(marker in sentence for marker in _NEGATIVE_MARKERS)
                    and any(marker in sentence for marker in _POSITIVE_MARKERS)):
                findings.append(f"{name}：{sentence}。")
    summary = "\n".join(findings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"case_id": case.case_id, "target_action": target_action, "summary": summary, "excluded_sections": excluded_sections}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, excluded_sections


def _parse_hidden_context(raw: str | None, case, target_action: str, summary_root: Path) -> dict[str, Any]:
    if raw is None:
        summary, excluded_sections = _positive_clinical_summary(case, target_action, summary_root)
        target_result = case.real_results[target_action]
        known_results = []
        seen_results: set[str] = set()
        for action, result in case.real_results.items():
            # A DiagnosticReport may appear through multiple ServiceRequests.
            # Mask the target item and duplicate reports, while keeping only one
            # copy of every other duplicated examination.
            if (action == target_action or target_action in action or action in target_action
                    or result == target_result):
                continue
            fingerprint = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
            if fingerprint in seen_results:
                continue
            seen_results.add(fingerprint)
            known_results.append({"action": action, "result": result})
        return {
            "demographics": case.initial_information.get("demographics", {}),
            "positive_clinical_summary": summary,
            "other_completed_examinations": known_results,
            "expected_result_schema": _result_schema(target_result),
            "gold_diagnoses_for_simulation": list(case.diagnoses),
            "masking_note": {
                "masked_examination": target_action,
                "excluded_clinical_sections": excluded_sections,
                "reason": "These sections can contain the masked examination result.",
            },
        }
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--hidden-context mustYes JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[1] / "data/FISC/Group 8")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[1] / ".env")
    parser.add_argument("--count", type=int, default=3, help="number of cases to run")
    parser.add_argument("--case", type=Path, action="append", help="specified FHIR Case；may be supplied repeatedly")
    parser.add_argument("--target-action", help="English textofExaminationEnglish text，for example：Complete blood count")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--cache-root", type=Path, help="English text；defaultEnglish textModelEnglish text")
    parser.add_argument("--audit-root", type=Path, default=Path("runs/simulation_examples/audits"))
    parser.add_argument("--summary-root", type=Path, default=Path("runs/simulation_examples/context_summaries"))
    parser.add_argument("--simulation-model", help="override .env in PATIENT_MODEL，for execution simulation only")
    parser.add_argument("--hidden-context", help="reviewed de-identified replacement for automatically built、English text JSON object")
    parser.add_argument("--show-masked-reference", action="store_true", help="ModelEnglish textResult，for offline comparison only")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    config = load_simulation_llm_config(args.env_file)
    if args.simulation_model:
        config = LLMConfig(config.api_key, config.base_url, args.simulation_model, config.timeout_seconds)
    model_directory = _path_component(config.model)
    system_prompt = load_prompt("execution_simulation.txt")
    prompt_version = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
    cache_root = args.cache_root or Path("runs/simulation_examples/cache") / model_directory / prompt_version
    environment = ResultEnvironment(LLMSimulationProvider(config), cache_root)
    paths = tuple(args.case) if args.case else discover_cases(args.data_root)
    if not paths:
        raise FileNotFoundError(f"not found FHIR Case：{args.data_root}")

    completed = 0
    for path in paths:
        case = load_case(path)
        if not case.actions:
            continue
        if args.target_action:
            action = args.target_action
            if action not in case.actions:
                continue
        else:
            action = next(iter(case.actions))
        if action not in case.real_results:
            # Leave-one-out evaluation requires the masked real result as a
            # reference.
            continue
        context = _parse_hidden_context(args.hidden_context, case, action, args.summary_root)
        model_input = {"action": action, "hidden_context": context, "seed": args.seed}
        audit_dir = args.audit_root / model_directory / prompt_version / f"{case.case_id}__{action}__{args.seed}"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (audit_dir / "model_input.json").write_text(json.dumps(model_input, ensure_ascii=False, indent=2), encoding="utf-8")
        (audit_dir / "run_metadata.json").write_text(json.dumps({"model": config.model, "provider": "LLMSimulationProvider", "prompt": "execution_simulation.txt", "prompt_version": prompt_version, "cache_root": str(cache_root)}, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            observation = environment.execute(
                case.case_id, action, args.seed, {}, context, mode="simulated"
            )
        except Exception as error:
            (audit_dir / "error.json").write_text(json.dumps({"error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise
        result = {
            "case_id": case.case_id,
            "file": str(path),
            "seed": args.seed,
            "action": observation.action,
            "value": observation.value,
            "source": observation.source,
        }
        (audit_dir / "prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (audit_dir / "masked_reference.json").write_text(json.dumps({"case_id": case.case_id, "action": action, "value": case.real_results[action]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**result, "audit_directory": str(audit_dir)}, ensure_ascii=False))
        if args.show_masked_reference:
            print(json.dumps({
                "case_id": case.case_id,
                "action": action,
                "masked_reference_for_evaluation_only": case.real_results[action],
            }, ensure_ascii=False))
        completed += 1
        if completed == args.count:
            break
    if completed < args.count:
        raise RuntimeError(f"only {completed} English textCasecontain executable examinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
