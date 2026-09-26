"""Evaluate Table 1 metrics for completed sequential clinical runs.

This program is deliberately offline-only: labels are read after all model
runs finish and are never supplied to a policy or an execution environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from data_ingestion.fhir_bundle import discover_cases, load_case
from catalog.actions import build_catalog


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_labels(path: Path) -> Path:
    """Bridge the removed project-root config directory during migration."""
    if path.is_file():
        return path
    replacement = Path(__file__).parents[2] / "config/full_planning.json"
    if path.name == "main_experiment_labels.json" and replacement.is_file():
        print(f"Configuration migrated：{path} → {replacement}", file=sys.stderr)
        return replacement
    return path


def _code(value: object) -> str:
    return str(value or "").strip().upper()


def _matches(predicted: str, gold: str, level: str) -> bool:
    if level == "exact":
        return predicted == gold
    if level == "icd3":
        return predicted.replace(".", "")[:3] == gold.replace(".", "")[:3]
    raise ValueError("match_level mustYes exact or icd3")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _bootstrap_mean(values: list[float], samples: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [round(draws[int(samples * .025)], 6), round(draws[min(samples - 1, int(samples * .975))], 6)]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="English text self_depending__<case_id> English textofResultEnglish text")
    parser.add_argument("--data-root", type=Path, required=True, help="FHIR CaseEnglish text，English text")
    parser.add_argument("--labels", type=Path, required=True, help="testEnglish textofEnglish textDiagnosis/criticalEnglish text JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.labels = _resolve_labels(args.labels)
    labels = _read_json(args.labels)
    if labels.get("schema_version") != "main-experiment-labels-v1":
        raise SystemExit("labels mustEnglish text main-experiment-labels-v1 schema")
    # Main experiment correctness is intentionally ICD-10 three-character
    # matching; the full extension is retained for audit only.
    match_level = "icd3"
    label_cases = labels.get("cases")
    if not isinstance(label_cases, dict):
        raise SystemExit("labels.cases mustYesEnglish text case_id English textofobject")
    objective = labels.get("objective", {})
    diagnostic_error_cost = float(objective.get("diagnostic_error_cost", 10000.0))
    critical_extra_cost = float(objective.get("critical_extra_cost", 0.0))
    safety = labels.get("safety", {})
    critical_risk_threshold = safety.get("critical_risk_threshold")
    if critical_risk_threshold is not None:
        critical_risk_threshold = float(critical_risk_threshold)

    case_paths = {load_case(path).case_id: path for path in discover_cases(args.data_root)}
    global_catalog = build_catalog(load_case(path) for path in discover_cases(args.data_root)).actions
    price_by_loinc = {str(loinc): float(row.get("price", 0.0))
                      for row in global_catalog.values() for loinc in row.get("loinc", [])}
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for checkpoint_path in sorted(args.run_root.glob("*/checkpoint.json")):
        result = _read_json(checkpoint_path)
        case_id = str(result.get("case_id") or "")
        if not result.get("completed"):
            excluded.append({"run": str(checkpoint_path.parent), "reason": "not_completed"}); continue
        if case_id not in label_cases:
            excluded.append({"run": str(checkpoint_path.parent), "reason": "missing_frozen_label"}); continue
        if case_id not in case_paths:
            excluded.append({"run": str(checkpoint_path.parent), "reason": "case_not_found_under_data_root"}); continue
        label = label_cases[case_id]
        gold = _code(label.get("primary_icd10_code"))
        predicted = _code(result.get("final_icd10_code"))
        if not gold or not predicted:
            excluded.append({"run": str(checkpoint_path.parent), "reason": "missing_primary_icd10"}); continue
        case = load_case(case_paths[case_id])
        observations = list(result.get("history", []))
        cost = sum(price_by_loinc.get(str(item.get("action")),
                                      float(case.actions.get(item.get("action"), {}).get("price", 0.0)))
                   for item in observations)
        # Original price is the complete FHIR-recorded physician trajectory.
        # A clinician-reviewed diagnostic-only price may optionally override it
        # for the resource-saving secondary analysis.
        original_physician_cost = sum(float(item.get("price", 0.0)) for item in case.actions.values())
        original_physician_test_count = len(case.actions)
        physician_cost = float(label.get("physician_diagnostic_cost", original_physician_cost))
        correct = _matches(predicted, gold, "icd3")
        critical = bool(label.get("critical", False))
        real_count = sum(item.get("source") == "real" for item in observations)
        differential_history = list(result.get("differential_history", []))
        critical_risk = differential_history[-1].get("critical_risk") if differential_history else None
        safe = None if critical_risk is None or critical_risk_threshold is None else float(critical_risk) <= critical_risk_threshold
        termination = result.get("termination")
        stop_class = None
        if safe is not None:
            stop_class = "active-safe" if termination == "model_stop" and safe else "boundary-safe" if safe else "forced-unsafe"
        rows.append({
            "case_id": case_id, "run_directory": str(checkpoint_path.parent), "gold_primary_icd10": gold,
            "predicted_primary_icd10": predicted, "correct": correct, "critical": critical,
            "critical_miss": critical and not correct, "test_count": len(observations), "test_cost": round(cost, 2),
            "original_physician_path_cost": round(original_physician_cost, 2),
            "original_physician_test_count": original_physician_test_count,
            "physician_diagnostic_cost": round(physician_cost, 2), "saving": round(physician_cost - cost, 2),
            "real_result_count": real_count, "simulated_result_count": len(observations) - real_count,
            "termination": termination, "critical_risk_at_stop": critical_risk, "stop_classification": stop_class,
            "total_objective": round(cost + diagnostic_error_cost * (not correct) + critical_extra_cost * (critical and not correct), 2),
        })

    if not rows:
        raise SystemExit("No completed cases are available for evaluation；Please check --run-root、--data-root andEnglish text")
    critical_rows = [row for row in rows if row["critical"]]
    total_tests = sum(row["test_count"] for row in rows)
    stop_rows = [row for row in rows if row["stop_classification"] is not None]
    summary = {
        "method": labels.get("method_name", "Direct LLM"), "n": len(rows), "match_level": match_level,
        "accuracy": _mean([float(row["correct"]) for row in rows]),
        "accuracy_bootstrap_95ci": _bootstrap_mean([float(row["correct"]) for row in rows], args.bootstrap_samples, args.seed),
        "critical_miss_rate": _mean([float(row["critical_miss"]) for row in critical_rows]),
        "critical_miss_count": f"{sum(row['critical_miss'] for row in critical_rows)}/{len(critical_rows)}",
        "avg_cost": _mean([float(row["test_cost"]) for row in rows]),
        "avg_cost_bootstrap_95ci": _bootstrap_mean([float(row["test_cost"]) for row in rows], args.bootstrap_samples, args.seed),
        "avg_tests": _mean([float(row["test_count"]) for row in rows]),
        "avg_original_physician_path_cost": _mean([float(row["original_physician_path_cost"]) for row in rows]),
        "total_original_physician_path_cost": round(sum(float(row["original_physician_path_cost"]) for row in rows), 2),
        "avg_original_physician_test_count": _mean([float(row["original_physician_test_count"]) for row in rows]),
        "total_original_physician_test_count": sum(int(row["original_physician_test_count"]) for row in rows),
        "avg_physician_diagnostic_cost": _mean([float(row["physician_diagnostic_cost"]) for row in rows]),
        "avg_total_objective": _mean([float(row["total_objective"]) for row in rows]),
        "safe_stop_rate": _mean([float(row["stop_classification"] != "forced-unsafe") for row in stop_rows]),
        "active_safe_stop_rate": _mean([float(row["stop_classification"] == "active-safe") for row in stop_rows]),
        "boundary_safe_stop_rate": _mean([float(row["stop_classification"] == "boundary-safe") for row in stop_rows]),
        "unsafe_forced_stop_rate": _mean([float(row["stop_classification"] == "forced-unsafe") for row in stop_rows]),
        "safe_stop_rate_note": None if len(stop_rows) == len(rows) else "Some run artifacts do not contain critical_risk orEnglish text，excluded from the safe-stop-rate denominator。",
        "critical_risk_threshold": critical_risk_threshold,
        "real_result_rate": round(sum(row["real_result_count"] for row in rows) / total_tests, 6) if total_tests else None,
        "simulated_result_rate": round(sum(row["simulated_result_count"] for row in rows) / total_tests, 6) if total_tests else None,
        "avg_saving_vs_physician_diagnostic_path": _mean([float(row["saving"]) for row in rows]),
        "relative_saving_rate_vs_physician_diagnostic_path": round(sum(row["saving"] for row in rows) / sum(row["physician_diagnostic_cost"] for row in rows), 6) if sum(row["physician_diagnostic_cost"] for row in rows) else None,
        "objective_weights": {"diagnostic_error_cost": diagnostic_error_cost, "critical_extra_cost": critical_extra_cost},
        "excluded_runs": excluded,
    }
    _write_json(args.output_dir / "main_experiment_summary.json", summary)
    chinese_headers = {
        "case_id": "Case ID", "run_directory": "run directory", "gold_primary_icd10": "Gold standardEnglish textICD-10",
        "predicted_primary_icd10": "predicted primaryICD-10", "correct": "ICD-10English textYesNoEnglish text", "critical": "critical case",
        "critical_miss": "critical miss", "test_count": "policy examination count", "test_cost": "total policy examination cost",
        "original_physician_test_count": "original physician examination count", "original_physician_path_cost": "original physician examination cost",
        "physician_diagnostic_cost": "physician diagnostic examination cost", "saving": "savings relative to physician pathway",
        "real_result_count": "real result count", "simulated_result_count": "simulated result count",
        "termination": "termination reason", "critical_risk_at_stop": "critical risk at termination",
        "stop_classification": "termination classification", "total_objective": "total monetized objective",
    }
    with (args.output_dir / "main_experiment_per_case.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [chinese_headers.get(key, key) for key in rows[0]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({chinese_headers.get(key, key): value for key, value in row.items()} for row in rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
