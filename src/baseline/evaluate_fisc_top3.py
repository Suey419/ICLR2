"""Offline evaluation for FISC Top-k ICD-10 and examination-cost experiments.

The evaluator reads only completed run checkpoints.  Gold ICD-10 labels are
the post-inference labels persisted in each checkpoint's ``evaluation`` field;
they are never supplied to a policy or an LLM by this program.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from catalog.actions import build_catalog
from data_ingestion.fhir_bundle import discover_cases, load_case


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _icd3(value: object) -> str:
    return str(value or "").strip().upper().replace(".", "")[:3]


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def _price_maps(data_root: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Return LOINC prices and cases, covering both sequential action formats."""
    cases = [load_case(path) for path in discover_cases(data_root)]
    catalog = build_catalog(cases).actions
    loinc_prices = {
        str(loinc): float(row.get("price", 0.0))
        for row in catalog.values() for loinc in row.get("loinc", [])
    }
    return loinc_prices, {case.case_id: case for case in cases}


def _strategy_cost(result: dict[str, Any], case: Any, loinc_prices: dict[str, float]) -> tuple[float, int, list[str]]:
    history = list(result.get("history") or [])
    cost, unresolved = 0.0, []
    for item in history:
        action = str(item.get("action") if isinstance(item, dict) else getattr(item, "action", ""))
        if action in case.actions:
            cost += float(case.actions[action].get("price", 0.0))
        elif action in loinc_prices:
            cost += loinc_prices[action]
        else:
            unresolved.append(action)
    return round(cost, 2), len(history), unresolved


def _top_codes(result: dict[str, Any]) -> list[str]:
    rows = result.get("top3_differential_diagnoses") or []
    codes = [_icd3(item.get("icd10_code")) for item in rows if isinstance(item, dict)]
    # A completed run may only store a final diagnosis in older formats.
    if not codes and result.get("final_icd10_code"):
        codes = [_icd3(result["final_icd10_code"])]
    return [code for code in codes if code]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="for example runs/FISC_TOP3_gpt-6-astra，each subdirectory contains one method")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[2] / "data/FISC")
    parser.add_argument("--output-dir", type=Path,
                        help="write by default to --run-root/evaluation_icd3")
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.run_root / "evaluation_icd3"
    print("[1/3] Loading FISC cases and building examination-price map...", flush=True)
    loinc_prices, cases = _price_maps(args.data_root)
    print(f"[2/3] Scanning completed checkpoints for {len(cases)} source cases...", flush=True)
    per_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: dict[str, list[dict[str, str]]] = defaultdict(list)

    for checkpoint in sorted(args.run_root.glob("*/*/checkpoint.json")):
        method = checkpoint.parent.parent.name
        try:
            result = _read(checkpoint)
        except (OSError, json.JSONDecodeError) as error:
            excluded[method].append({"run": str(checkpoint.parent), "reason": f"unreadable: {error}"}); continue
        case_id = str(result.get("case_id") or "")
        if not result.get("completed"):
            excluded[method].append({"run": str(checkpoint.parent), "reason": "not_completed"}); continue
        if case_id not in cases:
            excluded[method].append({"run": str(checkpoint.parent), "reason": "case_not_found"}); continue
        gold = {_icd3(code) for code in result.get("evaluation", {}).get("gold_icd10_codes", []) if _icd3(code)}
        top = _top_codes(result)
        if not gold or not top:
            excluded[method].append({"run": str(checkpoint.parent), "reason": "missing_gold_or_prediction"}); continue
        strategy_cost, strategy_count, unresolved = _strategy_cost(result, cases[case_id], loinc_prices)
        actual_cost = round(sum(float(action.get("price", 0.0)) for action in cases[case_id].actions.values()), 2)
        per_method[method].append({
            "method": method, "case_id": case_id, "gold_icd10_3char": ";".join(sorted(gold)),
            "top1_icd10_3char": top[0], "top2_icd10_3char": ";".join(top[:2]), "top3_icd10_3char": ";".join(top[:3]),
            "top1_correct": top[0] in gold, "top2_correct": bool(gold & set(top[:2])),
            "top3_correct": bool(gold & set(top[:3])), "strategy_test_cost": strategy_cost,
            "strategy_test_count": strategy_count, "actual_test_cost": actual_cost,
            "actual_test_count": len(cases[case_id].actions), "unresolved_strategy_actions": ";".join(unresolved),
        })

    summaries = []
    print("[3/3] Writing per-method and summary reports...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method, rows in sorted(per_method.items()):
        count = len(rows)
        summary = {
            "method": method, "completed_case_count": count,
            "top1_icd10_3char_accuracy": _mean(float(row["top1_correct"]) for row in rows),
            "top2_icd10_3char_accuracy": _mean(float(row["top2_correct"]) for row in rows),
            "top3_icd10_3char_accuracy": _mean(float(row["top3_correct"]) for row in rows),
            "strategy_avg_test_cost": _mean(float(row["strategy_test_cost"]) for row in rows),
            "strategy_avg_test_count": _mean(float(row["strategy_test_count"]) for row in rows),
            "actual_avg_test_cost": _mean(float(row["actual_test_cost"]) for row in rows),
            "actual_avg_test_count": _mean(float(row["actual_test_count"]) for row in rows),
            "excluded_run_count": len(excluded[method]), "excluded_runs": excluded[method],
        }
        summaries.append(summary)
        with (args.output_dir / f"{method}_per_case.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"match_level": "ICD-10 three-character", "methods": summaries}, handle, ensure_ascii=False, indent=2)
    if summaries:
        with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=[key for key in summaries[0] if key != "excluded_runs"])
            writer.writeheader(); writer.writerows({key: value for key, value in row.items() if key != "excluded_runs"} for row in summaries)
    print(json.dumps({"output_dir": str(args.output_dir), "methods": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
