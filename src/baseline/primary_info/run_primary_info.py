"""Run the all-information diagnostic baseline on one FHIR case."""
from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

from baseline.primary_info.primary_info import PrimaryInfoProvider, visible_case_information
from baseline.self_depending.icd10 import DEFAULT_CODEBOOK, icd3, require_icd10_cn_v2
from baseline.self_depending.run_self_depending import _gold_icd10_codes
from core.run_store import RunStore
from data_ingestion.fhir_bundle import discover_cases, load_case
from providers.llm import load_llm_config


def _audit(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="FHIR JSON path or a unique patient-id fragment")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[3] / "data/FISC")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[3] / ".env")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).parents[3] / "runs/FISC_TOP3_gpt-6-astra/primary_info")
    parser.add_argument("--run-id")
    parser.add_argument("--model", default="gpt-6-astra",
                        help="diagnosis model name；default gpt-6-astra。can be explicitly overridden to match the server model ID ID")
    parser.add_argument("--icd10-codebook", type=Path, default=DEFAULT_CODEBOOK)
    args = parser.parse_args()
    path = Path(args.case)
    if not path.is_file():
        matches = [item for item in discover_cases(args.data_root) if args.case in item.name]
        if len(matches) != 1:
            raise SystemExit("Case path does not exist，or patient id match count is not 1")
        path = matches[0]
    case = load_case(path)
    store = RunStore(args.run_root, args.run_id or f"primary_info__{case.case_id}")
    saved = store.resume() or {}
    if saved.get("completed"):
        print(json.dumps({"case_id": case.case_id, "status": "already_completed", "run_directory": str(store.directory)}, ensure_ascii=False))
        return 0
    group_prior = {
        key: case.initial_information[key]
        for key in ("disease_group", "candidate_diseases", "disclosure_note")
        if key in case.initial_information
    }
    information = visible_case_information(path, group_prior)
    audit_dir = store.directory / "interactions"
    _audit(audit_dir / "primary_info_input.json", information)
    # This baseline is intentionally pinned to the experiment's GPT-6 Astra
    # setting rather than silently inheriting another baseline's DOCTOR_MODEL.
    provider = PrimaryInfoProvider(replace(load_llm_config(args.env_file), model=args.model), args.icd10_codebook)
    errors = []
    for attempt in range(1, 5):
        try:
            provider.audit = lambda stage, payload, current=attempt: _audit(audit_dir / f"diagnosis_attempt_{current}_{stage}.json", payload)
            final = provider.diagnose(information)
            code, official_name = require_icd10_cn_v2(final.icd10_code, args.icd10_codebook)
            break
        except Exception as error:
            errors.append({"attempt": attempt, "error_type": type(error).__name__, "error": str(error)})
            _audit(audit_dir / f"diagnosis_attempt_{attempt}_error.json", {"error": str(error), "traceback": traceback.format_exc()})
    else:
        result = {"case_id": case.case_id, "policy": "primary_info", "generation_status": {"status": "retry_exhausted", "errors": errors}, "completed": False}
        store.checkpoint(result); store.event("generation_retry_exhausted", errors=errors)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 2
    gold_codes = _gold_icd10_codes(path)
    result = {"case_id": case.case_id, "history": [], "differential_history": [], "termination": "primary_info_single_pass",
              "policy": "primary_info", "initial_group_information": group_prior,
              "execution_config": {"model": args.model, "information_source": "FHIR narrative + all ServiceRequest/DiagnosticReport results", "condition_resources_excluded": True,
                                   "initial_group_prior_disclosed": True},
              "total_cost": 0.0, "final_diagnosis": final.diagnosis, "final_icd10_code": code,
              "final_icd10_code_system": "China Clinical ICD-10 v2.0", "final_icd10_official_name": official_name,
              "final_confidence": final.confidence, "final_reason": final.reason,
              "top3_differential_diagnoses": list(final.top_differential_diagnoses),
              "evaluation": {"gold_icd10_codes": gold_codes, "icd10_exact_correct": code in gold_codes,
                             "icd10_three_character_correct": any(icd3(code) == icd3(gold) for gold in gold_codes)}, "completed": True}
    store.event("final_diagnosis", llm_request=final.request, llm_response=final.response, diagnosis=final.diagnosis,
                icd10_code=code, official_name=official_name, confidence=final.confidence, evaluation=result["evaluation"])
    store.checkpoint(result); store.event("completed", steps=0)
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
