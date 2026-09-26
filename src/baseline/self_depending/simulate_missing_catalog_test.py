"""Generate one missing catalog test and persist it as a FHIR result Bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog.actions import build_catalog
from core.types import Observation
from data_ingestion.fhir_bundle import discover_cases, load_case
from providers.llm import LLMSimulationProvider, load_simulation_llm_config


def _fhir_value(value):
    if isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
        return {"valueQuantity": {key: value[key] for key in ("value", "unit") if key in value}}
    return {"valueString": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}


def _bundle(case_id: str, loinc: str, display: str, observation: Observation) -> dict:
    request_id, report_id = f"sim-order-{loinc}", f"sim-report-{loinc}"
    entries = [{"resource": {"resourceType": "ServiceRequest", "id": request_id, "status": "completed", "intent": "order",
        "subject": {"reference": f"Patient/{case_id}"}, "code": {"concept": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]}}}}]
    refs = []
    for index, item in enumerate(observation.value, 1):
        obs_id = f"sim-obs-{loinc}-{index}"
        resource = {"resourceType": "Observation", "id": obs_id, "status": "final", "subject": {"reference": f"Patient/{case_id}"},
                    "code": {"text": str(item["name"])}, **_fhir_value(item["value"]), "referenceRange": item.get("reference_range", [])}
        entries.append({"resource": resource}); refs.append({"reference": f"Observation/{obs_id}"})
    entries.append({"resource": {"resourceType": "DiagnosticReport", "id": report_id, "status": "final", "subject": {"reference": f"Patient/{case_id}"},
        "basedOn": [{"reference": f"ServiceRequest/{request_id}"}], "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]}, "result": refs,
        "extension": [{"url": "https://leap.example/simulation-source", "valueString": "LLM_SIMULATED"}]}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("case"); p.add_argument("--data-root", type=Path, default=Path(__file__).parents[2]/"data/FISC")
    p.add_argument("--env-file", type=Path, default=Path(__file__).parents[2]/".env"); p.add_argument("--output-root", type=Path, default=Path(__file__).parents[2]/"runs/self_depending")
    p.add_argument("--loinc"); p.add_argument("--seed", type=int, default=0); a=p.parse_args()
    path=Path(a.case)
    if not path.is_file():
        matches=[x for x in discover_cases(a.data_root) if a.case in x.name]
        if len(matches)!=1: raise SystemExit("Case path does not exist，ormatch is not unique")
        path=matches[0]
    case=load_case(path)
    catalog=build_catalog(load_case(x) for x in discover_cases(a.data_root)).actions
    existing={code for item in case.actions.values() for code in item.get("order_loinc", [])}
    choices=[]
    for row in catalog.values():
        for code in row.get("loinc", []):
            if code not in existing: choices.append((code, row.get("aliases", [code])[0]))
    selected=next(((code,name) for code,name in choices if code==a.loinc), None) if a.loinc else (choices[0] if choices else None)
    if not selected: raise SystemExit("specified LOINC not in the unified catalog，orEnglish textCaseEnglish textYesEnglish textExamination")
    loinc,display=selected
    raw=json.loads(path.read_text(encoding="utf-8"))
    simulator=LLMSimulationProvider(load_simulation_llm_config(a.env_file))
    observation=simulator.simulate(loinc, {"complete_fhir_bundle": raw, "requested_test": {"loinc_code":loinc,"display_name":display}}, a.seed)
    out=a.output_root/f"self_depending__{case.case_id}"/"simulated_fhir_results"; out.mkdir(parents=True,exist_ok=True)
    target=out/f"{case.case_id}__{loinc}__{a.seed}.fhir.json"; target.write_text(json.dumps(_bundle(case.case_id,loinc,display,observation),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"case_id":case.case_id,"loinc_code":loinc,"display_name":display,"source":"simulated","fhir_result":str(target),"value":observation.value},ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
