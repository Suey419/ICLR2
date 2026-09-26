"""Build the frozen 169-case main-experiment label manifest from FHIR rank-1 diagnoses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.icd10 import require_icd10_cn_v2
from data_ingestion.fhir_bundle import discover_cases

# Reviewed mappings from legacy/source coding to China's Clinical ICD-10 v2.0.
NORMALIZATION = {
    "K31.8806": "K26.100",  # Duodenal perforation -> acute duodenal ulcer with perforation
    "J82XX01": "J82.X01",
    "I60": "I60.900",
    "S06.6": "S06.600",
    "I26.9": "I26.900",
    "E06.1": "E06.100",
    "H46XX05": "H46.X00",
    "J47XX03": "J47.X03",
    "J67": "J67.900",
    # Hospital source extensions whose diagnosis names map to a different
    # six-character suffix in the frozen v2.0 reference table.
    "J45.101": "J45.005",  # Cough-variant asthma
    "S06.601": "S06.600",  # Traumatic subarachnoid hemorrhage
    "I61.401": "I61.400",  # Cerebellar hemorrhage
    "I26.910": "I26.900",  # Acute pulmonary embolism; no acute cor pulmonale recorded
    "I21.902": "I21.900",  # Acute myocardial infarction
    "K76.812": "K76.816",  # Hepatocellular jaundice
    "K82.002": "K83.109",  # Obstructive jaundice
    "C25.901": "C25.900", "C25.001": "C25.000", "C25.101": "C25.100",
    "K72.004": "K72.100",  # Acute-on-chronic liver failure; no separate reference entry
    "G36.001": "G36.000",  # Neuromyelitis optica (Devic disease)
    "C34.304": "C34.300", "C34.905": "C34.900", "C34.907": "C34.901",
    "C34.103": "C34.100", "C34.303": "C34.300", "C34.904": "C34.900",
    "I28.807": "I28.800",  # Connective-tissue-disease-associated pulmonary vasculitis
}
# Frozen clinical critical-disease rule.  Malignancy codes are intentionally
# excluded: a cancer diagnosis alone is not an acute critical event here.
CRITICAL_PREFIXES = ("I21", "I26", "I60", "I61", "I63", "K26", "K35", "K72", "K81", "K85", "O00", "N83")
CRITICAL_EXACT = {"K80.301"}  # Acute suppurative cholangitis


def _rank_one(bundle: dict) -> tuple[str, str]:
    matches = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Condition":
            continue
        rank = next((item.get("valuePositiveInt") for item in resource.get("extension", [])
                     if str(item.get("url", "")).endswith("diagnosis-rank")), None)
        if rank != 1:
            continue
        coding = next((item for item in resource.get("code", {}).get("coding", [])
                       if "icd-10" in str(item.get("system", "")).lower()), None)
        if coding:
            matches.append((str(coding.get("code", "")), str(coding.get("display", "") or resource.get("code", {}).get("text", ""))))
    if len(matches) != 1:
        raise ValueError(f"English textYesEnglish text diagnosis-rank=1 of ICD-10 Diagnosis，actual value {matches!r}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    changes = []
    for path in discover_cases(args.data_root):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        patient = next(resource for resource in (entry.get("resource", {}) for entry in bundle.get("entry", []))
                       if resource.get("resourceType") == "Patient")
        case_id = str(patient["id"])
        raw_code, name = _rank_one(bundle)
        first_code = raw_code.split(";", 1)[0].strip().upper()
        code = NORMALIZATION.get(first_code, first_code)
        validated, official_name = require_icd10_cn_v2(code)
        critical = validated in CRITICAL_EXACT or validated.replace(".", "")[:3] in CRITICAL_PREFIXES
        cases[case_id] = {
            "primary_icd10_code": validated,
            "primary_diagnosis_name": name.split(";", 1)[0].strip(),
            "critical": critical,
            "label_source": "FHIR Condition: Discharge diagnosis, diagnosis-rank=1",
        }
        if raw_code != validated:
            changes.append({"case_id": case_id, "source": str(path), "raw_rank1_code": raw_code,
                            "normalized_primary_icd10_code": validated, "official_name": official_name})
    payload = {
        "schema_version": "main-experiment-labels-v1", "method_name": "Direct LLM", "match_level": "exact",
        "objective": {"diagnostic_error_cost": 1000, "critical_extra_cost": 100000},
        "safety": {"critical_risk_threshold": 0.1},
        "critical_rule": {"prefixes": list(CRITICAL_PREFIXES), "exact_codes": sorted(CRITICAL_EXACT),
                          "excluded": "Malignancy alone is not classified as critical"},
        "normalization": {"mapping": NORMALIZATION, "compound_rank1_handling": "Keep the first code before the semicolon"},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = args.output.with_name(args.output.stem + "_normalization_audit.json")
    audit_path.write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "critical_cases": sum(x["critical"] for x in cases.values()),
                      "normalized_cases": len(changes), "output": str(args.output), "audit": str(audit_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
