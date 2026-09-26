"""FHIR R5 Bundle -> POMDP-safe case representation."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# The FISC catalog group is the only prior that may be revealed in the first
# round. Its candidate list defines the disease range, not a case diagnosis;
# later examinations must still distinguish the specific diagnosis.
GROUP_CANDIDATE_DISEASES: dict[str, tuple[str, ...]] = {
    "Group 1": ("Acute appendicitis", "Acute pancreatitis", "English text/English text/Duodenal diverticulitis", "Acute cholecystitis or cholelithiasis", "Duodenal perforation", "Gastric ulcer"),
    "Group 2": ("Acute female pelvic inflammatory disease", "Ovarian torsion", "Tubal pregnancy with hemorrhage", "Cornual pregnancy with hemorrhage"),
    "Group 3": ("Cough-variant asthma", "Gastroesophageal-reflux-related cough", "Eosinophilic-pneumonia-associated bronchitis"),
    "Group 4": ("Bronchiectasis with infection", "Pulmonary tuberculosis", "Malignant neoplasm of lung or bronchus", "Hypersensitivity pneumonitis", "Connective-tissue-disease-associated pulmonary vasculitis"),
    "Group 5": ("Acute pulmonary embolism", "Pulmonary embolism", "Acute myocardial infarction"),
    "Group 6": ("Hepatocellular or obstructive jaundice", "Malignant neoplasm of pancreas", "Common-bile-duct or intrahepatic-bile-duct stones", "Chronic hepatitis B", "Drug-induced hepatitis", "Acute or acute-on-chronic liver failure", "Bile duct obstruction or acute cholangitis"),
    "Group 7": ("Urinary tract obstruction",),
    "Group 8": ("Subacute thyroiditis", "Graves disease"),
    "Group 9": ("Cerebral infarction", "Lacunar cerebral infarction", "Cerebellar infarction", "Cerebral hemorrhage", "Subarachnoid hemorrhage"),
    "Group 10": ("Optic neuritis", "Neuromyelitis optica", "Retinal vein occlusion"),
}


def _group_initial_information(path: Path, sections: dict[str, str]) -> dict[str, Any]:
    """Return the group prior plus the clinically available first-visit history."""
    group_name = path.parent.name
    diseases = GROUP_CANDIDATE_DISEASES.get(group_name)
    if diseases is None:
        raise ValueError(f"FISC Case is not in a configured disease group：{group_name}")
    visible_section_names = ("Chief complaint", "History of present illness", "Past history", "Physical examination", "Specialist examination")
    visible_sections = {name: sections[name] for name in visible_section_names if sections.get(name)}
    return {
        "disease_group": group_name,
        "candidate_diseases": list(diseases),
        "clinical_history": visible_sections,
        "disclosure_note": "Candidate diseases are only the possible range for this group，English textYesEnglish textCaseDiagnosis。English textMedical historyandEnglish textfirst roundEnglish text；Ancillary examination、actual order results and gold-standard diagnoses are not provided。",
    }


def _text(concept: dict[str, Any] | None) -> str:
    concept = concept or {}
    if concept.get("text"):
        return str(concept["text"])
    for coding in concept.get("coding", []):
        if coding.get("display") or coding.get("code"):
            return str(coding.get("display") or coding["code"])
    return ""


def _reference(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("reference", value)
    return str(value or "")


def _observation_value(item: dict[str, Any]) -> Any:
    for key in ("valueQuantity", "valueCodeableConcept", "valueString", "valueBoolean", "valueInteger", "valueRange", "valueRatio"):
        if key not in item:
            continue
        value = item[key]
        if key == "valueQuantity":
            return {k: value[k] for k in ("value", "unit", "code", "system") if k in value}
        if key == "valueCodeableConcept":
            return _text(value)
        return value
    if item.get("component"):
        return [{"name": _text(x.get("code")), "value": _observation_value(x)} for x in item["component"]]
    return item.get("interpretation") or ""


def _clean_xhtml(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def loinc_codes(concept: dict[str, Any] | None) -> tuple[str, ...]:
    """Return the orderable LOINC codes carried by a FHIR CodeableConcept."""
    concept = concept or {}
    return tuple(sorted({
        str(coding["code"])
        for coding in concept.get("coding", [])
        if coding.get("system") == "http://loinc.org" and coding.get("code")
    }))


@dataclass(frozen=True)
class CaseData:
    case_id: str
    source_path: Path
    initial_information: dict[str, Any]
    diagnoses: tuple[str, ...]
    actions: dict[str, dict[str, Any]]
    real_results: dict[str, Any]
    hidden_context: dict[str, Any]


def real_results_by_loinc(case: CaseData) -> dict[str, Any]:
    """Map unambiguous order LOINCs to this case's recorded results."""
    results: dict[str, Any] = {}
    for action, data in case.actions.items():
        codes = data.get("order_loinc", [])
        if len(codes) == 1 and action in case.real_results:
            results.setdefault(str(codes[0]), case.real_results[action])
    return results


def load_case(path: str | Path) -> CaseData:
    path = Path(path)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    resources = [entry.get("resource", {}) for entry in bundle.get("entry", [])]
    by_ref = {f"{r.get('resourceType')}/{r.get('id')}": r for r in resources if r.get("id")}
    patient = next((r for r in resources if r.get("resourceType") == "Patient"), {})
    case_id = str(patient.get("id") or path.stem.replace(".fhir", ""))
    sections: dict[str, str] = {}
    for composition in (r for r in resources if r.get("resourceType") == "Composition"):
        for section in composition.get("section", []):
            name = str(section.get("title") or _text(section.get("code")))
            body = _clean_xhtml(str(section.get("text", {}).get("div", "")))
            if name and body:
                sections[name] = body
    demographics = {"gender": patient.get("gender")}
    for extension in patient.get("extension", []):
        if str(extension.get("url", "")).endswith("recorded-age"):
            demographics["age"] = extension.get("valueAge", {}).get("value")
    requests = {f"ServiceRequest/{r.get('id')}": r for r in resources if r.get("resourceType") == "ServiceRequest"}
    reports_by_request: dict[str, list[dict[str, Any]]] = {}
    for report in (r for r in resources if r.get("resourceType") == "DiagnosticReport"):
        for ref in report.get("basedOn", []):
            reports_by_request.setdefault(_reference(ref), []).append(report)
    charges_by_request: dict[str, list[dict[str, Any]]] = {}
    for charge in (r for r in resources if r.get("resourceType") == "ChargeItem"):
        for service in charge.get("service", []):
            charges_by_request.setdefault(_reference(service.get("reference", service)), []).append(charge)
    actions: dict[str, dict[str, Any]] = {}
    real_results: dict[str, Any] = {}
    for ref, request in requests.items():
        request_code = request.get("code", {})
        concept = request_code.get("concept", request_code) if isinstance(request_code, dict) else {}
        action = _text(concept) or _text(request_code) or request.get("id", ref)
        # A name collision is made deterministic and remains executable.
        action = str(action)
        if action in actions:
            action = f"{action} [{request.get('id')}]"
        reports = reports_by_request.get(ref, [])
        results = []
        for report in reports:
            for result_ref in report.get("result", []):
                observation = by_ref.get(_reference(result_ref), {})
                if observation:
                    results.append({"name": _text(observation.get("code")), "value": _observation_value(observation), "reference_range": observation.get("referenceRange", [])})
        charges = charges_by_request.get(ref, [])
        price = sum(float(c.get("unitPriceComponent", {}).get("amount", {}).get("value", 0)) * float(c.get("quantity", {}).get("value", 1)) for c in charges)
        # The code on ServiceRequest is the order code.  Report codes describe
        # returned panels and must not be used to create a new order.
        loinc = list(loinc_codes(concept))
        for report in reports:
            loinc.extend(c.get("code") for c in report.get("code", {}).get("coding", []) if c.get("system") == "http://loinc.org")
        actions[action] = {
            "request_reference": ref,
            "loinc": sorted({str(code) for code in loinc if code}),
            "order_loinc": list(loinc_codes(concept)),
            "price": price,
            "has_real_result": bool(results),
        }
        if results:
            real_results[action] = results
    diagnoses = tuple(_text(r.get("code")) for r in resources if r.get("resourceType") == "Condition" and _text(r.get("code")))
    initial = _group_initial_information(path, sections)
    # The execution simulator must never receive the gold-standard diagnosis
    # from Condition or any real examination result. Richer context must be
    # reviewed and de-identified during preprocessing before being added here.
    simulation_context = {"demographics": demographics}
    return CaseData(case_id, path, initial, diagnoses, actions, real_results, simulation_context)


def discover_cases(root: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(root).rglob("*.fhir.json")))
