"""One-shot differential diagnosis from all non-label information in a FHIR case."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from baseline.self_depending.icd10 import DEFAULT_CODEBOOK, _codebook
from providers.llm import LLMConfig, _Client


@dataclass(frozen=True)
class PrimaryInfoDiagnosis:
    diagnosis: str
    icd10_code: str
    confidence: float
    reason: str
    top_differential_diagnoses: tuple[dict[str, Any], ...]
    request: dict[str, Any]
    response: dict[str, Any]


def _text(concept: Mapping[str, Any] | None) -> str:
    concept = concept or {}
    if concept.get("text"):
        return str(concept["text"])
    for coding in concept.get("coding", []):
        if isinstance(coding, Mapping) and (coding.get("display") or coding.get("code")):
            return str(coding.get("display") or coding["code"])
    return ""


def _clean_html(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))).strip()


def _reference(value: object) -> str:
    return str(value.get("reference", value) if isinstance(value, Mapping) else value or "")


def _observation_value(item: Mapping[str, Any]) -> Any:
    for key in ("valueQuantity", "valueCodeableConcept", "valueString", "valueBoolean", "valueInteger", "valueRange", "valueRatio"):
        if key not in item:
            continue
        value = item[key]
        if key == "valueQuantity" and isinstance(value, Mapping):
            return {name: value[name] for name in ("value", "unit", "code", "system") if name in value}
        if key == "valueCodeableConcept":
            return _text(value if isinstance(value, Mapping) else {})
        return value
    if item.get("component"):
        return [{"name": _text(part.get("code")), "value": _observation_value(part)} for part in item["component"]]
    return item.get("interpretation") or ""


def visible_case_information(path: str | Path, initial_group_information: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract clinical narrative and returned tests, excluding diagnostic labels.

    FHIR ``Condition`` resources are intentionally never returned: in FISC they
    are the evaluation labels, so including them would leak the answer.
    """
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    resources = [entry.get("resource", {}) for entry in bundle.get("entry", [])]
    by_ref = {f"{row.get('resourceType')}/{row.get('id')}": row for row in resources if row.get("id")}
    patient = next((row for row in resources if row.get("resourceType") == "Patient"), {})
    demographics: dict[str, Any] = {key: patient[key] for key in ("gender", "birthDate") if patient.get(key) is not None}
    for extension in patient.get("extension", []):
        if str(extension.get("url", "")).endswith("recorded-age"):
            demographics["age"] = extension.get("valueAge", {}).get("value")

    narrative = []
    for composition in (row for row in resources if row.get("resourceType") == "Composition"):
        for section in composition.get("section", []):
            body = _clean_html(section.get("text", {}).get("div", ""))
            if body:
                narrative.append({"section": str(section.get("title") or _text(section.get("code")) or "Unnamed record section"), "text": body})

    reports_by_request: dict[str, list[Mapping[str, Any]]] = {}
    for report in (row for row in resources if row.get("resourceType") == "DiagnosticReport"):
        for based_on in report.get("basedOn", []):
            reports_by_request.setdefault(_reference(based_on), []).append(report)
    examinations = []
    used_report_ids: set[str] = set()
    used_observation_refs: set[str] = set()
    for request in (row for row in resources if row.get("resourceType") == "ServiceRequest"):
        ref = f"ServiceRequest/{request.get('id')}"
        code = request.get("code", {})
        name = _text(code.get("concept", code) if isinstance(code, Mapping) else {}) or str(request.get("id") or "Unnamed examination")
        results = []
        for report in reports_by_request.get(ref, []):
            used_report_ids.add(str(report.get("id") or ""))
            for result_ref in report.get("result", []):
                observation_ref = _reference(result_ref)
                used_observation_refs.add(observation_ref)
                observation = by_ref.get(observation_ref, {})
                if observation:
                    results.append({"name": _text(observation.get("code")), "value": _observation_value(observation),
                                    "reference_range": observation.get("referenceRange", [])})
            conclusion = _clean_html(report.get("conclusion") or report.get("conclusionCode", {}).get("text", ""))
            if conclusion:
                results.append({"name": "Report conclusion", "value": conclusion})
        # Include orders even if no result was returned; this preserves the
        # fact that an examination was performed/requested without inventing a result.
        examinations.append({"name": name, "status": request.get("status"), "results": results})
    # Some exports contain a report or an observation without a linked
    # ServiceRequest.  They are still returned clinical evidence and must not
    # silently disappear from an all-information baseline.
    for report in (row for row in resources if row.get("resourceType") == "DiagnosticReport" and str(row.get("id") or "") not in used_report_ids):
        results = []
        for result_ref in report.get("result", []):
            observation_ref = _reference(result_ref)
            used_observation_refs.add(observation_ref)
            observation = by_ref.get(observation_ref, {})
            if observation:
                results.append({"name": _text(observation.get("code")), "value": _observation_value(observation),
                                "reference_range": observation.get("referenceRange", [])})
        conclusion = _clean_html(report.get("conclusion") or report.get("conclusionCode", {}).get("text", ""))
        if conclusion:
            results.append({"name": "Report conclusion", "value": conclusion})
        examinations.append({"name": _text(report.get("code")) or str(report.get("id") or "Unnamed examination report"),
                             "status": report.get("status"), "results": results})
    for observation in (row for row in resources if row.get("resourceType") == "Observation" and f"Observation/{row.get('id')}" not in used_observation_refs):
        examinations.append({"name": _text(observation.get("code")) or str(observation.get("id") or "Unnamed examination result"),
                             "status": observation.get("status"), "results": [{"name": _text(observation.get("code")),
                             "value": _observation_value(observation), "reference_range": observation.get("referenceRange", [])}]})
    result = {"patient": demographics, "clinical_narrative": narrative, "examinations": examinations,
              "label_exclusion": "Condition/DiagnosisEnglish textexclude，cannotEnglish textDiagnosisEnglish text。"}
    if initial_group_information:
        # This is the same permitted first-round prior passed to the sequential
        # policies.  It comes from the case directory, never from FHIR labels.
        result["initial_group_information"] = dict(initial_group_information)
    return result


class PrimaryInfoProvider(_Client):
    def __init__(self, config: LLMConfig, icd10_codebook: str | Path = DEFAULT_CODEBOOK) -> None:
        super().__init__(config, "primary_info_diagnosis.txt")
        self.icd10_codebook = Path(icd10_codebook)

    def _candidates(self, information: Mapping[str, Any]) -> list[dict[str, str]]:
        book = _codebook(str(self.icd10_codebook.resolve()))
        text = json.dumps(information, ensure_ascii=False).lower()
        grams = {text[index:index + 2] for index in range(len(text) - 1) if all("\u4e00" <= char <= "\u9fff" for char in text[index:index + 2])}
        scored = []
        for code, name in book.items():
            if code[0] in {"T", "V", "Y", "Z"}:
                continue
            name_grams = {name[index:index + 2] for index in range(len(name) - 1) if all("\u4e00" <= char <= "\u9fff" for char in name[index:index + 2])}
            score = len(grams & name_grams)
            if name.lower() in text:
                score += 30
            if score:
                scored.append((score, code, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [{"code": code, "name": name} for _, code, name in scored[:80]]
        selected_codes = {item["code"] for item in selected}
        # A sparse narrative can mention only one disease term.  Still supply
        # enough legal codes to satisfy the required three-way differential.
        for code, name in book.items():
            if len(selected) >= 80:
                break
            if code[0] not in {"T", "V", "Y", "Z"} and code not in selected_codes:
                selected.append({"code": code, "name": name})
        return selected

    def diagnose(self, information: Mapping[str, Any]) -> PrimaryInfoDiagnosis:
        candidates = self._candidates(information)
        request = {"case_information": information, "coarse_candidates": candidates,
                   "instruction": "English text case_information；English textin initial_group_information Yesfirst roundEnglish textofEnglish text，"
                                  "not a confirmed diagnosis for this case；mustEnglish text coarse_candidates English textGoodEnglish text ICD-10 English text。"}
        output = self.ask(request)
        if not isinstance(output, Mapping):
            raise ValueError("All_information must return JSON object")
        raw = output.get("top_differential_diagnoses")
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("must returnEnglish textGood 3 items top_differential_diagnoses")
        book = _codebook(str(self.icd10_codebook.resolve()))
        allowed = {item["code"] for item in candidates}
        ranked, used = [], set()
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("English textitemsEnglish textDiagnosismust be an object")
            code = str(item.get("icd10_code") or "").strip().upper()
            if code not in allowed or code not in book or code in used:
                raise ValueError("ICD-10 mustEnglish text、English text2.0English text coarse_candidates")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                raise ValueError("English textitemsEnglish textDiagnosismustYes confidence") from None
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be in 0 to 1 between")
            used.add(code)
            ranked.append({"icd10_code": code, "diagnosis": book[code], "confidence": confidence,
                           "reason": str(item.get("reason") or "")})
        if any(ranked[index]["confidence"] < ranked[index + 1]["confidence"] for index in range(2)):
            raise ValueError("English textDiagnosismustEnglish text confidence sorted from high to low")
        first = ranked[0]
        return PrimaryInfoDiagnosis(first["diagnosis"], first["icd10_code"], first["confidence"], first["reason"], tuple(ranked), request, dict(output))
