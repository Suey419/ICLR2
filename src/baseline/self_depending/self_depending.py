"""A direct sequential LLM baseline with no access to unreleased results."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Mapping, Sequence
import re
from pathlib import Path

from core.types import ClinicalState
from providers.llm import LLMConfig, _Client
from baseline.self_depending.icd10 import DEFAULT_CODEBOOK, _codebook


@dataclass(frozen=True)
class DirectDecision:
    loinc_code: str | None
    stop: bool
    critical_risk: float
    rationale: str = ""
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    differential_diagnoses: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FinalDiagnosis:
    diagnosis: str
    icd10_code: str
    confidence: float
    reason: str
    top_differential_diagnoses: tuple[dict[str, Any], ...]
    request: dict[str, Any]
    response: dict[str, Any]


class SelfDependingProvider(_Client):
    """Chooses an examination from the supplied visible state only."""

    def __init__(self, config: LLMConfig, stop_confidence: float = 0.85) -> None:
        super().__init__(config, "self_depending.txt")
        self.stop_confidence = stop_confidence

    def recommend(
        self,
        state: ClinicalState,
        candidates: Sequence[Mapping[str, str]],
        audit: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> DirectDecision:
        # Deliberately construct this payload here instead of reusing CaseData:
        # CaseData also contains diagnoses, real_results and hidden_context.
        request = {
            "initial_information": state.initial_information,
            "history": [{"action": item.action, "summary": item.value, "source": item.source} for item in state.history],
            # The model orders by LOINC.  Display names are explanatory only.
            "candidate_tests": [dict(item) for item in candidates],
            "stop_confidence_threshold": self.stop_confidence,
            "minimum_required_actions": 1,
        }
        if audit:
            audit("input", {"system_prompt": self.system, "user_payload": request})
        output = self.ask(request)
        if audit:
            audit("output", output)
        if not isinstance(output, dict):
            raise ValueError("Direct_LLM must return JSON object")
        differential = output.get("differential_diagnoses")
        if not isinstance(differential, list) or not differential:
            raise ValueError("Direct_LLM must returnEnglish text differential_diagnoses")
        normalized: list[dict[str, Any]] = []
        for item in differential:
            if not isinstance(item, dict) or not str(item.get("diagnosis") or "").strip():
                raise ValueError("differential_diagnoses ofEnglish textitemsmustEnglish text diagnosis")
            try:
                probability = float(item.get("probability"))
            except (TypeError, ValueError):
                raise ValueError("differential_diagnoses ofEnglish textitemsmustEnglish text probability") from None
            if not 0 <= probability <= 1:
                raise ValueError("differential_diagnoses.probability must be in 0 to 1")
            normalized.append({"diagnosis": str(item["diagnosis"]).strip(), "probability": probability,
                               "reason": str(item.get("reason") or "")})
        stop = output.get("stop") is True
        try:
            critical_risk = float(output.get("critical_risk"))
        except (TypeError, ValueError):
            raise ValueError("Direct_LLM must return 0 to 1 of critical_risk") from None
        if not 0 <= critical_risk <= 1:
            raise ValueError("critical_risk must be in 0 to 1")
        loinc_code = output.get("loinc_code")
        candidate_codes = {item["loinc_code"] for item in candidates}
        if stop:
            if not state.history:
                raise ValueError("Direct_LLM first roundmustEnglish textitemsExamination，cannotstop")
            confidence = max(item["probability"] for item in normalized)
            if confidence < self.stop_confidence:
                raise ValueError(f"stopEnglish textDiagnosisconfidencemust >= {self.stop_confidence}")
            return DirectDecision(None, True, critical_risk, str(output.get("rationale", "")), request, output, tuple(normalized))
        if not isinstance(loinc_code, str) or loinc_code not in candidate_codes:
            raise ValueError("Direct_LLM mustEnglish text candidate_tests select from loinc_code，orEnglish text stop=true")
        return DirectDecision(loinc_code, False, critical_risk, str(output.get("rationale", "")), request, output, tuple(normalized))

    def final_diagnosis(
        self, state: ClinicalState, audit: Callable[[str, dict[str, Any]], None] | None = None,
        icd10_codebook: str | Path = DEFAULT_CODEBOOK, exclude_symptom_codes: bool = True,
    ) -> FinalDiagnosis:
        """Make a separate final call using only the information revealed so far."""
        final_client = _Client(self.config, "final_diagnosis.txt")
        book = _codebook(str(Path(icd10_codebook).resolve()))
        visible = " ".join([str(state.initial_information)] + [str(x.value) for x in state.history]).lower()
        terms = set(re.findall(r"[a-z][a-z0-9.]{2,}|[\u4e00-\u9fff]{2,}", visible))
        scored = []
        for code, name in book.items():
            if exclude_symptom_codes and code.startswith("R"):
                continue
            name_terms = set(re.findall(r"[a-z][a-z0-9.]{2,}|[\u4e00-\u9fff]{2,}", name.lower()))
            chunks = {name[i:i + 2] for i in range(max(0, len(name) - 1)) if name[i:i + 2].strip()}
            score = sum(1 for term in name_terms if term in visible or term in terms)
            score += sum(0.25 for chunk in chunks if chunk in visible)
            if name in visible:
                score += 10
            if score:
                scored.append((score, code, name))
        scored.sort(reverse=True)
        candidates = [{"code": code, "name": name} for _, code, name in scored[:40]]
        if not candidates:
            candidates = [{"code": code, "name": name} for code, name in list(book.items())[:40]]
        request = {
            "initial_information": state.initial_information,
            "history": [{"action": item.action, "summary": item.value, "source": item.source} for item in state.history],
            "coarse_candidates": candidates,
            "instruction": "onlyEnglish text coarse_candidates select fromEnglish text；English textretrieval，then rank the candidates precisely，English textofEnglish text。",
        }
        if audit:
            audit("input", {"system_prompt": final_client.system, "user_payload": request})
        output = final_client.ask(request)
        if audit:
            audit("output", output)
        if not isinstance(output, dict):
            raise ValueError("English textDiagnosisModelmust return JSON object")
        allowed = {item["code"] for item in candidates}
        raw_top3 = output.get("top_differential_diagnoses")
        if not isinstance(raw_top3, list) or len(raw_top3) != 3:
            raise ValueError("English textDiagnosisModelmust returnEnglish textGood 3 items top_differential_diagnoses")
        top3, used = [], set()
        for item in raw_top3:
            code = str(item.get("icd10_code") or "").strip().upper() if isinstance(item, Mapping) else ""
            if code not in allowed or code in used:
                raise ValueError("Top-3 ICD-10 mustEnglish text coarse_candidates")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError):
                raise ValueError("Top-3 ICD-10 mustEnglish text 0 to 1 of confidence") from None
            if not 0 <= confidence <= 1:
                raise ValueError("Top-3 ICD-10 of confidence must be in 0 to 1")
            used.add(code); top3.append({"icd10_code": code, "diagnosis": book[code], "confidence": confidence, "reason": str(item.get("reason") or "")})
        if any(top3[i]["confidence"] < top3[i + 1]["confidence"] for i in range(2)):
            raise ValueError("Top-3 ICD-10 mustEnglish textDiagnosisEnglish textsorted from high to low")
        top1 = top3[0]
        return FinalDiagnosis(top1["diagnosis"], top1["icd10_code"], top1["confidence"], top1["reason"], tuple(top3), request, output)
