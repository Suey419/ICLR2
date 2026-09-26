"""Independent OpenAI-compatible providers; secrets stay in the process environment."""
from __future__ import annotations

import json
import os
import time
import threading
import urllib.request
import re
from functools import lru_cache
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.types import ClinicalState, Observation, OutcomeBranch
from prompts import load_prompt
from baseline.self_depending.icd10 import DEFAULT_CODEBOOK, require_icd10_cn_v2, _codebook


# Planning explores several hypothetical branches.  Its providers share this
# limiter so an AND-OR expansion cannot burst requests into the gateway.
_REQUEST_PACING_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0
_KEY_ROTATION_LOCK = threading.Lock()


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 60
    api_keys: tuple[str, ...] = ()


def _load_llm_config(
    env_file: str | Path | None,
    model_keys: tuple[str, ...],
    api_key_names: tuple[str, ...],
    base_url_names: tuple[str, ...],
    role: str,
) -> LLMConfig:
    if env_file:
        _load_dotenv(Path(env_file))
    # Read the base key plus every numbered sibling (e.g. DOCTOR_API_KEY1..
    # DOCTOR_API_KEY8) so adding credentials in .env requires no code change.
    prefixes = tuple(dict.fromkeys(re.sub(r"\d+$", "", name) for name in api_key_names))
    discovered: list[str] = []
    for prefix in prefixes:
        names = [prefix] + [f"{prefix}{index}" for index in range(1, 65)]
        discovered.extend(os.getenv(name) for name in names if os.getenv(name))
    keys = tuple(dict.fromkeys(discovered))
    if not keys:
        raise RuntimeError(f"not configured {role} API key；please set {' or '.join(api_key_names)}")
    model = next((os.getenv(name) for name in model_keys if os.getenv(name)), None)
    if not model:
        raise RuntimeError(f"not configured {role} Model；please set {' or '.join(model_keys)}")
    base_url = next((os.getenv(name) for name in base_url_names if os.getenv(name)), "https://api.openai.com/v1")
    try:
        timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    except ValueError:
        raise RuntimeError("LLM_TIMEOUT_SECONDS must be a positive integer") from None
    if timeout_seconds <= 0:
        raise RuntimeError("LLM_TIMEOUT_SECONDS must be a positive integer")
    return LLMConfig(keys[0], base_url.rstrip("/"), model, timeout_seconds, keys)


def load_llm_config(env_file: str | Path | None = None) -> LLMConfig:
    """Diagnosis/planning config with ordered DOCTOR_API_KEY failover."""
    return _load_llm_config(
        env_file, ("DOCTOR_MODEL", "OPENAI_MODEL", "MODEL"),
        ("DOCTOR_API_KEY",),
        ("DOCTOR_API_KEY_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"), "diagnosis/planning",
    )


def load_simulation_llm_config(env_file: str | Path | None = None) -> LLMConfig:
    """Load execution-simulation model configuration, preferring PROVIDERS_MODEL.

    ``PROVIDERS_API_KEY`` is kept as the primary name for compatibility;
    numbered keys are accepted as additional credentials for rotation.
    """
    return _load_llm_config(
        env_file,
        ("PROVIDERS_MODEL", "SIMULATION_MODEL", "PATIENT_MODEL", "OPENAI_SIMULATION_MODEL", "OPENAI_MODEL", "MODEL"),
        ("PROVIDERS_API_KEY",),
        ("PROVIDERS_API_KEY_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"), "execution simulation",
    )


def load_ablation_llm_config(env_file: str | Path | None = None) -> LLMConfig:
    """Dedicated key/channel; automatically rotates ABLATION_API_KEY[1..6]."""
    return _load_llm_config(
        env_file,
        ("ABLATION_MODEL", "DOCTOR_MODEL", "OPENAI_MODEL", "MODEL"),
        ("ABLATION_API_KEY",),
        ("ABLATION_BASE_URL",), "ablation experiment",
    )


class _Client:
    def __init__(self, config: LLMConfig, prompt_name: str) -> None:
        self.config, self.system = config, load_prompt(prompt_name)
        self._next_key = 0
        # Optional local audit hook.  It is deliberately unset by default so
        # callers explicitly opt in before persisting clinical payloads.
        self.audit: Callable[[str, dict[str, Any]], None] | None = None

    @staticmethod
    def _pace_request() -> None:
        """Apply a process-wide minimum gap between outbound model requests."""
        global _LAST_REQUEST_AT
        try:
            interval = float(os.getenv("LLM_MIN_REQUEST_INTERVAL_SECONDS", "1"))
        except ValueError:
            raise RuntimeError("LLM_MIN_REQUEST_INTERVAL_SECONDS must be non-negative") from None
        if interval < 0:
            raise RuntimeError("LLM_MIN_REQUEST_INTERVAL_SECONDS must be non-negative")
        with _REQUEST_PACING_LOCK:
            wait = interval - (time.monotonic() - _LAST_REQUEST_AT)
            if wait > 0:
                time.sleep(wait)
            _LAST_REQUEST_AT = time.monotonic()

    def _audit(self, stage: str, payload: dict[str, Any]) -> None:
        if self.audit:
            self.audit(stage, payload)

    def ask(self, payload: dict[str, Any]) -> Any:
        self._audit("request", {"system_prompt": self.system, "user_payload": payload})
        safe_payload = json.dumps(payload, ensure_ascii=False).replace("G-English text", "English textNegativeEnglish text").replace("English text", "BacteriaEnglish text")
        body = json.dumps({"model": self.config.model, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": safe_payload}]}).encode()
        last_error: Exception | None = None
        api_keys = self.config.api_keys or (self.config.api_key,)
        with _KEY_ROTATION_LOCK:
            start_key = self._next_key % len(api_keys)
            self._next_key = (self._next_key + 1) % len(api_keys)
        for attempt in range(3):
            try:
                api_key = api_keys[(start_key + attempt) % len(api_keys)]
                request = urllib.request.Request(
                    self.config.base_url + "/chat/completions", data=body,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                )
                self._pace_request()
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                last_error = error
                retry_with_next_key = len(api_keys) > 1 and error.code in {401, 403, 429, 500, 502, 503, 504}
                if error.code not in {429, 500, 502, 503, 504} and not retry_with_next_key or attempt == 2:
                    # The gateway's response is the only reliable explanation
                    # for a 400 (unsupported parameter, context limit, etc.).
                    # Keep it bounded so audit/error files remain safe to read.
                    try:
                        detail = error.read().decode("utf-8", errors="replace").strip()
                    except OSError:
                        detail = ""
                    suffix = f"：{detail[:2000]}" if detail else ""
                    raise RuntimeError(f"Model service request failed（HTTP {error.code}，Model：{self.config.model}）{suffix}") from error
            except urllib.error.URLError as error:
                last_error = error
                if attempt == 2:
                    raise RuntimeError(f"Unable to connect to model service（Model：{self.config.model}）：{error.reason}") from error
            time.sleep(2 ** attempt)
        else:  # Defensive: the loop always returns or breaks.
            raise RuntimeError(f"Model service request failed（Model：{self.config.model}）：{last_error}")
        content = data["choices"][0]["message"]["content"]
        self._audit("raw_response", {"content": content})
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            self._audit("json_parse_error", {"content": content, "error": str(error)})
            raise
        self._audit("parsed_response", {"response": parsed})
        return parsed


def _state(state: ClinicalState) -> dict[str, Any]:
    return {"case_id": state.case_id, "initial_information": state.initial_information, "history": [{"action": x.action, "value": x.value, "source": x.source} for x in state.history]}


class LLMDiagnosisProvider(_Client):
    def __init__(self, config: LLMConfig, differential_count: int = 3,
                 icd10_codebook: str | Path = DEFAULT_CODEBOOK) -> None:
        super().__init__(config, "diagnosis_belief.txt")
        self.differential_count, self.icd10_codebook = differential_count, Path(icd10_codebook)

    @staticmethod
    def _text(state: ClinicalState) -> str:
        """Compact searchable text from only currently visible information."""
        parts = [str(state.initial_information)]
        parts.extend(str(item.value) for item in state.history)
        return " ".join(parts).lower()

    def _coarse_candidates(self, state: ClinicalState, diseases: Sequence[str]) -> list[dict[str, str]]:
        """Recall legal ICD-10 codes first; the model may only rank this set."""
        book = _codebook(str(self.icd10_codebook.resolve()))
        text = self._text(state)
        # Do not force a frozen/global disease universe into every case.  That
        # produced irrelevant candidates (for example cardiac or neoplasm
        # codes for a diverticulitis case), which in turn made every reported
        # confidence artificially tiny.  `diseases` is retained only for the
        # backwards-compatible interface; candidates are refreshed from the
        # currently visible case text on every inference call.
        selected: list[str] = []
        tokens = set(re.findall(r"[a-z][a-z0-9.]{2,}|[\u4e00-\u9fff]{2,}", text))
        # Chinese clinical terms are often embedded in long strings (e.g.
        # ``cecocolic diverticulitis``), while codebook names use variants such
        # as ``colonic diverticulitis``. Bigram overlap recalls these variants
        # much better than comparing only greedy full Chinese spans.
        text_bigrams = {text[i:i + 2] for i in range(len(text) - 1)
                        if all("\u4e00" <= ch <= "\u9fff" for ch in text[i:i + 2])}
        generic = {"Examination", "personal history", "family history", "other", "status", "medical", "long-term", "recent", "Not seen", "Normal", "general"}
        scored = []
        for code, name in book.items():
            # Coarse retrieval is disease-name retrieval only.  Exclude
            # history/status/external-cause chapters (Z/T/V/Y), which are not
            # differential diseases and previously polluted the candidate set.
            if not code[:1].isalpha() or code[:1] in {"R", "Z", "T", "V", "Y"}:
                continue
            if code in selected:
                continue
            name_tokens = set(re.findall(r"[a-z][a-z0-9.]{2,}|[\u4e00-\u9fff]{2,}", name.lower()))
            name_bigrams = {name[i:i + 2] for i in range(len(name) - 1)
                            if all("\u4e00" <= ch <= "\u9fff" for ch in name[i:i + 2])}
            overlap = sum(3 for token in name_tokens
                          if token in text or token in tokens
                          if token not in generic)
            overlap += sum(1 for chunk in name_bigrams & text_bigrams
                           if chunk not in generic)
            if name and name in text:
                overlap += 20
            if overlap:
                scored.append((overlap, code, name))
        scored.sort(reverse=True)
        for _, code, _ in scored[: max(20, self.differential_count * 4)]:
            if code not in selected:
                selected.append(code)
        if not selected:
            selected = list(book)[: max(20, self.differential_count * 4)]
        return [{"code": code, "name": book[code]} for code in selected[: max(20, self.differential_count * 4)]]

    def predict(self, state: ClinicalState, diseases: list[str] | tuple[str, ...]) -> dict[str, float]:
        candidates = self._coarse_candidates(state, diseases)
        output = self.ask({"state": _state(state), "coarse_candidates": candidates,
                           "differential_count": self.differential_count,
                           "instruction": "may only select from coarse_candidates.code select from；English textretrievalEnglish text，then rank the candidates precisely。"})
        if not isinstance(output, Mapping):
            raise ValueError("Diagnosis Provider must return JSON object")
        # ``probabilities`` is the current contract.  Accept the former
        # direct mapping as a backwards-compatible fallback for providers
        # that cached or retained the previous prompt format.
        probabilities = output.get("probabilities", output)
        if not isinstance(probabilities, Mapping):
            raise ValueError("Diagnosis Provider of probabilities must be an object")
        allowed = {item["code"] for item in candidates}
        ranked = []
        for original, score in probabilities.items():
            code = str(original).strip().upper()
            if code not in allowed:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= value <= 1:
                ranked.append((code, value))
        # Deterministic legal fallback keeps planning alive if the model emits
        # an out-of-candidate code or malformed score.
        if not ranked:
            ranked = [(item["code"], 1.0 / (index + 1)) for index, item in enumerate(candidates[:self.differential_count])]
        ranked = sorted(dict(ranked).items(), key=lambda item: item[1], reverse=True)[:self.differential_count]
        return dict(ranked)


class LLMActionCandidateProvider(_Client):
    """Propose valid LOINC orders before the AND-OR planner scores them."""

    def __init__(self, config: LLMConfig) -> None: super().__init__(config, "planning_candidates.txt")

    def propose(self, state: ClinicalState, differentials: Mapping[str, float],
                available_actions: Sequence[Mapping[str, Any]], count: int) -> tuple[str, ...]:
        output = self.ask({"state": _state(state), "differential_diagnoses": dict(differentials),
                           "candidate_tests": list(available_actions), "candidate_count": count})
        if not isinstance(output, Mapping) or not isinstance(output.get("loinc_codes"), list):
            raise ValueError("candidate examination Provider must return an object containing loinc_codes listof JSON object")
        codes = tuple(str(code).strip() for code in output["loinc_codes"])
        valid = {str(action["loinc_code"]) for action in available_actions}
        if len(codes) != count or not all(codes) or len(set(codes)) != len(codes):
            raise ValueError(f"candidate examination Provider must return {count} distinct LOINC English text")
        invalid = [code for code in codes if code not in valid]
        if invalid:
            raise ValueError(f"candidate examination Provider returned a code outside the catalog LOINC English text：{invalid}")
        return codes

    def propose_batch(self, requests: Sequence[Mapping[str, Any]], count: int) -> tuple[tuple[str, ...], ...]:
        """Generate candidates for several visible states in one model call.

        Each request has its own state, belief, and legal candidate tests.  No
        request may contain hidden case results.  The response is indexed by
        the request's ``id`` so the planner can put each result back into its
        branch-local cache.
        """
        payload = []
        for request in requests:
            payload.append({"id": str(request["id"]), "state": _state(request["state"]),
                            "differential_diagnoses": dict(request["differentials"]),
                            "candidate_tests": list(request["available_actions"]),
                            "candidate_count": count})
        output = self.ask({"batch": payload, "instruction": "English text batch inEnglish textcandidate examination；onlyEnglish textitems candidate_tests select from。"})
        if not isinstance(output, Mapping) or not isinstance(output.get("results"), list):
            raise ValueError("batch candidate Provider must return an object containing results listof JSON object")
        by_id = {str(item.get("id")): item for item in output["results"] if isinstance(item, Mapping)}
        result = []
        for request in requests:
            item = by_id.get(str(request["id"]))
            if item is None or not isinstance(item.get("loinc_codes"), list):
                raise ValueError(f"batch candidate Provider missingEnglish text {request['id']} ofResult")
            codes = tuple(str(code).strip() for code in item["loinc_codes"])
            valid = {str(action["loinc_code"]) for action in request["available_actions"]}
            if len(codes) != count or len(set(codes)) != count or not all(code in valid for code in codes):
                raise ValueError(f"batch candidate Provider English text {request['id']} ofEnglish text")
            result.append(codes)
        return tuple(result)


class LLMTestQueryProvider(_Client):
    """Generate concrete test-name queries for local BM25 retrieval."""

    def __init__(self, config: LLMConfig) -> None: super().__init__(config, "planning_queries.txt")

    def generate(self, state: ClinicalState, differentials: Mapping[str, float], count: int) -> tuple[str, ...]:
        output = self.ask({"state": _state(state), "differential_diagnoses": dict(differentials), "query_count": count})
        if not isinstance(output, Mapping) or not isinstance(output.get("queries"), list):
            raise ValueError("examination retrieval Provider must return an object containing queries listof JSON object")
        queries = tuple(str(query).strip() for query in output["queries"])
        if len(queries) != count or not all(queries) or len(set(queries)) != len(queries):
            raise ValueError(f"examination retrieval Provider must return {count} English textofEnglish text")
        return queries

    def generate_batch(self, requests: Sequence[Mapping[str, Any]], count: int) -> tuple[tuple[str, ...], ...]:
        payload = [{"id": str(request["id"]), "state": _state(request["state"]),
                    "differential_diagnoses": dict(request["differentials"]),
                    "query_count": count} for request in requests]
        output = self.ask({"batch": payload,
                           "instruction": "English text batch inEnglish textretrieval query；onlyEnglish text。"})
        if not isinstance(output, Mapping) or not isinstance(output.get("results"), list):
            raise ValueError("batch retrieval Provider must return an object containing results listof JSON object")
        by_id = {str(item.get("id")): item for item in output["results"] if isinstance(item, Mapping)}
        result = []
        for request in requests:
            item = by_id.get(str(request["id"]))
            if item is None or not isinstance(item.get("queries"), list):
                raise ValueError(f"batch retrieval Provider missingEnglish text {request['id']} ofResult")
            queries = tuple(str(query).strip() for query in item["queries"])
            if len(queries) != count or len(set(queries)) != count or not all(queries):
                raise ValueError(f"batch retrieval Provider English text {request['id']} ofEnglish text query")
            result.append(queries)
        return tuple(result)


class LLMOutcomeProvider(_Client):
    def __init__(self, config: LLMConfig) -> None: super().__init__(config, "planning_outcome.txt")
    def predict(self, state: ClinicalState, action: str, branch_count: int,
                expected_result_schema: Sequence[Mapping[str, Any]] = ()) -> tuple[OutcomeBranch, ...]:
        output = self.ask({"state": _state(state), "action": action, "branch_count": branch_count,
                           "expected_result_schema": list(expected_result_schema)})
        if not isinstance(output, Mapping):
            raise ValueError("result prediction Provider must return JSON object")
        branches = output.get("branches")
        if not isinstance(branches, list):
            raise ValueError("result prediction Provider must return branches list")
        if len(branches) != branch_count:
            raise ValueError(f"result prediction Provider must return {branch_count} English textbranch，actual value {len(branches)}")
        for item in branches:
            if not isinstance(item, Mapping) or not isinstance(item.get("value"), list) or not item["value"]:
                raise ValueError("result prediction Provider ofEnglish textbranch value must be a non-empty list")
            returned = {str(field.get("name")): field for field in item["value"] if isinstance(field, Mapping)}
            expected_names = {str(field["name"]) for field in expected_result_schema}
            if expected_names and set(returned) != expected_names:
                raise ValueError("result prediction Provider English textoffieldmustEnglish text expected_result_schema")
            for expected in expected_result_schema:
                field = returned.get(str(expected["name"]))
                if not field or "value" not in field:
                    raise ValueError(f"result prediction Provider missingfield {expected['name']}")
                value = field["value"]
                if expected.get("value_kind") == "numeric" and not isinstance(value, Mapping):
                    raise ValueError(f"result prediction Provider ofnumeric field {expected['name']} must be an object")
                if expected.get("value_kind") == "text" and not isinstance(value, str):
                    raise ValueError(f"result prediction Provider oftext field {expected['name']} must be a string")
        return tuple(OutcomeBranch(Observation(action, item["value"], "predicted"), float(item.get("probability", 0))) for item in branches)


class LLMSimulationProvider(_Client):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, "execution_simulation.txt")
        # Optional audit hook.  The execution environment never exposes this
        # payload to the decision model.
        self.audit: Callable[[str, dict[str, Any]], None] | None = None

    def simulate(self, action: str, hidden_context: dict[str, Any], seed: int) -> Observation:
        request = {"action": action, "hidden_context": hidden_context, "seed": seed}
        if self.audit:
            self.audit("input", {"system_prompt": self.system, "user_payload": request})
        output = self.ask(request)
        if self.audit:
            self.audit("output", output)
        value = output.get("value")
        if not isinstance(value, list) or not value:
            raise ValueError("English textof value must be a non-empty list")
        for item in value:
            if not isinstance(item, Mapping) or not str(item.get("name", "")).strip() or "value" not in item:
                raise ValueError("English textResultitemsEnglish textmustEnglish text name and value")
            item_value = item["value"]
            if item_value is None or (isinstance(item_value, str) and not item_value.strip()):
                raise ValueError("English textExaminationEnglish text")
            if isinstance(item_value, str) and any(token in item_value.replace(" ", "") for token in ("Not provided", "No data", "cannot determine", "Unknown result")):
                raise ValueError("English textexamination result")
        expected_schema = hidden_context.get("expected_result_schema", [])
        if expected_schema:
            returned = {str(item["name"]): item["value"] for item in value}
            expected_names = {str(item["name"]) for item in expected_schema}
            if set(returned) != expected_names:
                missing = expected_names - set(returned)
                extra = set(returned) - expected_names
                raise ValueError(f"English textResultfieldandEnglish text schema English text；missing={sorted(missing)}，extra={sorted(extra)}")
            for expected in expected_schema:
                item_value = returned[str(expected["name"])]
                if expected["value_kind"] == "numeric":
                    if not isinstance(item_value, Mapping) or not isinstance(item_value.get("value"), (int, float)):
                        raise ValueError(f"numeric field {expected['name']} did not return a numeric value")
                elif expected["value_kind"] == "text" and not isinstance(item_value, str):
                    raise ValueError(f"text field {expected['name']} did not return text")
        return Observation(action, value, "simulated")


class LLMClinicalSummaryProvider(_Client):
    """Separately compresses already-visible results; never sees hidden context."""
    def __init__(self, config: LLMConfig) -> None: super().__init__(config, "clinical_result_summary.txt")

    def summarize(self, action: str, value: Any, audit: Callable[[str, dict[str, Any]], None] | None = None) -> str:
        request = {"action": action, "result": value}
        if audit:
            audit("input", {"system_prompt": self.system, "user_payload": request})
        output = self.ask(request)
        if audit:
            audit("output", output)
        summary = str(output.get("summary") or "").strip()
        if not summary:
            # Preserve progress when a provider returns valid JSON but omits
            # the optional compression field. The raw result is already
            # visible and is safer than failing the whole patient run.
            if isinstance(value, (str, int, float)):
                summary = str(value)
            else:
                summary = json.dumps(value, ensure_ascii=False)
            if not summary.strip():
                raise ValueError("Result summarizer did not return summary，English textthe raw result is empty")
            if audit:
                audit("effective_summary", {"summary": summary, "fallback_used": True,
                                            "reason": "provider_returned_empty_summary"})
        elif audit:
            audit("effective_summary", {"summary": summary, "fallback_used": False})
        return summary
