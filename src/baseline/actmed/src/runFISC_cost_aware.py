#!/usr/bin/env python3
"""Run ACTMED active test selection on LEAP FISC/FHIR cases.

The adapter keeps the existing FHIR ingestion, real-result lookup, simulation
mode and ACTMED expected-KL policy, while persisting one checkpoint per
case for resumable concurrent execution.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

SCRIPT = Path(__file__).resolve()
LEAP = SCRIPT.parents[4]
WORKSPACE = LEAP.parents[1]
SRC = LEAP / "src"
ACTMED_LIB = WORKSPACE / "actmed" / "lib"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ACTMED_LIB))

from data_ingestion.fhir_bundle import discover_cases, load_case, real_results_by_loinc
from execution_environment.environment import ResultEnvironment
from providers.llm import LLMClinicalSummaryProvider, LLMSimulationProvider, LLMActionCandidateProvider, load_llm_config, load_simulation_llm_config
from baseline.self_depending.self_depending import SelfDependingProvider
from baseline.self_depending.icd10 import DEFAULT_CODEBOOK, require_icd10_cn_v2, _codebook
from core.types import ClinicalState, Observation
from baseline.actmed.src.fisc_chat import CompatibleChat


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

def coarse_actions(all_actions, case_text, limit=12):
    terms = set(case_text.lower().split())
    scored = []
    for code, row in all_actions.items():
        name = str(row.get("display_name", "")).lower()
        score = sum(1 for t in terms if len(t) > 1 and t in name)
        scored.append((score, str(code), row))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [row for _, _, row in scored[:limit]]

def global_action_catalog(paths):
    catalog = {}
    for path in paths:
        try:
            item_case = load_case(path)
        except Exception:
            continue
        for name, item in item_case.actions.items():
            codes = item.get("order_loinc") or item.get("loinc") or []
            if not codes:
                continue
            code = str(codes[0])
            catalog.setdefault(code, {"display_name": name, "price": float(item.get("price") or 0),
                                      "fields": [name], "loinc_code": code})
    return catalog


def final_diagnosis(chat_config, case, history, audit_dir: Path | None = None) -> dict:
    """Generate the same validated final diagnosis fields as Direct_LLM."""
    state = ClinicalState(case.case_id, case.initial_information,
                          tuple(Observation(x["action"], x.get("summary") or x.get("result"), x.get("source", "unknown")) for x in history), {})
    provider = SelfDependingProvider(chat_config)
    audit = None
    if audit_dir:
        def audit(stage, payload):
            atomic(audit_dir / f"final_diagnosis_{stage}.json", payload)
    result = provider.final_diagnosis(state, audit, DEFAULT_CODEBOOK)
    code, official_name = require_icd10_cn_v2(result.icd10_code, DEFAULT_CODEBOOK)
    gold_names = {str(x).strip() for x in case.diagnoses}
    gold_codes = [code for code, name in _codebook(str(Path(DEFAULT_CODEBOOK).resolve())).items() if name in gold_names]
    return {"final_diagnosis": result.diagnosis, "final_icd10_code": code,
            "final_icd10_code_system": "China Clinical ICD-10 v2.0",
            "final_icd10_official_name": official_name, "final_confidence": result.confidence,
            "final_reason": result.reason, "top3_differential_diagnoses": list(result.top_differential_diagnoses), "evaluation": {
                "gold_diagnoses": list(case.diagnoses), "gold_icd10_codes": gold_codes,
                "icd10_exact_correct": code in gold_codes}}


class FISCModel:
    """ACTMED model whose features are FHIR-orderable actions."""
    def __init__(self, chat, case_text, samples=1):
        self.model = chat
        self.model_name = chat.model
        self.case_text = case_text
        self.N = samples
        self.REFERENCE_TABLE = {}

    def format_known_data(self, known):
        if isinstance(known, pd.DataFrame):
            known = known.iloc[0].to_dict()
        return "Casecurrently visible information：\n" + self.case_text + "\nReturned examinations：\n" + "\n".join(
            f"{k}: {v}" for k, v in (known or {}).items() if str(v) not in ("", "nan", "None"))

    def predict_risk(self, known, extra_info=""):
        prompt = ("English textYesEnglish textDiagnosisEnglish text。onlyEnglish textCaseEnglish textandReturned examinations，"
                  "English textDiagnosisofconfidence。English text0to1betweenofEnglish text，do notEnglish text。\n" +
                  self.format_known_data(known) + "\nNew information：" + str(extra_info))
        try:
            return min(1.0, max(0.0, float(str(self.model(prompt, model_name=self.model_name)).strip())))
        except Exception:
            return 0.5

    def sample_random_variable(self, known, feature):
        prompt = ("English textCaseEnglish text，English textitemsExaminationEnglish textofEnglish textResult。English textResulttext，"
                  "do notEnglish text，do notEnglish textJSON。\nExamination：" + feature + "\n" + self.format_known_data(known))
        try:
            return str(self.model(prompt, model_name=self.model_name)).strip()
        except RuntimeError as error:
            # Some provider gateways reject free-text clinical simulation
            # prompts. This sample is only an ACTMED ranking surrogate, never
            # a patient observation, so retain progress with a neutral value.
            if "invalid_prompt" in str(error):
                return "Result pending confirmation"
            raise


def run_one(path: Path, args) -> dict:
    case = load_case(path)
    out = args.run_root / f"fisc_{args.policy_name}__{case.case_id}"
    checkpoint_path = out / "checkpoint.json"
    saved = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else {}
    if saved.get("completed") and saved.get("final_icd10_code"):
        return {"case_id": case.case_id, "status": "already_completed", "run": str(out)}
    history = list(saved.get("history", []))
    used = {x.get("action") for x in history}
    try:
        config = load_llm_config(args.env_file)
        sim_config = load_simulation_llm_config(args.env_file)
        os.environ.setdefault("OPENAI_BASE_URL", config.base_url)
        os.environ.setdefault("OPENAI_API_KEY", config.api_key)
        chat = CompatibleChat(config.model)
        chat.base_url = config.base_url
        chat.api_key = config.api_key
        chat.api_keys = config.api_keys or (config.api_key,)
        sim = LLMSimulationProvider(sim_config)
        summary_provider = LLMClinicalSummaryProvider(sim_config)
        env = ResultEnvironment(sim, out / "simulation_cache")
        case_text = json.dumps(case.initial_information, ensure_ascii=False)
        model = FISCModel(chat, case_text, args.samples)
        candidate_provider = LLMActionCandidateProvider(config)
        available = {code: dict(spec) for code, spec in args.global_actions.items() if code not in used}
        real = real_results_by_loinc(case)
        termination = saved.get("termination")
        for step in range(len(history), args.max_actions):
            if not available:
                termination = "no_available_actions"
                break
            known = {"Medical history": case_text}
            for item in history:
                known[item["action"]] = item.get("summary") or item.get("result") or "Examined"
            frame = pd.DataFrame([known])
            prior = model.predict_risk(frame)
            # ``predict_risk`` is a confidence estimate for the current most
            # likely diagnosis.  Permit an early stop once it is sufficient;
            # max_actions remains a ceiling, not a required test count.
            if len(history) >= args.min_actions and prior >= args.stop_confidence:
                termination = "model_confident_stop"
                atomic(out / "steps" / f"step_{step + 1:02d}_stop.json", {
                    "trace": {"step": step + 1, "prior_confidence": prior,
                              "stop_confidence": args.stop_confidence,
                              "policy": "actmed", "stop": True}
                })
                atomic(checkpoint_path, {
                    "case_id": case.case_id, "source": str(path), "history": history,
                    "termination": termination, "completed": False,
                })
                break
            coarse = coarse_actions(available, case_text, args.coarse_top_k)
            # Price must not influence ACTMED's candidate generation.  Keep
            # price in `available` for accounting, but hide it from the LLM;
            # ACTMED's decision is based only on information gain (KL).
            candidate_actions = tuple({key: value for key, value in row.items() if key != "price"}
                                      for row in coarse)
            proposed = candidate_provider.propose(
                ClinicalState(case.case_id, case.initial_information,
                              tuple(Observation(x["action"], x.get("summary") or x.get("result"), x.get("source", "unknown")) for x in history), {}),
                {}, candidate_actions, min(args.candidate_count, len(candidate_actions)))
            candidates = {code: available[code] for code in proposed if code in available}
            scored = []
            trace = {"step": step + 1, "prior_risk": prior, "candidates": []}
            for action, spec in candidates.items():
                sample = model.sample_random_variable(frame, spec["display_name"])
                posterior = model.predict_risk(frame, f"{spec['display_name']}: {sample}")
                kl = model.calculate_kl_divergence([prior], [posterior])
                price = max(float(spec["price"]), 0.01)
                utility = kl / math.sqrt(price) if args.score_mode == "sqrt_price" else kl
                scored.append((utility, action, sample, posterior, kl))
                trace["candidates"].append({"action": action, "name": spec["display_name"], "price": spec["price"], "sampled_result": sample, "posterior_risk": posterior, "kl": kl, "utility": utility, "score_mode": args.score_mode})
            _, action, sampled, _, _ = max(scored)
            spec = candidates[action]
            real_value = real.get(action)
            if real_value is not None:
                result = real_value; source = "real"
            else:
                obs = env.execute(case.case_id, action, args.seed, {}, {"requested_test": spec}, args.mode)
                result = obs.value; source = obs.source
            interactions = out / "interactions"
            summary = summary_provider.summarize(
                spec["display_name"], result,
                lambda stage, payload: atomic(interactions / f"turn_{step:02d}_summary_{stage}.json", payload),
            )
            row = {"step": step + 1, "action": action, "name": spec["display_name"], "price": spec["price"], "result": result, "summary": summary, "value": summary, "source": source}
            atomic(interactions / f"turn_{step:02d}_observation.json", {"loinc_code": action, "action": action, "value": result, "source": source})
            atomic(interactions / f"turn_{step:02d}_clinical_summary.json", {"loinc_code": action, "display_name": spec["display_name"], "summary": summary})
            history.append(row); used.add(action); available.pop(action, None)
            trace["selected_action"] = action; trace["selected_name"] = spec["display_name"]; trace["actual_result_source"] = source
            atomic(out / "steps" / f"step_{step + 1:02d}.json", {"trace": trace, "result": row})
            atomic(checkpoint_path, {"case_id": case.case_id, "source": str(path), "history": history, "completed": False})
        final = final_diagnosis(config, case, history, out / "interactions")
        completed = {"case_id": case.case_id, "source": str(path), "history": history,
                     "total_cost": sum(float(x.get("price", 0)) for x in history),
                     "termination": termination or ("max_steps" if len(history) >= args.max_actions else "completed"),
                     "policy": args.policy_name,
                     "execution_config": {"max_steps": args.max_actions, "stop_confidence": args.stop_confidence,
                                          "mode": args.mode, "seed": args.seed, "score_mode": args.score_mode},
                     **final, "completed": True}
        atomic(checkpoint_path, completed)
        return {"case_id": case.case_id, "status": "completed", "checks": len(history),
                "total_cost": completed["total_cost"], **final, "run": str(out)}
    except Exception as exc:
        atomic(checkpoint_path, {"case_id": case.case_id, "source": str(path), "history": history, "error": repr(exc), "completed": False})
        return {"case_id": case.case_id, "status": "failed", "error": repr(exc), "run": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=LEAP / "data/FISC")
    ap.add_argument("--run-root", type=Path, default=LEAP / "runs/FISC_TOP3_gpt-6-astra/actmed")
    ap.add_argument("--env-file", type=Path, default=LEAP / ".env")
    ap.add_argument("--case", help="FHIR JSON path or a unique patient-id fragment; omit for all cases")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-actions", type=int, default=5, help="Maximum number of tests; not a required count")
    ap.add_argument("--min-actions", type=int, default=3, help="Minimum number of tests before confidence-based stopping")
    ap.add_argument("--stop-confidence", type=float, default=0.9,
                    help="Stop before another test when current diagnostic confidence is at least this value")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--candidate-count", type=int, default=3)
    ap.add_argument("--coarse-top-k", type=int, default=12)
    ap.add_argument("--mode", choices=("mixed", "simulated", "real_only"), default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-name", choices=("actmed", "greedy"), default="actmed")
    ap.add_argument("--score-mode", choices=("kl", "sqrt_price"), default="kl")
    args = ap.parse_args()
    if not 0 <= args.stop_confidence <= 1:
        ap.error("--stop-confidence must be between 0 and 1")
    args.run_root.mkdir(parents=True, exist_ok=True)
    paths = list(discover_cases(args.data_root))
    if args.case:
        supplied = Path(args.case)
        matches = [path for path in paths if supplied.is_file() and path.resolve() == supplied.resolve()]
        matches += [path for path in paths if not supplied.is_file() and args.case in path.name]
        if len(matches) != 1:
            ap.error("--case mustYesCasepathorEnglish text patient-id English text")
        paths = matches
    paths = paths[:args.limit] if args.limit else paths
    args.global_actions = global_action_catalog(list(discover_cases(args.data_root)))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, path, args) for path in paths]
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
            atomic(args.run_root / "manifest.json", {"results": results, "total": len(paths)})
    print(json.dumps({"total": len(paths), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
