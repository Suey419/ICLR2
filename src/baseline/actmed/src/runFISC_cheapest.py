#!/usr/bin/env python3
"""FISC baseline: stop when sufficiently confident, else select the cheapest test.

All ingestion, simulation, checkpoint and concurrent execution behavior is
shared with the FISC ACTMED adapter; only the action-selection policy differs.
"""
from __future__ import annotations

import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import runFISC_cost_aware as base


def run_one(path: Path, args) -> dict:
    case = base.load_case(path)
    out = args.run_root / f"fisc_cheapest__{case.case_id}"
    checkpoint_path = out / "checkpoint.json"
    saved = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else {}
    if saved.get("completed") and saved.get("final_icd10_code"):
        return {"case_id": case.case_id, "status": "already_completed", "run": str(out)}
    history = list(saved.get("history", []))
    used = {x.get("action") for x in history}
    try:
        config = base.load_llm_config(args.env_file)
        sim_config = base.load_simulation_llm_config(args.env_file)
        base.os.environ.setdefault("OPENAI_BASE_URL", config.base_url)
        base.os.environ.setdefault("OPENAI_API_KEY", config.api_key)
        chat = base.CompatibleChat(config.model)
        chat.base_url, chat.api_key = config.base_url, config.api_key
        chat.api_keys = config.api_keys or (config.api_key,)
        sim = base.LLMSimulationProvider(sim_config)
        summary_provider = base.LLMClinicalSummaryProvider(sim_config)
        env = base.ResultEnvironment(sim, out / "simulation_cache")
        case_text = json.dumps(case.initial_information, ensure_ascii=False)
        model = base.FISCModel(chat, case_text, args.samples)
        candidate_provider = base.LLMActionCandidateProvider(config)
        real = base.real_results_by_loinc(case)
        available = {code: dict(spec) for code, spec in args.global_actions.items() if code not in used}
        termination = saved.get("termination")
        for step in range(len(history), args.max_actions):
            if not available:
                termination = "no_available_actions"
                break
            known = {"Medical history": case_text}
            for item in history:
                known[item["action"]] = item.get("summary") or item.get("result") or "Examined"
            frame = base.pd.DataFrame([known])
            prior = model.predict_risk(frame)
            # ``predict_risk`` is an LLM estimate of confidence in the current
            # most likely diagnosis (despite its legacy name).  A high value
            # therefore permits an early, model-driven stop; max_actions is a
            # ceiling rather than a required number of tests.
            if history and prior >= args.stop_confidence:
                termination = "model_confident_stop"
                base.atomic(out / "steps" / f"step_{step + 1:02d}_stop.json", {
                    "trace": {"step": step + 1, "prior_confidence": prior,
                              "stop_confidence": args.stop_confidence,
                              "policy": "cheapest", "stop": True}
                })
                base.atomic(checkpoint_path, {
                    "case_id": case.case_id, "source": str(path), "history": history,
                    "termination": termination, "completed": False,
                })
                break
            # Coarse retrieval: keep only three candidates. Deterministic order
            # makes resume behavior stable; final ranking is strictly by price.
            coarse = base.coarse_actions(available, case_text, args.coarse_top_k)
            state = base.ClinicalState(case.case_id, case.initial_information,
                                       tuple(base.Observation(x["action"], x.get("summary") or x.get("result"), x.get("source", "unknown")) for x in history), {})
            proposed = candidate_provider.propose(state, {}, coarse, min(args.candidate_count, len(coarse)))
            candidates = {code: available[code] for code in proposed if code in available}
            trace = {"step": step + 1, "prior_risk": prior, "policy": "cheapest", "candidate_count": 3, "candidates": []}
            for action, spec in candidates.items():
                sample = model.sample_random_variable(frame, spec["display_name"])
                posterior = model.predict_risk(frame, f"{spec['display_name']}: {sample}")
                trace["candidates"].append({"action": action, "name": spec["display_name"], "price": spec["price"], "sampled_result": sample, "posterior_risk": posterior})
            action, spec = min(candidates.items(), key=lambda pair: (pair[1]["price"], pair[0]))
            real_value = real.get(action)
            if real_value is not None:
                result, source = real_value, "real"
            else:
                obs = env.execute(case.case_id, action, args.seed, {}, {"requested_test": spec}, args.mode)
                result, source = obs.value, obs.source
            interactions = out / "interactions"
            summary = summary_provider.summarize(
                spec["display_name"], result,
                lambda stage, payload: base.atomic(interactions / f"turn_{step:02d}_summary_{stage}.json", payload),
            )
            row = {"step": step + 1, "action": action, "name": spec["display_name"], "price": spec["price"], "result": result, "summary": summary, "value": summary, "source": source}
            base.atomic(interactions / f"turn_{step:02d}_observation.json", {"loinc_code": action, "action": action, "value": result, "source": source})
            base.atomic(interactions / f"turn_{step:02d}_clinical_summary.json", {"loinc_code": action, "display_name": spec["display_name"], "summary": summary})
            history.append(row); used.add(action); available.pop(action, None)
            trace["selected_action"], trace["selected_name"] = action, spec["display_name"]
            trace["actual_result_source"] = source
            base.atomic(out / "steps" / f"step_{step + 1:02d}.json", {"trace": trace, "result": row})
            base.atomic(checkpoint_path, {"case_id": case.case_id, "source": str(path), "history": history, "completed": False})
        final = base.final_diagnosis(config, case, history, out / "interactions")
        completed = {"case_id": case.case_id, "source": str(path), "history": history,
                     "total_cost": sum(float(x.get("price", 0)) for x in history),
                     "termination": termination or ("max_steps" if len(history) >= args.max_actions else "completed"),
                     "policy": "cheapest",
                     "execution_config": {"max_steps": args.max_actions, "stop_confidence": args.stop_confidence,
                                          "mode": args.mode, "seed": args.seed},
                     **final, "completed": True}
        base.atomic(checkpoint_path, completed)
        return {"case_id": case.case_id, "status": "completed", "checks": len(history),
                "total_cost": completed["total_cost"], **final, "run": str(out)}
    except Exception as exc:
        base.atomic(checkpoint_path, {"case_id": case.case_id, "source": str(path), "history": history, "error": repr(exc), "completed": False})
        return {"case_id": case.case_id, "status": "failed", "error": repr(exc), "run": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=base.LEAP / "data/FISC")
    ap.add_argument("--run-root", type=Path, default=base.LEAP / "runs/FISC_TOP3_gpt-6-astra/cheapest")
    ap.add_argument("--env-file", type=Path, default=base.LEAP / ".env")
    ap.add_argument("--case", help="FHIR JSON path or a unique patient-id fragment; omit for all cases")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-actions", type=int, default=5, help="Maximum number of tests; not a required count")
    ap.add_argument("--stop-confidence", type=float, default=0.85,
                    help="Stop before another test when current diagnostic confidence is at least this value")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--candidate-count", type=int, default=3); ap.add_argument("--coarse-top-k", type=int, default=12)
    ap.add_argument("--mode", choices=("mixed", "simulated", "real_only"), default="mixed"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.stop_confidence <= 1:
        ap.error("--stop-confidence must be between 0 and 1")
    args.run_root.mkdir(parents=True, exist_ok=True)
    paths = list(base.discover_cases(args.data_root))
    if args.case:
        supplied = Path(args.case)
        matches = [path for path in paths if supplied.is_file() and path.resolve() == supplied.resolve()]
        matches += [path for path in paths if not supplied.is_file() and args.case in path.name]
        if len(matches) != 1:
            ap.error("--case mustYesCasepathorEnglish text patient-id English text")
        paths = matches
    paths = paths[:args.limit] if args.limit else paths
    args.global_actions = base.global_action_catalog(list(base.discover_cases(args.data_root)))
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, path, args) for path in paths]
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
            base.atomic(args.run_root / "manifest.json", {"results": results, "total": len(paths)})
    print(json.dumps({"total": len(paths), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
