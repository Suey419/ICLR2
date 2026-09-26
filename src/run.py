"""Unified interactive launcher for the FISC policy-comparison methods.

The common execution contract lives in ``config/fisc_policy_comparison.json``.
Policies differ only in how they rank/select the next LOINC test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path

from data_ingestion.fhir_bundle import discover_cases, load_case


SRC = Path(__file__).resolve().parent


def _config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    execution = payload.get("execution", {})
    if payload.get("schema_version") != "fisc-policy-comparison-v1":
        raise ValueError("--config mustYes fisc-policy-comparison-v1")
    if not isinstance(execution.get("max_steps"), int) or execution["max_steps"] < 1:
        raise ValueError("execution.max_steps must be a positive integer")
    confidence = float(execution.get("stop_confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("execution.stop_confidence must be in 0 to 1")
    if execution.get("mode") not in {"mixed", "simulated", "real_only"}:
        raise ValueError("execution.mode invalid")
    return payload


def _select_cases(raw: str, count: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(1, count + 1))
    selected: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            selected.update(range(start, end + 1))
        elif token:
            selected.add(int(token))
    if not selected or min(selected) < 1 or max(selected) > count:
        raise ValueError("CaseEnglish text")
    return sorted(selected)


def _command(args, payload: dict, case: str, max_steps: int, confidence: float, mode: str, seed: int, run_root: Path) -> list[str]:
    """Build the policy-specific child command behind the common CLI."""
    common = ["--data-root", str(args.data_root.resolve()), "--env-file", str(args.env_file.resolve()),
              "--run-root", str(run_root.resolve()), "--mode", mode, "--seed", str(seed)]
    if args.env_sens_mode and args.policy == "full_planning":
        common.extend(["--env-sens-mode", args.env_sens_mode])
    if args.policy == "full_planning":
        spec = (SRC / payload["full_planning"]["disease_spec"]).resolve()
        command = [sys.executable, str(SRC / "run_case.py"), case, *common,
                "--disease-spec", str(spec), "--max-steps", str(max_steps),
                "--belief-threshold", str(confidence),
                "--critical-loss-threshold", str(payload["full_planning"]["critical_loss_threshold"])]
        if args.search_depth is not None:
            command.extend(["--search-depth", str(args.search_depth)])
        if args.ablation:
            command.extend(["--ablation", args.ablation])
        return command
    if args.policy == "self_depending":
        return [sys.executable, "-m", "baseline.self_depending.run_self_depending", case, *common,
                "--max-steps", str(max_steps), "--stop-confidence", str(confidence)]
    if args.policy == "primary_info":
        return [sys.executable, "-m", "baseline.primary_info.run_primary_info", case,
                "--data-root", str(args.data_root.resolve()), "--env-file", str(args.env_file.resolve()),
                "--run-root", str(run_root.resolve())]
    if args.policy == "cheapest":
        script = SRC / "baseline/actmed/src/runFISC_cheapest.py"
        extra = []
    elif args.policy == "greedy":
        script = SRC / "baseline/greedy/runFISC_greedy.py"
        extra = []
    else:
        script = SRC / "baseline/actmed/src/runFISC_cost_aware.py"
        extra = ["--policy-name", "actmed", "--score-mode", "kl"]
    return [sys.executable, str(script), "--case", case, *common,
            "--max-actions", str(max_steps), "--stop-confidence", str(confidence), "--workers", "1", *extra]


def _completed_checkpoint(policy: str, run_root: Path, path: Path) -> bool:
    """A child may report exit 0 after saving a failed adapter checkpoint."""
    case_id = load_case(path).case_id
    prefix = {"cheapest": "fisc_cheapest__", "actmed": "fisc_actmed__", "greedy": "fisc_greedy__",
              "self_depending": "self_depending__", "primary_info": "primary_info__",
              "full_planning": "full_planning__"}[policy]
    checkpoint = run_root / f"{prefix}{case_id}" / "checkpoint.json"
    try:
        return bool(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed"))
    except (OSError, json.JSONDecodeError):
        return False


def _non_retryable_checkpoint(policy: str, run_root: Path, path: Path) -> bool:
    case_id = load_case(path).case_id
    prefix = {"cheapest": "fisc_cheapest__", "actmed": "fisc_actmed__", "greedy": "fisc_greedy__",
              "self_depending": "self_depending__", "primary_info": "primary_info__",
              "full_planning": "full_planning__"}[policy]
    checkpoint = run_root / f"{prefix}{case_id}" / "checkpoint.json"
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "API HTTP 400" in json.dumps(payload, ensure_ascii=False) or "invalid_prompt" in json.dumps(payload, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True,
                        choices=("cheapest", "actmed", "greedy", "self_depending", "primary_info", "full_planning"))
    parser.add_argument("--case", help="Run one FHIR JSON path or unique patient-id fragment")
    parser.add_argument("--cases", help="Batch selection: all, 1,4,7, or 1-3,7; omit for interactive selection")
    parser.add_argument("--workers", type=int, help="Parallel cases; omit for interactive selection")
    parser.add_argument("--case-retries", type=int, default=3,
                        help="Retry only unfinished cases this many times after their first attempt")
    parser.add_argument("--config", type=Path, default=SRC / "config/fisc_policy_comparison.json")
    parser.add_argument("--data-root", type=Path, default=SRC.parent / "data/FISC")
    parser.add_argument("--env-file", type=Path, default=SRC.parent / ".env")
    parser.add_argument("--run-root", type=Path, help="Defaults to runs/FISC_TOP3_gpt-6-astra/<policy>")
    parser.add_argument("--mode", choices=("mixed", "simulated", "real_only"))
    parser.add_argument("--env-sens-mode", choices=("actual_or_unknown", "simulated_always"),
                        help="environment-sensitivity experiment mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--search-depth", type=int, help="Override Full Planning search depth only")
    parser.add_argument("--stop-confidence", type=float)
    parser.add_argument("--ablation", choices=("w_o_summary", "w_o_simulator", "w_o_retrieval", "w_o_critical_loss", "w_o_cost", "w_o_planning"))
    args = parser.parse_args()

    payload = _config(args.config)
    execution = payload["execution"]
    max_steps = args.max_steps if args.max_steps is not None else execution["max_steps"]
    confidence = args.stop_confidence if args.stop_confidence is not None else float(execution["stop_confidence"])
    mode = args.mode or execution["mode"]
    seed = args.seed if args.seed is not None else int(execution["seed"])
    if max_steps < 1 or not 0 <= confidence <= 1:
        parser.error("--max-steps must >= 1，--stop-confidence must be in 0 to 1")
    if args.search_depth is not None and args.search_depth < 1:
        parser.error("--search-depth must >= 1")
    if args.case_retries < 0:
        parser.error("--case-retries must >= 0")
    run_root = args.run_root or SRC.parent / "runs/FISC_TOP3_gpt-6-astra" / args.policy
    if args.case:
        return subprocess.run(_command(args, payload, args.case, max_steps, confidence, mode, seed, run_root), cwd=SRC, check=False).returncode

    paths = list(discover_cases(args.data_root))
    if not paths:
        parser.error("--data-root inEnglish textYes FHIR Case")
    for index, path in enumerate(paths, start=1):
        print(f"{index:>3}. {load_case(path).case_id}  {path.name}")
    try:
        selected = _select_cases(args.cases or input("Select cases（all、1,4、1-3）："), len(paths))
        workers = args.workers if args.workers is not None else int(input("Concurrent case count（default 1）：") or "1")
    except ValueError as error:
        parser.error(str(error))
    if workers < 1:
        parser.error("--workers must >= 1")
    chosen = [paths[index - 1] for index in selected]
    # Do not overwrite successful results when resuming an interrupted batch.
    completed_before = [path for path in chosen if _completed_checkpoint(args.policy, run_root, path)]
    pending = [path for path in chosen if path not in completed_before]
    print(f"English text={args.policy}，English text={len(chosen)}，English textCompletedSkipped={len(completed_before)}，pending={len(pending)}，concurrency={workers}，English textExamination={max_steps}，stopconfidence={confidence}", flush=True)
    if not pending:
        print("All cases are already completed，no run is needed。", flush=True)
        return 0
    attempts = {path: 0 for path in chosen}
    started = time.monotonic()
    round_no = 0
    while pending:
        round_no += 1
        print(f"[Progress] number  {round_no} round started：pending {len(pending)}，English text {len(chosen) - len(pending) - len(completed_before)}/{len(chosen) - len(completed_before)}", flush=True)
        retrying: list[Path] = []
        # The first pass may use the requested parallelism.  Retries are
        # deliberately serialized so a transient API/rate-limit failure in
        # one case cannot hide behind eight simultaneous retries.
        round_workers = workers if round_no == 1 else 1
        print(f"[Progress] concurrency this round={round_workers}；started cases：{', '.join(load_case(path).case_id for path in pending)}", flush=True)
        with ThreadPoolExecutor(max_workers=round_workers) as pool:
            futures = {pool.submit(subprocess.run, _command(args, payload, str(path), max_steps, confidence, mode, seed, run_root), cwd=SRC, check=False): path for path in pending}
            for future in as_completed(futures):
                completed, path = future.result(), futures[future]
                case_id = load_case(path).case_id
                if completed.returncode == 0 and _completed_checkpoint(args.policy, run_root, path):
                    print(f"[Completed] {case_id}（remaining this round {len(pending) - len(retrying) - 1}）", flush=True)
                    continue
                if _non_retryable_checkpoint(args.policy, run_root, path):
                    print(f"[Failed，do not retry HTTP 400] {case_id}", flush=True)
                    continue
                attempts[path] += 1
                if attempts[path] <= args.case_retries:
                    retrying.append(path)
                    print(f"[Failed，retry {attempts[path]}/{args.case_retries}] {case_id}", flush=True)
                else:
                    print(f"[Failed，retry limit exhausted] {case_id}", flush=True)
        pending = retrying
        print(f"[Progress] number  {round_no} round ended：Succeeded/Skipped {len(chosen) - len(pending) - len(completed_before)}，still awaiting retry {len(pending)}，Elapsed time {time.monotonic() - started:.1f}s", flush=True)
    return 1 if any(not _completed_checkpoint(args.policy, run_root, path) for path in chosen) else 0


if __name__ == "__main__":
    raise SystemExit(main())
