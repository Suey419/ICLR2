"""Run Direct_LLM over several FHIR cases concurrently and resumably."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path

from data_ingestion.fhir_bundle import discover_cases, load_case


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_one(index: int, path: Path, args) -> dict[str, object]:
    case_id = load_case(path).case_id
    run_id = f"self_depending__{case_id}"
    command = [
        # The child deliberately runs from src/baseline so its module path is
        # stable.  Resolve parent-provided paths first: otherwise a path such
        # as ../data/FISC, valid for a parent launched from src, is interpreted
        # relative to src/baseline by the child and all cases fail instantly.
        sys.executable, "-m", "self_depending.run_self_depending", str(path.resolve()),
        "--env-file", str(args.env_file.resolve()), "--run-root", str(args.run_root.resolve()), "--run-id", run_id,
        "--max-steps", str(args.max_steps), "--stop-confidence", str(args.stop_confidence),
        "--mode", args.mode, "--seed", str(args.seed),
    ]
    started_at = time.time()
    completed = subprocess.run(command, cwd=Path(__file__).parents[1], check=False)
    return {"index": index, "case_id": case_id, "source": str(path), "run_id": run_id,
            "return_code": completed.returncode, "started_at": started_at, "finished_at": time.time()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[3] / "data/FISC")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[3] / ".env")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).parents[3] / "runs/self_depending")
    parser.add_argument("--limit", type=int, default=3, help="English textpathEnglish text N English textCase")
    parser.add_argument("--batch-id", default="first_3")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--stop-confidence", type=float, default=0.85)
    parser.add_argument("--mode", choices=("mixed", "simulated", "real_only"), default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="English textnumber of cases to run；default 1（English text），English text 3 English text，English text API English textandEnglish text")
    parser.add_argument("--retry-failed", action="store_true",
                        help="English textYes checkpoint English text completed English text true ofCase")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    paths = discover_cases(args.data_root)[:args.limit]
    if args.retry_failed:
        pending = []
        for path in paths:
            case_id = load_case(path).case_id
            checkpoint = args.run_root / f"self_depending__{case_id}" / "checkpoint.json"
            try:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if not saved.get("completed"):
                pending.append(path)
        paths = pending
        if not paths:
            print(json.dumps({"status": "nothing_to_retry", "run_root": str(args.run_root)}, ensure_ascii=False))
            return 0
    if not args.retry_failed and len(paths) < args.limit:
        raise SystemExit(f"English text {len(paths)} English textCase，English text --limit={args.limit}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest = args.run_root / f"batch_{args.batch_id}.json"
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, index, path, args) for index, path in enumerate(paths, start=1)]
        for future in as_completed(futures):
            record = future.result()
            results.append(record)
            _write_json(manifest, {"batch_id": args.batch_id, "limit": args.limit, "workers": args.workers,
                                   "results": sorted(results, key=lambda item: item["index"])})
            if record["return_code"]:
            # A per-case failure is checkpointed by the child and must not
            # prevent later independent patients from being evaluated.
                print(json.dumps({"status": "failed_continuing", **record}, ensure_ascii=False), file=sys.stderr)
    results.sort(key=lambda item: item["index"])
    failed = [item for item in results if item["return_code"]]
    print(json.dumps({"status": "completed_with_failures" if failed else "completed", "manifest": str(manifest), "workers": args.workers,
                      "failed_case_count": len(failed), "results": results}, ensure_ascii=False, indent=2))
    # Do not report a green shell status when every child failed before doing
    # any model work.  Individual failures are still checkpointed and do not
    # prevent the other cases from being attempted.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
