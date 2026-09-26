"""Run primary_info over multiple FHIR cases, like the other FISC baselines."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_ingestion.fhir_bundle import discover_cases, load_case


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_one(index: int, path: Path, args: argparse.Namespace) -> dict[str, object]:
    case_id = load_case(path).case_id
    run_id = f"primary_info__{case_id}"
    # Run from ``src`` so ``baseline.primary_info`` and its sibling core /
    # providers packages resolve exactly as they do for the single-case CLI.
    command = [sys.executable, "-m", "baseline.primary_info.run_primary_info", str(path.resolve()),
               "--env-file", str(args.env_file.resolve()), "--run-root", str(args.run_root.resolve()),
               "--run-id", run_id, "--model", args.model, "--icd10-codebook", str(args.icd10_codebook.resolve())]
    started_at = time.time()
    completed = subprocess.run(command, cwd=Path(__file__).parents[2], check=False)
    return {"index": index, "case_id": case_id, "source": str(path), "run_id": run_id,
            "return_code": completed.returncode, "started_at": started_at, "finished_at": time.time()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[3] / "data/FISC")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parents[3] / ".env")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).parents[3] / "runs/FISC_TOP3_gpt-6-astra/primary_info")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--icd10-codebook", type=Path, default=Path(__file__).parents[4] / "PWM_NEW/baseline/actmed/data/icd10_cn_clinical_v2_six.json")
    parser.add_argument("--limit", type=int, default=3, help="English textpathEnglish text，English text N English textCase")
    parser.add_argument("--batch-id", default="first_3")
    parser.add_argument("--workers", type=int, default=1, help="parallel case count")
    parser.add_argument("--retry-failed", action="store_true", help="rerun only incomplete cases")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        parser.error("--limit and --workers English textmust be at least 1")
    paths = list(discover_cases(args.data_root)[:args.limit])
    if args.retry_failed:
        paths = [path for path in paths if not (args.run_root / f"primary_info__{load_case(path).case_id}" / "checkpoint.json").is_file() or not json.loads((args.run_root / f"primary_info__{load_case(path).case_id}" / "checkpoint.json").read_text(encoding="utf-8")).get("completed")]
    if not paths:
        print(json.dumps({"status": "nothing_to_run", "run_root": str(args.run_root)}, ensure_ascii=False)); return 0
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest = args.run_root / f"batch_{args.batch_id}.json"
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, index, path, args) for index, path in enumerate(paths, 1)]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            _write_json(manifest, {"batch_id": args.batch_id, "limit": args.limit, "workers": args.workers,
                                   "model": args.model, "results": sorted(results, key=lambda row: row["index"])})
    results.sort(key=lambda row: row["index"])
    failed = [row for row in results if row["return_code"]]
    print(json.dumps({"status": "completed_with_failures" if failed else "completed", "manifest": str(manifest),
                      "model": args.model, "failed_case_count": len(failed), "results": results}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
