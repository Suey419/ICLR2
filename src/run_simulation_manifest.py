"""Run every leave-one-test-out item recorded in a manifest, sequentially."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/random_50_manifest.json"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--limit", type=int, help="English text N items，English text")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["items"][:args.limit]
    results = []
    for index, item in enumerate(items, start=1):
        command = [
            sys.executable, "src/run_simulation_examples.py",
            "--case", item["file"],
            "--target-action", item["target_action"],
            "--count", "1",
            "--seed", str(args.seed),
        ]
        print(f"[{index}/{len(items)}] {item['case_id']} | {item['target_action']}", flush=True)
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        results.append({
            "index": index,
            "case_id": item["case_id"],
            "target_action": item["target_action"],
            "return_code": completed.returncode,
        })

    summary_path = args.manifest.with_name(args.manifest.stem + "_run_summary.json")
    summary_path.write_text(json.dumps({
        "manifest": str(args.manifest),
        "seed": args.seed,
        "total": len(results),
        "succeeded": sum(x["return_code"] == 0 for x in results),
        "failed": sum(x["return_code"] != 0 for x in results),
        "items": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
