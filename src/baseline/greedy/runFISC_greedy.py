"""Run the greedy FISC baseline: ACTMED-style KL divided by sqrt(price)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
ACTMED_SRC = SRC / "baseline" / "actmed" / "src"
sys.path.insert(0, str(ACTMED_SRC))

import runFISC_cost_aware as base


if __name__ == "__main__":
    sys.argv.extend(["--policy-name", "greedy", "--score-mode", "sqrt_price"])
    raise SystemExit(base.main())
