"""Baselines with the same execution environment as LEAP.

The batch runners are commonly launched from ``src/baseline`` as
``PYTHONPATH=. python -m self_depending.…``.  Add the project ``src`` directory
in that supported invocation so sibling packages such as ``data_ingestion``
and ``core`` remain importable.
"""
from __future__ import annotations

import sys
from pathlib import Path


_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
