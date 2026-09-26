"""Validation for China's Clinical ICD-10 v2.0 six-character extension codes."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_PATTERN = re.compile(r"[A-Z][0-9]{2}\.[0-9A-Z]{3}")
# This module lives under ICLR/LEAP/src/baseline/self_depending.  The shared
# clinical ICD-10 v2 codebook is a sibling project of LEAP under ICLR/PWM_NEW.
DEFAULT_CODEBOOK = Path(__file__).resolve().parents[4] / "PWM_NEW/baseline/actmed/data/icd10_cn_clinical_v2_six.json"


@lru_cache(maxsize=8)
def _codebook(path: str) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("code_system") != "China Clinical ICD-10 v2.0":
        raise ValueError(f"ICD-10 Codebook version is incorrect: {path}")
    return {str(code).upper(): str(name) for code, name in payload["codes"].items()}


def require_icd10_cn_v2(value: Any, codebook_path: str | Path = DEFAULT_CODEBOOK) -> tuple[str, str]:
    """Return validated code and official Chinese name, or raise a clear error."""
    code = str(value or "").strip().upper()
    if not _PATTERN.fullmatch(code):
        raise ValueError(f"final_icd10_code mustYesEnglish text ICD-10 English text2.0extended code，such as K57.104: {value!r}")
    codes = _codebook(str(Path(codebook_path).resolve()))
    if code not in codes:
        raise ValueError(f"final_icd10_code is not in ICD-10 English text2.0the codebook: {code}")
    return code, codes[code]


def icd3(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())[:3]
