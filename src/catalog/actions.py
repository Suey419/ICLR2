from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data_ingestion.fhir_bundle import CaseData


@dataclass(frozen=True)
class ActionCatalog:
    actions: dict[str, dict]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": 1, "actions": self.actions}, ensure_ascii=False, indent=2), encoding="utf-8")


def build_catalog(cases: Iterable[CaseData]) -> ActionCatalog:
    rows: dict[str, dict] = {}
    for case in cases:
        for name, data in case.actions.items():
            codes = tuple(sorted({str(x) for x in data["loinc"] if x}))
            # LOINC is the cross-site semantic identifier; textual names are retained
            # as aliases when a Bundle lacks a mapped code.
            action_id = "loinc:" + "+".join(codes) if codes else "fhir-name:" + name
            row = rows.setdefault(action_id, {"case_count": 0, "real_result_case_count": 0, "prices": [], "loinc": [], "aliases": []})
            row["case_count"] += 1
            row["real_result_case_count"] += int(data["has_real_result"])
            if data["price"]:
                row["prices"].append(data["price"])
            row["loinc"].extend(data["loinc"])
            row["aliases"].append(name)
    for row in rows.values():
        prices = row.pop("prices")
        row["price"] = sum(prices) / len(prices) if prices else 0.0
        row["real_result_available"] = bool(row["real_result_case_count"])
        row["loinc"] = sorted({x for x in row.pop("loinc") if x})
        row["aliases"] = sorted(set(row["aliases"]))
    return ActionCatalog(dict(sorted(rows.items())))
