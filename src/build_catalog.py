from __future__ import annotations

import argparse
from pathlib import Path

from catalog.actions import build_catalog
from data_ingestion.fhir_bundle import discover_cases, load_case


def main() -> int:
    parser = argparse.ArgumentParser(description="English text FISC FHIR Bundle Build a unified examination-action catalog")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parents[1] / "data/FISC")
    parser.add_argument("--output", type=Path, default=Path("catalog/actions.json"))
    args = parser.parse_args()
    cases = [load_case(path) for path in discover_cases(args.data_root)]
    catalog = build_catalog(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalog.save(args.output)
    print(f"Case={len(cases)}，English text={len(catalog.actions)}，English text={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
