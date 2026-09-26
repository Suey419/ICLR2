"""Small, side-effect-free FHIR order representation.

The planner selects an action label from a prebuilt catalogue.  This module is
the narrow boundary that turns that selected action into an orderable FHIR
ServiceRequest payload.  Network submission deliberately belongs in a later
FHIR client layer.
"""
from __future__ import annotations

from typing import Any, Mapping


LOINC_SYSTEM = "http://loinc.org"


def orderable_actions(actions: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Only actions with exactly one unambiguous ServiceRequest LOINC are orderable."""
    return tuple(name for name, data in actions.items() if len(data.get("order_loinc", ())) == 1)


def build_service_request(action: str, action_data: Mapping[str, Any], patient_id: str) -> dict[str, Any]:
    codes = tuple(action_data.get("order_loinc", ()))
    if len(codes) != 1:
        raise ValueError(f"Examination {action!r} English textYesEnglish textof ServiceRequest LOINC，cannot be ordered")
    return {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "subject": {"reference": f"Patient/{patient_id}"},
        "code": {"concept": {"coding": [{"system": LOINC_SYSTEM, "code": codes[0], "display": action}], "text": action}},
    }
