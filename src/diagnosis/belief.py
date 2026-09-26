from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from core.types import ClinicalState


class DiagnosisProvider(Protocol):
    def predict(self, history: ClinicalState, diseases: Sequence[str]) -> Mapping[str, float]: ...


class DiagnosisBelief:
    """Preserve independently reported diagnostic confidences in [0, 1]."""

    def __init__(self, provider: DiagnosisProvider, calibrate=lambda scores: scores,
                 max_differentials: int | None = None) -> None:
        if max_differentials is not None and max_differentials < 1:
            raise ValueError("max_differentials must >= 1")
        self.provider, self.calibrate, self.max_differentials = provider, calibrate, max_differentials

    def infer(self, state: ClinicalState, diseases: Sequence[str]) -> dict[str, float]:
        scores = dict(self.calibrate(dict(self.provider.predict(state, diseases))))
        # The provider now proposes and locally validates the differential
        # ICD-10 codes itself; ``diseases`` remains in the interface for
        # backwards compatibility with earlier fixed-universe providers.
        values = {str(d).strip().upper(): float(score) for d, score in scores.items()}
        if any(value < 0 or value > 1 for value in values.values()):
            raise ValueError("Diagnosis Provider ofconfidencemust be in 0 to 1 between")
        if not any(values.values()):
            raise ValueError("Diagnosis Provider English textprobability")
        if self.max_differentials is None:
            return values
        selected = sorted(((disease, probability) for disease, probability in values.items() if probability > 0),
                          key=lambda item: item[1], reverse=True)[:self.max_differentials]
        return dict(selected)

    @staticmethod
    def preferred(belief: Mapping[str, float]) -> tuple[str, float]:
        if not belief:
            raise ValueError("DiagnosisEnglish textcannotEnglish text")
        return max(belief.items(), key=lambda item: item[1])
