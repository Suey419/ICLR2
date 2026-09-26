from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _tokens(value: str) -> tuple[str, ...]:
    """Tokenize English medical terms and Chinese names without dependencies."""
    normalized = str(value).lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
    result: list[str] = []
    for word in words:
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            result.extend(word)
            result.extend(word[index:index + 2] for index in range(len(word) - 1))
        else:
            result.append(word)
    return tuple(result)


@dataclass(frozen=True)
class RetrievedAction:
    action: dict[str, Any]
    score: float


class BM25ActionLibrary:
    """A local, deterministic BM25 index over allowed LOINC order names."""

    def __init__(self, actions: Mapping[str, Mapping[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.actions = {str(code): dict(row) for code, row in actions.items()}
        self.k1, self.b = k1, b
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for code, row in self.actions.items():
            document = " ".join((str(row.get("display_name", "")), str(code)))
            frequencies = Counter(_tokens(document))
            self.term_frequencies[code] = frequencies
            self.lengths[code] = sum(frequencies.values())
            document_frequency.update(frequencies)
        self.document_frequency = document_frequency
        self.average_length = sum(self.lengths.values()) / len(self.lengths) if self.lengths else 1.0

    def retrieve(self, queries: Sequence[str], limit: int, allowed_codes: Iterable[str] | None = None) -> tuple[RetrievedAction, ...]:
        if limit < 1:
            raise ValueError("BM25 retrieval limit must >= 1")
        allowed = set(map(str, allowed_codes)) if allowed_codes is not None else set(self.actions)
        scores: defaultdict[str, float] = defaultdict(float)
        total_documents = len(self.actions)
        for query in queries:
            query_terms = set(_tokens(query))
            for term in query_terms:
                frequency = self.document_frequency.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(1 + (total_documents - frequency + .5) / (frequency + .5))
                for code in allowed:
                    term_frequency = self.term_frequencies.get(code, {}).get(term, 0)
                    if not term_frequency:
                        continue
                    denominator = term_frequency + self.k1 * (1 - self.b + self.b * self.lengths[code] / self.average_length)
                    scores[code] += inverse_frequency * term_frequency * (self.k1 + 1) / denominator
        # Keep a deterministic fallback ordering if a query has no lexical
        # overlap; the re-ranker still receives only ``limit`` valid LOINCs.
        ranked = sorted(allowed, key=lambda code: (-scores[code], code))[:limit]
        return tuple(RetrievedAction(self.actions[code], round(scores[code], 8)) for code in ranked)
