from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class RunStore:
    """Lightweight persistent checkpoint and event storage; a run_id can be reopened safely."""

    def __init__(self, root: str | Path, run_id: str) -> None:
        self.directory = Path(root) / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.event_path = self.directory / "events.jsonl"
        self.logger = logging.getLogger(f"pomdp.{run_id}")

    @staticmethod
    def _default(value: Any) -> Any:
        return asdict(value) if is_dataclass(value) else str(value)

    def checkpoint(self, payload: dict[str, Any]) -> None:
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, default=self._default, indent=2), encoding="utf-8")
        temporary.replace(self.checkpoint_path)

    def resume(self) -> dict[str, Any] | None:
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8")) if self.checkpoint_path.exists() else None

    def event(self, name: str, **payload: Any) -> None:
        record = {"event": name, **payload}
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=self._default) + "\n")
        self.logger.info("%s %s", name, payload)
