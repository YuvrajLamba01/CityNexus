"""Lightweight metrics logger: in-memory rolling window + optional JSONL + console."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Append-only training/eval metrics. Optional JSONL persistence for re-plotting."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        console: bool = True,
        window: int = 10,
        log_filename: str = "training.jsonl",
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path: Path | None = self.log_dir / log_filename
            # Truncate on init so each run starts clean.
            self._jsonl_path.write_text("")
        else:
            self._jsonl_path = None
        self.console = console
        self.window = window
        self._records: list[dict] = []

    # ----- write ----------------------------------------------------------

    def log(self, **fields: Any) -> None:
        record = {k: v for k, v in fields.items() if v is not None}
        self._records.append(record)
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a") as f:
                f.write(json.dumps(record, default=_jsonable) + "\n")
        if self.console:
            self._print_compact(record)

    # ----- read -----------------------------------------------------------

    def all(self) -> list[dict]:
        return list(self._records)

    def column(self, key: str) -> list[Any]:
        return [r.get(key) for r in self._records if key in r]

    def rolling_avg(self, key: str, window: int | None = None) -> float | None:
        w = window or self.window
        recent = [r.get(key) for r in self._records[-w:] if isinstance(r.get(key), (int, float))]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def export_csv(self, path: str | Path) -> None:
        if not self._records:
            return
        keys: list[str] = sorted({k for r in self._records for k in r.keys()})
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in self._records:
                writer.writerow({k: _jsonable(r.get(k)) for k in keys})

    # ----- console display ------------------------------------------------

    def _print_compact(self, r: dict) -> None:
        ep = r.get("episode")
        phase = r.get("phase", "train")
        d = r.get("difficulty")
        score = r.get("score")
        succ = r.get("delivery_success_rate")
        rew = r.get("summed_reward")
        cong = r.get("peak_congestion")
        gated = r.get("n_gated_ticks")
        bits = []
        if ep is not None:
            bits.append(f"{phase}:ep{ep:>3}")
        if d is not None:
            bits.append(f"d={d:.2f}")
        if score is not None:
            bits.append(f"score={score:.2f}")
        if succ is not None:
            bits.append(f"succ={succ:.2f}")
        if rew is not None:
            bits.append(f"R_sum={rew:+.1f}")
        if cong is not None:
            bits.append(f"cong={cong:.2f}")
        if gated is not None:
            bits.append(f"gated={gated}")
        rolling = self.rolling_avg("score")
        if rolling is not None:
            bits.append(f"avg{self.window}={rolling:.2f}")
        print("  " + " ".join(bits))


def _jsonable(v: Any) -> Any:
    """Best-effort coercion for json.dumps."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return str(v)
