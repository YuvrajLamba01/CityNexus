"""MemoryStore — JSON-backed persistent store with kind/spatial/recency queries."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path

from citynexus.memory.schemas import (
    HighRiskZone,
    MemoryKind,
    MemoryRecord,
    PastFailure,
    SuccessfulStrategy,
)


_RECORD_CLASSES: dict[str, type[MemoryRecord]] = {
    "PastFailure": PastFailure,
    "SuccessfulStrategy": SuccessfulStrategy,
    "HighRiskZone": HighRiskZone,
}


class MemoryStore:
    """Persistent memory store. Pass `path` to enable file persistence."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path | None = None, *, autoload: bool = True) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._records: dict[str, MemoryRecord] = {}
        self._next_id = 1
        if autoload and self.path is not None and self.path.exists():
            self.load()

    # ----- CRUD -----------------------------------------------------------

    def add(self, record: MemoryRecord) -> str:
        if not record.id:
            record.id = f"{record.kind.value}-{self._next_id:06d}"
            self._next_id += 1
        self._records[record.id] = record
        return record.id

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def remove(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    # ----- shorthand filters ----------------------------------------------

    def by_kind(self, kind: MemoryKind) -> list[MemoryRecord]:
        return [r for r in self._records.values() if r.kind == kind]

    # ----- composed query -------------------------------------------------

    def query(
        self,
        *,
        kind: MemoryKind | None = None,
        near: tuple[int, int] | None = None,
        max_distance: int = 5,
        since: int | None = None,
        min_confidence: float = 0.0,
        current_tick: int | None = None,
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        """Composite query. `min_confidence` uses *effective* (decayed) confidence
        when `current_tick` is given, otherwise the stored value."""
        results: list[MemoryRecord] = list(self._records.values())

        if kind is not None:
            results = [r for r in results if r.kind == kind]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]

        # Confidence threshold (decayed if we have a current tick).
        if min_confidence > 0:
            if current_tick is not None:
                results = [
                    r for r in results
                    if r.effective_confidence(current_tick) >= min_confidence
                ]
            else:
                results = [r for r in results if r.confidence >= min_confidence]

        # Spatial filter — distance to any of the record's locations.
        if near is not None:
            cx, cy = near

            def _dist(rec: MemoryRecord) -> int:
                coords = self._record_coords(rec)
                if not coords:
                    return 10**9
                return min(abs(c[0] - cx) + abs(c[1] - cy) for c in coords)

            scored = [(r, _dist(r)) for r in results]
            scored = [(r, d) for r, d in scored if d <= max_distance]
            scored.sort(key=lambda rd: (rd[1], -rd[0].confidence))
            results = [r for r, _ in scored]

        if top_k is not None:
            results = results[:top_k]
        return results

    # ----- aggregation ---------------------------------------------------

    def hottest_zones(self, top_k: int = 5) -> list[HighRiskZone]:
        zones = [r for r in self.by_kind(MemoryKind.HIGH_RISK_ZONE) if isinstance(r, HighRiskZone)]
        zones.sort(key=lambda z: -z.risk_score)
        return zones[:top_k]

    def common_failure_modes(self, top_n: int = 5) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for r in self.by_kind(MemoryKind.PAST_FAILURE):
            if isinstance(r, PastFailure):
                counts[r.failure_mode] = counts.get(r.failure_mode, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        return ranked[:top_n]

    def stats(self) -> dict:
        by_kind: dict[str, int] = {}
        for r in self._records.values():
            by_kind[r.kind.value] = by_kind.get(r.kind.value, 0) + 1
        return {
            "total": len(self._records),
            "by_kind": by_kind,
            "next_id": self._next_id,
            "path": str(self.path) if self.path else None,
        }

    # ----- decay / pruning ------------------------------------------------

    def prune(self, current_tick: int, *, min_effective: float = 0.05) -> int:
        """Remove records whose effective confidence has decayed below `min_effective`."""
        removed = 0
        for rec_id in list(self._records.keys()):
            r = self._records[rec_id]
            if r.effective_confidence(current_tick) < min_effective:
                del self._records[rec_id]
                removed += 1
        return removed

    # ----- persistence ----------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("No path set on MemoryStore — pass `path` to save().")
        data = {
            "version": self.SCHEMA_VERSION,
            "next_id": self._next_id,
            "records": [self._serialize(r) for r in self._records.values()],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else self.path
        if target is None or not target.exists():
            return
        data = json.loads(target.read_text())
        self._next_id = max(self._next_id, int(data.get("next_id", 1)))
        self._records = {}
        for raw in data.get("records", []):
            rec = self._deserialize(raw)
            if rec is not None:
                self._records[rec.id] = rec

    # ----- internal helpers ----------------------------------------------

    @staticmethod
    def _record_coords(rec: MemoryRecord) -> list[tuple[int, int]]:
        if isinstance(rec, HighRiskZone):
            return list(rec.coords)
        if isinstance(rec, PastFailure) and rec.location is not None:
            return [tuple(rec.location)]
        return []

    @staticmethod
    def _to_jsonable(v):
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, tuple):
            return [MemoryStore._to_jsonable(x) for x in v]
        if isinstance(v, list):
            return [MemoryStore._to_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: MemoryStore._to_jsonable(val) for k, val in v.items()}
        return v

    @classmethod
    def _serialize(cls, record: MemoryRecord) -> dict:
        if not is_dataclass(record):
            raise TypeError(f"non-dataclass record: {type(record)}")
        out: dict = {"_class": type(record).__name__}
        for f in fields(record):
            v = getattr(record, f.name)
            out[f.name] = cls._to_jsonable(v)
        return out

    @classmethod
    def _deserialize(cls, raw: dict) -> MemoryRecord | None:
        cls_name = raw.pop("_class", None)
        record_cls = _RECORD_CLASSES.get(cls_name)
        if record_cls is None:
            return None

        # MemoryKind comes back as a string; the dataclass field has a default,
        # so accept the str and let the enum constructor convert.
        if "kind" in raw and isinstance(raw["kind"], str):
            raw["kind"] = MemoryKind(raw["kind"])

        # HighRiskZone.coords: list-of-lists → list-of-tuples.
        if record_cls is HighRiskZone and "coords" in raw:
            raw["coords"] = [tuple(c) for c in raw["coords"]]
        # PastFailure.location: list → tuple (or stays None).
        if record_cls is PastFailure and raw.get("location") is not None:
            raw["location"] = tuple(raw["location"])

        try:
            return record_cls(**raw)
        except TypeError:
            # Forward-compat: drop unknown keys.
            valid_names = {f.name for f in fields(record_cls)}
            filtered = {k: v for k, v in raw.items() if k in valid_names}
            return record_cls(**filtered)
