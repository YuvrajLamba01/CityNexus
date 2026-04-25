"""Non-traffic incidents handled by PoliceAgent (disturbances, theft, protests)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IncidentKind(str, Enum):
    DISTURBANCE = "disturbance"
    THEFT = "theft"
    PROTEST = "protest"


@dataclass
class Incident:
    id: str
    kind: IncidentKind
    pos: tuple[int, int]
    severity: int = 1
    spawned_tick: int = 0
    ttl: int = 10
    assigned_unit: str | None = None
    resolved: bool = False

    @property
    def coord(self) -> tuple[int, int]:
        return self.pos
