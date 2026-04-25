"""Persistent memory record types.

Three kinds:
  * PastFailure         — what went wrong + the conditions when it did
  * SuccessfulStrategy  — what worked + the conditions that triggered it
  * HighRiskZone        — spatial cells with elevated risk (accident/incident hotspots)

Each record carries a `confidence` and `decay_factor`; `effective_confidence(tick)`
returns the time-decayed value used by queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MemoryKind(str, Enum):
    PAST_FAILURE = "past_failure"
    SUCCESSFUL_STRATEGY = "successful_strategy"
    HIGH_RISK_ZONE = "high_risk_zone"


@dataclass(kw_only=True)
class MemoryRecord:
    """Common envelope for all memory records."""
    id: str = ""
    kind: MemoryKind = MemoryKind.PAST_FAILURE
    timestamp: int = 0          # episode tick or absolute step counter
    confidence: float = 1.0     # base confidence (in [0, 1])
    decay_factor: float = 0.99  # multiplicative per-tick decay applied at query time
    metadata: dict = field(default_factory=dict)

    def effective_confidence(self, current_tick: int) -> float:
        """Confidence after applying time decay since `timestamp`."""
        age = max(0, current_tick - self.timestamp)
        return self.confidence * (self.decay_factor ** age)


@dataclass(kw_only=True)
class PastFailure(MemoryRecord):
    """Records a verifier-detected failure plus the contextual conditions."""
    kind: MemoryKind = MemoryKind.PAST_FAILURE
    failure_mode: str = ""               # e.g. "congestion_reduced", "emergency_solved"
    description: str = ""
    context: dict = field(default_factory=dict)
    location: tuple[int, int] | None = None
    suggested_avoidance: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class SuccessfulStrategy(MemoryRecord):
    """Records what action pattern produced positive reward, under what triggers."""
    kind: MemoryKind = MemoryKind.SUCCESSFUL_STRATEGY
    role: str = ""                       # which agent
    pattern: str = ""                    # human-readable summary
    triggers: dict = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    outcome_score: float = 0.0


@dataclass(kw_only=True)
class HighRiskZone(MemoryRecord):
    """A cluster of cells with elevated risk. `risk_score` accumulates over samples."""
    kind: MemoryKind = MemoryKind.HIGH_RISK_ZONE
    coords: list[tuple[int, int]] = field(default_factory=list)
    risk_score: float = 0.0              # in [0, 1]
    risk_factors: list[str] = field(default_factory=list)
    sample_count: int = 0
