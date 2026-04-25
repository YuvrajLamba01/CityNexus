"""Scenario / metrics / failure-mode schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citynexus.env.events import Weather
    from citynexus.scenarios.shocks import Shock


@dataclass
class Constraint:
    """Hard or soft rule the agents are expected to honour (informational for now)."""
    name: str
    description: str = ""


@dataclass
class Scenario:
    """One episode's specification: initial state + scheduled shocks + win/lose criteria."""
    id: str
    seed: int
    difficulty: float                  # 0.0–1.0
    initial_weather: "Weather"         # forward ref to citynexus.env.events.Weather
    shocks: list["Shock"] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    win_conditions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def shocks_at(self, tick: int) -> list["Shock"]:
        return [s for s in self.shocks if s.trigger_tick == tick]

    def summary(self) -> dict:
        from collections import Counter
        kinds = Counter(s.kind for s in self.shocks)
        return {
            "id": self.id,
            "difficulty": round(self.difficulty, 3),
            "initial_weather": getattr(self.initial_weather, "value", str(self.initial_weather)),
            "n_shocks": len(self.shocks),
            "shocks_by_kind": dict(kinds),
            "metadata": dict(self.metadata),
        }


@dataclass
class EpisodeMetrics:
    """Per-episode performance summary used by the curriculum."""
    episode_id: str
    scenario_id: str
    difficulty: float
    ticks_run: int

    # Deliveries
    deliveries_total: int = 0
    deliveries_completed: int = 0
    deliveries_failed: int = 0
    deliveries_open: int = 0

    # Accidents
    accidents_peak_concurrent: int = 0
    accidents_unresolved_at_end: int = 0

    # Incidents
    incidents_peak_concurrent: int = 0
    incidents_unresolved_at_end: int = 0

    # Traffic
    peak_congestion: float = 0.0
    avg_congestion: float = 0.0

    # Weather
    storm_ticks: int = 0

    # Communication
    messages_sent: int = 0

    # Adversarial
    shocks_fired: int = 0

    @property
    def delivery_success_rate(self) -> float:
        resolved = self.deliveries_completed + self.deliveries_failed
        if resolved == 0:
            return 1.0
        return self.deliveries_completed / resolved

    @property
    def overall_score(self) -> float:
        """Composite score in [0, 1]. Higher = better agent performance."""
        delivery_term = self.delivery_success_rate
        accident_term = max(0.0, 1.0 - self.accidents_unresolved_at_end / 8.0)
        incident_term = max(0.0, 1.0 - self.incidents_unresolved_at_end / 5.0)
        congestion_term = max(0.0, 1.0 - self.peak_congestion)
        return (
            0.40 * delivery_term
            + 0.25 * accident_term
            + 0.20 * incident_term
            + 0.15 * congestion_term
        )

    def summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "difficulty": round(self.difficulty, 3),
            "score": round(self.overall_score, 3),
            "delivery_success_rate": round(self.delivery_success_rate, 3),
            "deliveries": f"{self.deliveries_completed}/{self.deliveries_total} done, "
                          f"{self.deliveries_failed} failed, {self.deliveries_open} open",
            "accidents_peak/end": f"{self.accidents_peak_concurrent}/{self.accidents_unresolved_at_end}",
            "incidents_peak/end": f"{self.incidents_peak_concurrent}/{self.incidents_unresolved_at_end}",
            "peak_congestion": round(self.peak_congestion, 3),
            "storm_ticks": self.storm_ticks,
            "shocks_fired": self.shocks_fired,
            "messages_sent": self.messages_sent,
        }


@dataclass
class FailureMode:
    """Classifies what went wrong; suggests shock kinds that exploit the same weakness."""
    name: str
    severity: float                                # 0.0–1.0
    description: str = ""
    suggested_shock_kinds: list[str] = field(default_factory=list)
