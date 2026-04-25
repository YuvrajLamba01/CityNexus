"""Dynamic event types: weather, accidents, roadblocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"


# Multiplier applied to road throughput.
WEATHER_CAPACITY: dict[Weather, float] = {
    Weather.CLEAR: 1.00,
    Weather.RAIN: 0.70,
    Weather.STORM: 0.40,
}

# Per road-cell, per-tick base accident probability (modulated by traffic density).
WEATHER_ACCIDENT_RATE: dict[Weather, float] = {
    Weather.CLEAR: 0.003,
    Weather.RAIN: 0.012,
    Weather.STORM: 0.035,
}

# Markov transition matrix: P(next | current). Rows sum to 1.
WEATHER_TRANSITIONS: dict[Weather, dict[Weather, float]] = {
    Weather.CLEAR: {Weather.CLEAR: 0.88, Weather.RAIN: 0.10, Weather.STORM: 0.02},
    Weather.RAIN:  {Weather.CLEAR: 0.30, Weather.RAIN: 0.60, Weather.STORM: 0.10},
    Weather.STORM: {Weather.CLEAR: 0.05, Weather.RAIN: 0.45, Weather.STORM: 0.50},
}


class Severity(int, Enum):
    MINOR = 1
    MODERATE = 2
    MAJOR = 3


# Default ticks-to-clear by severity (no responder intervention).
ACCIDENT_BASE_TTL: dict[Severity, int] = {
    Severity.MINOR: 3,
    Severity.MODERATE: 6,
    Severity.MAJOR: 12,
}


@dataclass
class Accident:
    x: int
    y: int
    severity: Severity
    ttl: int                 # ticks remaining until cleared by passive decay
    spawned_tick: int

    @property
    def coord(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Roadblock:
    x: int
    y: int
    ttl: int | None = None   # None = permanent until externally cleared
    reason: str = "external"

    @property
    def coord(self) -> tuple[int, int]:
        return (self.x, self.y)
