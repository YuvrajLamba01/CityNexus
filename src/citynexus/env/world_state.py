"""WorldState — immutable-ish snapshot of the city at a single tick."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from citynexus.city.grid import Grid
from citynexus.env.events import Accident, Roadblock, Weather


@dataclass
class WorldState:
    tick: int
    grid: Grid                          # static layout (shared across ticks)
    traffic: np.ndarray                 # shape (H, W); [0,1] on road cells, -1 elsewhere
    weather: Weather
    accidents: list[Accident] = field(default_factory=list)
    roadblocks: list[Roadblock] = field(default_factory=list)

    @property
    def hour_of_day(self) -> int:
        return self.tick % 24

    @property
    def day(self) -> int:
        return self.tick // 24

    def is_blocked(self, x: int, y: int) -> bool:
        coord = (x, y)
        return any(a.coord == coord for a in self.accidents) or any(
            r.coord == coord for r in self.roadblocks
        )

    def avg_traffic(self) -> float:
        road_mask = self.traffic >= 0
        return float(self.traffic[road_mask].mean()) if road_mask.any() else 0.0

    def max_traffic(self) -> float:
        road_mask = self.traffic >= 0
        return float(self.traffic[road_mask].max()) if road_mask.any() else 0.0

    def congestion_ratio(self, threshold: float = 0.7) -> float:
        """Fraction of road cells whose density exceeds `threshold`."""
        road_mask = self.traffic >= 0
        if not road_mask.any():
            return 0.0
        return float((self.traffic[road_mask] > threshold).mean())

    def snapshot(self) -> dict:
        """Lightweight serializable summary (suitable for observations / logging)."""
        return {
            "tick": self.tick,
            "day": self.day,
            "hour_of_day": self.hour_of_day,
            "weather": self.weather.value,
            "n_accidents": len(self.accidents),
            "n_roadblocks": len(self.roadblocks),
            "avg_traffic": self.avg_traffic(),
            "max_traffic": self.max_traffic(),
            "congestion_ratio": self.congestion_ratio(),
        }
