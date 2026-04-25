"""CityNexusEnv — the engine façade. Owns lifecycle (reset/step/state)."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from citynexus.city.grid import Grid
from citynexus.city.zones import Zone
from citynexus.env import physics
from citynexus.env.events import Accident, Roadblock, Weather
from citynexus.env.world_state import WorldState


@dataclass
class EnvConfig:
    width: int = 20
    height: int = 20
    seed: int = 42
    road_spacing: int = 4
    hospital_count: int = 1
    initial_weather: Weather = Weather.CLEAR
    initial_traffic: float = 0.05
    max_ticks: int = 200


@dataclass
class StepInfo:
    """Per-tick summary returned from `step()`."""
    tick: int
    weather: Weather
    new_accidents: list[Accident]
    cleared_accidents: int
    active_accidents: int
    active_roadblocks: int
    avg_traffic: float
    congestion_ratio: float
    metrics: dict = field(default_factory=dict)


class CityNexusEnv:
    """Self-contained city engine.

    Usage:
        env = CityNexusEnv(EnvConfig(width=20, height=20, seed=7))
        state = env.reset()
        for _ in range(50):
            info = env.step()
            ...
        print(env.render_ascii())

    Scenarios and external agents inject perturbations via the
    `add_roadblock` / `clear_roadblock` / `clear_accident` methods.
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self._rng: Random | None = None
        self._state: WorldState | None = None

    # ----- lifecycle -------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> WorldState:
        s = self.config.seed if seed is None else seed
        self._rng = Random(s)

        grid = Grid.generate(
            self.config.width,
            self.config.height,
            seed=s,
            road_spacing=self.config.road_spacing,
            hospital_count=self.config.hospital_count,
        )

        traffic = np.full((grid.height, grid.width), -1.0, dtype=np.float32)
        for cell in grid.road_cells():
            traffic[cell.y, cell.x] = self.config.initial_traffic

        self._state = WorldState(
            tick=0,
            grid=grid,
            traffic=traffic,
            weather=self.config.initial_weather,
            accidents=[],
            roadblocks=[],
        )
        return self._state

    def step(self) -> StepInfo:
        if self._state is None or self._rng is None:
            raise RuntimeError("Call reset() before step().")
        prev = self._state

        # 1. Weather transition.
        new_weather = physics.step_weather(prev.weather, self._rng)

        # 2. Traffic field update (uses *previous* accidents/roadblocks for backup effects).
        new_traffic = physics.step_traffic(
            prev.traffic, prev.grid, new_weather,
            prev.accidents, prev.roadblocks,
            hour=prev.hour_of_day,
        )

        # 3. Spawn new accidents from the updated traffic field.
        spawned = physics.maybe_spawn_accidents(
            new_traffic, prev.grid, new_weather, prev.tick + 1, self._rng,
        )

        # 4. Decay existing accidents and time-bounded roadblocks.
        decayed_accidents = physics.decay_accidents(prev.accidents)
        cleared = len(prev.accidents) - len(decayed_accidents)
        decayed_roadblocks = physics.decay_roadblocks(prev.roadblocks)

        # 5. Commit new state.
        self._state = WorldState(
            tick=prev.tick + 1,
            grid=prev.grid,
            traffic=new_traffic,
            weather=new_weather,
            accidents=decayed_accidents + spawned,
            roadblocks=decayed_roadblocks,
        )

        return StepInfo(
            tick=self._state.tick,
            weather=new_weather,
            new_accidents=spawned,
            cleared_accidents=cleared,
            active_accidents=len(self._state.accidents),
            active_roadblocks=len(self._state.roadblocks),
            avg_traffic=self._state.avg_traffic(),
            congestion_ratio=self._state.congestion_ratio(),
        )

    @property
    def state(self) -> WorldState:
        if self._state is None:
            raise RuntimeError("Environment not initialised; call reset().")
        return self._state

    @property
    def done(self) -> bool:
        return self._state is not None and self._state.tick >= self.config.max_ticks

    # ----- external injection (scenarios + future agent layer) -------------

    def add_roadblock(
        self, x: int, y: int, *, ttl: int | None = None, reason: str = "external"
    ) -> bool:
        """Place a roadblock on a road cell. Returns True if placed."""
        if self._state is None:
            raise RuntimeError("reset() first.")
        if not self._state.grid.is_road((x, y)):
            return False
        if any(rb.coord == (x, y) for rb in self._state.roadblocks):
            return False
        self._state.roadblocks.append(Roadblock(x, y, ttl=ttl, reason=reason))
        return True

    def clear_roadblock(self, x: int, y: int) -> bool:
        """Remove a roadblock at (x, y). Returns True if one was removed."""
        if self._state is None:
            raise RuntimeError("reset() first.")
        before = len(self._state.roadblocks)
        self._state.roadblocks = [
            rb for rb in self._state.roadblocks if rb.coord != (x, y)
        ]
        return len(self._state.roadblocks) < before

    def clear_accident(self, x: int, y: int) -> bool:
        """Resolve an accident at (x, y) (e.g. via dispatched responders)."""
        if self._state is None:
            raise RuntimeError("reset() first.")
        before = len(self._state.accidents)
        self._state.accidents = [
            a for a in self._state.accidents if a.coord != (x, y)
        ]
        return len(self._state.accidents) < before

    # ----- visualisation ---------------------------------------------------

    def render_ascii(self) -> str:
        if self._state is None:
            return "<env not reset>"
        s = self._state
        zone_glyph = {
            Zone.EMPTY: " ",
            Zone.RESIDENTIAL: "R",
            Zone.COMMERCIAL: "C",
            Zone.HOSPITAL: "H",
            Zone.INDUSTRIAL: "I",
        }
        accident_set = {a.coord for a in s.accidents}
        block_set = {r.coord for r in s.roadblocks}
        rows: list[str] = []
        for y in range(s.grid.height):
            row_chars: list[str] = []
            for x in range(s.grid.width):
                if (x, y) in accident_set:
                    row_chars.append("X")
                elif (x, y) in block_set:
                    row_chars.append("B")
                else:
                    cell = s.grid[(x, y)]
                    if cell.zone == Zone.ROAD:
                        t = float(s.traffic[y, x])
                        if t < 0.2:
                            row_chars.append(".")
                        elif t < 0.4:
                            row_chars.append(":")
                        elif t < 0.6:
                            row_chars.append("+")
                        elif t < 0.8:
                            row_chars.append("*")
                        else:
                            row_chars.append("#")
                    else:
                        row_chars.append(zone_glyph[cell.zone])
            rows.append("".join(row_chars))
        header = (
            f"tick={s.tick:>3} hour={s.hour_of_day:>2} "
            f"weather={s.weather.value:<5} accidents={len(s.accidents)} "
            f"blocks={len(s.roadblocks)} avg_traffic={s.avg_traffic():.2f}"
        )
        return header + "\n" + "\n".join(rows)
