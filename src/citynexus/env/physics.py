"""Pure-function dynamics for weather, traffic, and accidents.

Every function takes the current state and returns the next-tick state. No
side effects, no global RNG — caller passes a `random.Random` instance.
"""

from __future__ import annotations

from random import Random

import numpy as np

from citynexus.city.grid import Grid
from citynexus.city.zones import Zone
from citynexus.env.events import (
    ACCIDENT_BASE_TTL,
    Accident,
    Roadblock,
    Severity,
    WEATHER_ACCIDENT_RATE,
    WEATHER_CAPACITY,
    WEATHER_TRANSITIONS,
    Weather,
)


# --- Weather ---------------------------------------------------------------

def step_weather(current: Weather, rng: Random) -> Weather:
    transitions = WEATHER_TRANSITIONS[current]
    nexts, probs = zip(*transitions.items())
    return rng.choices(nexts, weights=probs, k=1)[0]


# --- Traffic ---------------------------------------------------------------

def _time_of_day_demand(hour: int) -> tuple[float, float]:
    """Source rates (residential_outflow, commercial_outflow) per neighbouring road cell."""
    if 7 <= hour <= 9:           # morning rush
        return 0.35, 0.05
    if 17 <= hour <= 19:         # evening rush
        return 0.05, 0.35
    if 10 <= hour <= 16:         # daytime
        return 0.10, 0.15
    return 0.03, 0.02            # night


def _build_source_field(grid: Grid, hour: int) -> np.ndarray:
    """For each road cell, sum the demand contributed by adjacent zones."""
    res_src, com_src = _time_of_day_demand(hour)
    src = np.zeros((grid.height, grid.width), dtype=np.float32)
    for cell in grid.road_cells():
        for nx, ny in grid.neighbors4((cell.x, cell.y)):
            zone = grid[(nx, ny)].zone
            if zone == Zone.RESIDENTIAL:
                src[cell.y, cell.x] += res_src
            elif zone == Zone.COMMERCIAL:
                src[cell.y, cell.x] += com_src
            elif zone == Zone.INDUSTRIAL:
                src[cell.y, cell.x] += 0.08
            elif zone == Zone.HOSPITAL:
                src[cell.y, cell.x] += 0.05
    return src


def _road_mask(grid: Grid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for cell in grid.road_cells():
        mask[cell.y, cell.x] = True
    return mask


def step_traffic(
    traffic: np.ndarray,
    grid: Grid,
    weather: Weather,
    accidents: list[Accident],
    roadblocks: list[Roadblock],
    hour: int,
    *,
    decay: float = 0.85,
    diffusion: float = 0.15,
    backup_bump: float = 0.20,
) -> np.ndarray:
    """One-tick traffic field update.

    Pipeline: decay → source injection → 4-neighbour diffusion → block effects → clip.
    Non-road cells are kept at -1.0 as a sentinel.
    """
    H, W = traffic.shape
    road = _road_mask(grid)

    new = traffic.copy()
    new[road] *= decay

    # Demand from adjacent zones, scaled by weather-dependent capacity.
    capacity = WEATHER_CAPACITY[weather]
    src = _build_source_field(grid, hour) * capacity
    new[road] += src[road]

    # Symmetric diffusion (4-neighbourhood) on the road network only.
    base = np.where(road, new, 0.0)
    flow = np.zeros_like(new)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(base, shift=(dy, dx), axis=(0, 1))
        # Kill wrap-around rows/columns introduced by np.roll.
        if dy == -1:
            shifted[-1, :] = 0
        if dy == 1:
            shifted[0, :] = 0
        if dx == -1:
            shifted[:, -1] = 0
        if dx == 1:
            shifted[:, 0] = 0
        flow += shifted
    flow /= 4.0
    new[road] += diffusion * (flow[road] - base[road])

    # Hard blocks: zero capacity at the block, traffic backs up at upstream neighbours.
    blocked = {a.coord for a in accidents} | {r.coord for r in roadblocks}
    for x, y in blocked:
        if not (0 <= x < W and 0 <= y < H) or not road[y, x]:
            continue
        new[y, x] = 0.0
        for nx, ny in grid.neighbors4((x, y)):
            if road[ny, nx] and (nx, ny) not in blocked:
                new[ny, nx] = min(1.0, new[ny, nx] + backup_bump)

    new[road] = np.clip(new[road], 0.0, 1.0)
    new[~road] = -1.0
    return new


# --- Accidents -------------------------------------------------------------

def maybe_spawn_accidents(
    traffic: np.ndarray,
    grid: Grid,
    weather: Weather,
    tick: int,
    rng: Random,
    *,
    severity_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
) -> list[Accident]:
    """Sample fresh accidents on road cells; risk scales with density and weather."""
    base_rate = WEATHER_ACCIDENT_RATE[weather]
    severities = list(Severity)
    spawned: list[Accident] = []
    for cell in grid.road_cells():
        density = float(traffic[cell.y, cell.x])
        if density < 0:
            continue
        # Skip cells already blocked.
        p = base_rate * (0.3 + 0.7 * density)
        if rng.random() < p:
            severity = rng.choices(severities, weights=severity_weights, k=1)[0]
            spawned.append(
                Accident(cell.x, cell.y, severity, ACCIDENT_BASE_TTL[severity], tick)
            )
    return spawned


def decay_accidents(accidents: list[Accident]) -> list[Accident]:
    """Decrement TTL by one tick; drop accidents that have cleared."""
    out: list[Accident] = []
    for a in accidents:
        new_ttl = a.ttl - 1
        if new_ttl > 0:
            out.append(Accident(a.x, a.y, a.severity, new_ttl, a.spawned_tick))
    return out


# --- Roadblocks ------------------------------------------------------------

def decay_roadblocks(roadblocks: list[Roadblock]) -> list[Roadblock]:
    """Decrement TTL on time-bounded roadblocks; permanent ones pass through."""
    out: list[Roadblock] = []
    for rb in roadblocks:
        if rb.ttl is None:
            out.append(rb)
            continue
        new_ttl = rb.ttl - 1
        if new_ttl > 0:
            out.append(Roadblock(rb.x, rb.y, new_ttl, rb.reason))
    return out
