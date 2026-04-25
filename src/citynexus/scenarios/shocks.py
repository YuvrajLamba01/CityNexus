"""Shock taxonomy and dispatch.

A Shock is a perturbation injected into the simulation at a specific tick.
The scenario generator produces a list of these; the runner fires them on
schedule via `apply_shock`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, ClassVar

from citynexus.city.zones import Zone
from citynexus.entities.incident import Incident, IncidentKind
from citynexus.env.events import (
    ACCIDENT_BASE_TTL,
    Accident,
    Severity,
    Weather,
)

if TYPE_CHECKING:
    from citynexus.agents.base import AgentContext
    from citynexus.env.core import CityNexusEnv


class ShockKind(str, Enum):
    NOOP = "noop"
    TRAFFIC_SPIKE = "traffic_spike"
    EMERGENCY_CLUSTER = "emergency_cluster"
    BLOCKED_ROUTES = "blocked_routes"
    WEATHER_STORM = "weather_storm"
    INCIDENT_SURGE = "incident_surge"


# --- Shock hierarchy --------------------------------------------------------

@dataclass(kw_only=True)
class Shock:
    """Base shock. Subclasses set `kind` (ClassVar) and add their fields."""
    trigger_tick: int = 0
    kind: ClassVar[ShockKind] = ShockKind.NOOP


@dataclass(kw_only=True)
class TrafficSpike(Shock):
    """Boost traffic density on road cells inside a Manhattan disc."""
    kind: ClassVar[ShockKind] = ShockKind.TRAFFIC_SPIKE
    center: tuple[int, int] = (0, 0)
    radius: int = 3
    magnitude: float = 0.6        # added to current density, clamped to 1.0


@dataclass(kw_only=True)
class EmergencyCluster(Shock):
    """Spawn `count` accidents at random road cells inside a disc."""
    kind: ClassVar[ShockKind] = ShockKind.EMERGENCY_CLUSTER
    center: tuple[int, int] = (0, 0)
    radius: int = 4
    count: int = 3
    severity: int = 2             # 1=minor, 2=moderate, 3=major


@dataclass(kw_only=True)
class BlockedRoutes(Shock):
    """Drop a batch of timed roadblocks at the given road cells."""
    kind: ClassVar[ShockKind] = ShockKind.BLOCKED_ROUTES
    coords: list[tuple[int, int]] = field(default_factory=list)
    ttl: int = 15
    reason: str = "scenario:blocked_route"


@dataclass(kw_only=True)
class WeatherStorm(Shock):
    """Force weather to a specific state and lock it for `duration` ticks."""
    kind: ClassVar[ShockKind] = ShockKind.WEATHER_STORM
    force_to: str = "storm"       # "rain" | "storm" | "clear"
    duration: int = 10


@dataclass(kw_only=True)
class IncidentSurge(Shock):
    """Spawn `count` incidents (police domain) in commercial/residential zones."""
    kind: ClassVar[ShockKind] = ShockKind.INCIDENT_SURGE
    count: int = 3
    incident_kinds: list[str] = field(default_factory=lambda: ["protest", "disturbance", "theft"])
    severity: int = 2


# --- Application -----------------------------------------------------------

@dataclass
class ShockReport:
    """What an apply_shock call did. Useful for metrics + debugging."""
    kind: ShockKind
    accidents_added: int = 0
    incidents_added: int = 0
    blocks_added: int = 0
    cells_affected: int = 0
    weather_lock_until: int | None = None


def apply_shock(
    env: "CityNexusEnv",
    ctx: "AgentContext",
    shock: Shock,
    rng: Random,
) -> ShockReport:
    """Dispatch a shock to its application function. Returns a ShockReport."""
    if isinstance(shock, TrafficSpike):
        return _apply_traffic_spike(env, shock)
    if isinstance(shock, EmergencyCluster):
        return _apply_emergency_cluster(env, shock, rng)
    if isinstance(shock, BlockedRoutes):
        return _apply_blocked_routes(env, shock)
    if isinstance(shock, WeatherStorm):
        return _apply_weather_storm(env, shock)
    if isinstance(shock, IncidentSurge):
        return _apply_incident_surge(env, ctx, shock, rng)
    return ShockReport(kind=shock.kind)


def _apply_traffic_spike(env: "CityNexusEnv", shock: TrafficSpike) -> ShockReport:
    grid = env.state.grid
    traffic = env.state.traffic
    cx, cy = shock.center
    affected = 0
    for dy in range(-shock.radius, shock.radius + 1):
        for dx in range(-shock.radius, shock.radius + 1):
            if abs(dx) + abs(dy) > shock.radius:
                continue
            x, y = cx + dx, cy + dy
            if not grid.in_bounds((x, y)) or not grid.is_road((x, y)):
                continue
            traffic[y, x] = min(1.0, float(traffic[y, x]) + shock.magnitude)
            affected += 1
    return ShockReport(kind=shock.kind, cells_affected=affected)


def _apply_emergency_cluster(
    env: "CityNexusEnv", shock: EmergencyCluster, rng: Random,
) -> ShockReport:
    grid = env.state.grid
    cx, cy = shock.center
    candidates: list[tuple[int, int]] = []
    blocked_set = {a.coord for a in env.state.accidents} | {r.coord for r in env.state.roadblocks}
    for dy in range(-shock.radius, shock.radius + 1):
        for dx in range(-shock.radius, shock.radius + 1):
            if abs(dx) + abs(dy) > shock.radius:
                continue
            x, y = cx + dx, cy + dy
            if not grid.in_bounds((x, y)) or not grid.is_road((x, y)):
                continue
            if (x, y) in blocked_set:
                continue
            candidates.append((x, y))
    if not candidates:
        return ShockReport(kind=shock.kind)
    rng.shuffle(candidates)
    sev_value = max(1, min(3, shock.severity))
    sev = Severity(sev_value)
    added = 0
    for (x, y) in candidates[: shock.count]:
        env.state.accidents.append(Accident(
            x=x, y=y, severity=sev, ttl=ACCIDENT_BASE_TTL[sev],
            spawned_tick=env.state.tick,
        ))
        added += 1
    return ShockReport(kind=shock.kind, accidents_added=added, cells_affected=added)


def _apply_blocked_routes(env: "CityNexusEnv", shock: BlockedRoutes) -> ShockReport:
    added = 0
    for (x, y) in shock.coords:
        if env.add_roadblock(x, y, ttl=shock.ttl, reason=shock.reason):
            added += 1
    return ShockReport(kind=shock.kind, blocks_added=added, cells_affected=added)


def _apply_weather_storm(env: "CityNexusEnv", shock: WeatherStorm) -> ShockReport:
    try:
        target = Weather(shock.force_to)
    except ValueError:
        target = Weather.STORM
    env.state.weather = target
    return ShockReport(
        kind=shock.kind,
        weather_lock_until=env.state.tick + max(0, shock.duration),
    )


def _apply_incident_surge(
    env: "CityNexusEnv",
    ctx: "AgentContext",
    shock: IncidentSurge,
    rng: Random,
) -> ShockReport:
    grid = env.state.grid
    pool = list(grid.cells_of(Zone.COMMERCIAL)) + list(grid.cells_of(Zone.RESIDENTIAL))
    if not pool:
        return ShockReport(kind=shock.kind)
    rng.shuffle(pool)
    added = 0
    for cell in pool[: shock.count]:
        kind_str = rng.choice(shock.incident_kinds)
        try:
            kind = IncidentKind(kind_str)
        except ValueError:
            kind = IncidentKind.DISTURBANCE
        iid = f"shock-inc-{ctx.tick}-{cell.x}-{cell.y}-{added}"
        ctx.incidents[iid] = Incident(
            id=iid, kind=kind, pos=(cell.x, cell.y),
            severity=max(1, min(3, shock.severity)),
            spawned_tick=ctx.tick,
            ttl=rng.randint(8, 20),
        )
        added += 1
    return ShockReport(kind=shock.kind, incidents_added=added, cells_affected=added)
