"""Per-role partial-observability filters.

Each role gets a typed *View* dataclass that captures exactly what that agent
can see. `observe()` on each agent calls the matching `build_*_view(...)` and
encodes the result; `decide()` works strictly off the view, never touching
the full `WorldState`.

What each role sees:
  * Delivery   → roads near pending deliveries (corridor view)
  * Traffic    → intersection cells only
  * Emergency  → discs around active accidents
  * Police     → discs around incidents + smaller discs around accidents (hazards)
  * Planner    → aggregated metrics only (no per-cell, no per-entity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TYPE_CHECKING

import numpy as np

from citynexus.entities.unit import UnitKind

if TYPE_CHECKING:
    from citynexus.agents.base import AgentContext
    from citynexus.entities.delivery import Delivery
    from citynexus.entities.incident import Incident
    from citynexus.entities.unit import ResponderUnit
    from citynexus.env.events import Accident, Weather
    from citynexus.env.world_state import WorldState


Coord = tuple[int, int]


# --- Views ------------------------------------------------------------------

@dataclass
class DeliveryView:
    """Roads inside the bbox of each delivery (origin↔dest), padded by a radius."""
    visible_roads: set[Coord] = field(default_factory=set)
    blocked_in_view: set[Coord] = field(default_factory=set)
    traffic: dict[Coord, float] = field(default_factory=dict)
    deliveries: list = field(default_factory=list)
    weather: "Weather" = None  # type: ignore[assignment]
    hour_of_day: int = 0


@dataclass
class TrafficView:
    """Only intersection cells (road cells with ≥ 3 road neighbours) are visible."""
    intersections: set[Coord] = field(default_factory=set)
    intersection_traffic: dict[Coord, float] = field(default_factory=dict)
    intersection_blocks: set[Coord] = field(default_factory=set)
    weather: "Weather" = None  # type: ignore[assignment]
    hour_of_day: int = 0
    avg_traffic: float = 0.0
    max_traffic: float = 0.0
    congestion_ratio: float = 0.0


@dataclass
class EmergencyView:
    """Manhattan disc(R) around each active accident."""
    accidents: list = field(default_factory=list)
    ambulances: list = field(default_factory=list)
    visible_zones: set[Coord] = field(default_factory=set)
    traffic_in_zones: dict[Coord, float] = field(default_factory=dict)
    blocks_in_zones: set[Coord] = field(default_factory=set)
    weather: "Weather" = None  # type: ignore[assignment]


@dataclass
class PoliceView:
    """Disc(R_inc) around each incident + Disc(R_haz) around each accident."""
    incidents: list = field(default_factory=list)
    police: list = field(default_factory=list)
    accident_hazards: list = field(default_factory=list)   # locations + severity
    visible_zones: set[Coord] = field(default_factory=set)
    traffic_in_zones: dict[Coord, float] = field(default_factory=dict)
    blocks_in_zones: set[Coord] = field(default_factory=set)


@dataclass
class PlannerView:
    """Aggregated metrics only — counts, averages, ratios, weather. No per-entity."""
    weather: "Weather" = None  # type: ignore[assignment]
    tick: int = 0
    hour_of_day: int = 0
    avg_traffic: float = 0.0
    max_traffic: float = 0.0
    congestion_ratio: float = 0.0
    n_accidents: int = 0
    n_incidents: int = 0
    n_open_deliveries: int = 0
    n_units_idle: dict[str, int] = field(default_factory=dict)


# --- Geometry helpers ------------------------------------------------------

def _disc_cells(center: Coord, radius: int, w: int, h: int) -> set[Coord]:
    """Manhattan disc clipped to the grid."""
    cx, cy = center
    out: set[Coord] = set()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    out.add((x, y))
    return out


def _bbox_padded(coords: Iterable[Coord], pad: int, w: int, h: int) -> set[Coord]:
    """Axis-aligned bbox of the given coords, padded by `pad` cells, clipped."""
    cs = list(coords)
    if not cs:
        return set()
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    x0 = max(0, min(xs) - pad)
    x1 = min(w - 1, max(xs) + pad)
    y0 = max(0, min(ys) - pad)
    y1 = min(h - 1, max(ys) + pad)
    return {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}


# --- Builders --------------------------------------------------------------

def build_delivery_view(
    world: "WorldState",
    deliveries: Iterable["Delivery"],
    *,
    pad: int = 4,
) -> DeliveryView:
    grid = world.grid
    W, H = grid.width, grid.height
    visible: list[Delivery] = [d for d in deliveries if d.is_open and not d.is_assigned]
    if not visible:
        return DeliveryView(weather=world.weather, hour_of_day=world.hour_of_day)

    # Per-delivery corridor: bbox(origin, dest) padded.
    region: set[Coord] = set()
    for d in visible:
        region |= _bbox_padded([d.origin, d.dest], pad, W, H)

    visible_roads = {(x, y) for (x, y) in region if grid.is_road((x, y))}
    block_set = {a.coord for a in world.accidents} | {r.coord for r in world.roadblocks}
    blocked_in_view = visible_roads & block_set
    traffic = {(x, y): float(world.traffic[y, x]) for (x, y) in visible_roads}

    return DeliveryView(
        visible_roads=visible_roads,
        blocked_in_view=blocked_in_view,
        traffic=traffic,
        deliveries=visible,
        weather=world.weather,
        hour_of_day=world.hour_of_day,
    )


def build_traffic_view(world: "WorldState") -> TrafficView:
    grid = world.grid
    intersections: set[Coord] = set()
    for cell in grid.road_cells():
        n_road = sum(1 for n in grid.neighbors4((cell.x, cell.y)) if grid.is_road(n))
        if n_road >= 3:    # T-junction or crossroad
            intersections.add((cell.x, cell.y))

    int_traffic = {(x, y): float(world.traffic[y, x]) for (x, y) in intersections}
    block_set = {a.coord for a in world.accidents} | {r.coord for r in world.roadblocks}
    int_blocks = intersections & block_set

    if int_traffic:
        vals = np.fromiter(int_traffic.values(), dtype=np.float32)
        avg_t = float(vals.mean())
        max_t = float(vals.max())
        cong = float((vals > 0.7).mean())
    else:
        avg_t = max_t = cong = 0.0

    return TrafficView(
        intersections=intersections,
        intersection_traffic=int_traffic,
        intersection_blocks=int_blocks,
        weather=world.weather,
        hour_of_day=world.hour_of_day,
        avg_traffic=avg_t,
        max_traffic=max_t,
        congestion_ratio=cong,
    )


def build_emergency_view(
    world: "WorldState",
    ctx: "AgentContext",
    *,
    radius: int = 5,
) -> EmergencyView:
    grid = world.grid
    W, H = grid.width, grid.height

    zones: set[Coord] = set()
    for a in world.accidents:
        zones |= _disc_cells((a.x, a.y), radius, W, H)

    traffic_in = {
        (x, y): float(world.traffic[y, x])
        for (x, y) in zones
        if grid.is_road((x, y))
    }
    block_set = {a.coord for a in world.accidents} | {r.coord for r in world.roadblocks}
    blocks_in = zones & block_set
    ambulances = [u for u in ctx.units.values() if u.kind == UnitKind.AMBULANCE]

    return EmergencyView(
        accidents=list(world.accidents),
        ambulances=ambulances,
        visible_zones=zones,
        traffic_in_zones=traffic_in,
        blocks_in_zones=blocks_in,
        weather=world.weather,
    )


def build_police_view(
    world: "WorldState",
    ctx: "AgentContext",
    *,
    incident_radius: int = 4,
    hazard_radius: int = 2,
) -> PoliceView:
    grid = world.grid
    W, H = grid.width, grid.height

    zones: set[Coord] = set()
    for inc in ctx.incidents.values():
        zones |= _disc_cells(inc.pos, incident_radius, W, H)
    for a in world.accidents:
        zones |= _disc_cells((a.x, a.y), hazard_radius, W, H)

    traffic_in = {
        (x, y): float(world.traffic[y, x])
        for (x, y) in zones
        if grid.is_road((x, y))
    }
    block_set = {a.coord for a in world.accidents} | {r.coord for r in world.roadblocks}
    blocks_in = zones & block_set
    police = [u for u in ctx.units.values() if u.kind == UnitKind.POLICE_CAR]

    return PoliceView(
        incidents=list(ctx.incidents.values()),
        police=police,
        accident_hazards=list(world.accidents),
        visible_zones=zones,
        traffic_in_zones=traffic_in,
        blocks_in_zones=blocks_in,
    )


def build_planner_view(world: "WorldState", ctx: "AgentContext") -> PlannerView:
    by_kind: dict[str, int] = {}
    for u in ctx.units.values():
        if u.is_idle:
            by_kind[u.kind.value] = by_kind.get(u.kind.value, 0) + 1
    open_deliveries = sum(1 for d in ctx.deliveries.values() if d.is_open)
    return PlannerView(
        weather=world.weather,
        tick=world.tick,
        hour_of_day=world.hour_of_day,
        avg_traffic=world.avg_traffic(),
        max_traffic=world.max_traffic(),
        congestion_ratio=world.congestion_ratio(),
        n_accidents=len(world.accidents),
        n_incidents=len(ctx.incidents),
        n_open_deliveries=open_deliveries,
        n_units_idle=by_kind,
    )
