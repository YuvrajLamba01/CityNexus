"""DeliveryAgent — partial obs: roads inside the bbox of pending deliveries (padded)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from random import Random

import numpy as np

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent, NoOp
from citynexus.agents.messages import Advisory, EmergencyPriority, RouteBlocked
from citynexus.agents.observability import DeliveryView, build_delivery_view
from citynexus.agents.spaces import Box, DictSpace, Discrete, Space
from citynexus.env.world_state import WorldState
from citynexus.memory.schemas import HighRiskZone, MemoryKind


# --- Constants ---------------------------------------------------------------

MAX_PENDING = 8
FEAT_PER_DELIVERY = 6   # [origin_x, origin_y, dest_x, dest_y, time_left_norm, priority_norm]


# --- Action types ------------------------------------------------------------

@dataclass
class AssignRoute(Action):
    delivery_id: str = ""
    path: list[tuple[int, int]] = field(default_factory=list)
    kind: str = "assign_route"


@dataclass
class DeferDelivery(Action):
    delivery_id: str = ""
    until_tick: int = 0
    kind: str = "defer_delivery"


@dataclass
class CancelDelivery(Action):
    delivery_id: str = ""
    reason: str = ""
    kind: str = "cancel_delivery"


@dataclass
class RequestPriority(Action):
    delivery_id: str = ""
    kind: str = "request_priority"


DELIVERY_ACTION_TYPES: tuple[type[Action], ...] = (
    AssignRoute,
    DeferDelivery,
    CancelDelivery,
    RequestPriority,
    NoOp,
)


# --- Agent -------------------------------------------------------------------

class DeliveryAgent(BaseAgent):
    """Sees only roads near its pending deliveries.

    Heuristic: BFS *within the visible road subgraph* from origin to dest,
    avoiding cells flagged blocked (accidents/roadblocks visible in view) plus
    any cells the TrafficAgent has marked in advisories. If no in-view route
    exists, defer.
    """

    def __init__(
        self,
        *,
        max_pending: int = MAX_PENDING,
        visibility_pad: int = 4,
        rng: Random | None = None,
    ) -> None:
        super().__init__(AgentRole.DELIVERY, rng=rng)
        self.max_pending = max_pending
        self.visibility_pad = visibility_pad

    @property
    def observation_space(self) -> Space:
        return DictSpace({
            "weather": Discrete(3),
            "hour_of_day": Discrete(24),
            "n_pending": Discrete(self.max_pending + 1),
            "visible_avg_traffic": Box(0.0, 1.0, (1,)),
            "visible_blocked_count": Discrete(64),
            "deliveries": Box(-1.0, 1.0, (self.max_pending, FEAT_PER_DELIVERY)),
        })

    @property
    def action_space(self) -> Space:
        return Discrete(len(DELIVERY_ACTION_TYPES))

    @property
    def action_types(self) -> tuple[type[Action], ...]:
        return DELIVERY_ACTION_TYPES

    # --- observe -----------------------------------------------------------

    def observe(self, world: WorldState, ctx: AgentContext) -> dict:
        view = build_delivery_view(
            world, ctx.deliveries.values(), pad=self.visibility_pad,
        )

        feats = np.full((self.max_pending, FEAT_PER_DELIVERY), -1.0, dtype=np.float32)
        W, H = world.grid.width, world.grid.height
        wnorm, hnorm = max(1, W - 1), max(1, H - 1)
        for i, d in enumerate(view.deliveries[: self.max_pending]):
            time_left = max(0, d.deadline_tick - world.tick)
            feats[i, 0] = d.origin[0] / wnorm
            feats[i, 1] = d.origin[1] / hnorm
            feats[i, 2] = d.dest[0] / wnorm
            feats[i, 3] = d.dest[1] / hnorm
            feats[i, 4] = min(1.0, time_left / 60.0)
            feats[i, 5] = d.priority / 3.0

        weather_to_idx = {"clear": 0, "rain": 1, "storm": 2}
        avg_t = float(np.mean(list(view.traffic.values()))) if view.traffic else 0.0
        return {
            "weather": weather_to_idx[view.weather.value],
            "hour_of_day": view.hour_of_day,
            "n_pending": min(self.max_pending, len(view.deliveries)),
            "visible_avg_traffic": np.array([avg_t], dtype=np.float32),
            "visible_blocked_count": min(63, len(view.blocked_in_view)),
            "deliveries": feats,
            "_view": view,
        }

    # --- decide ------------------------------------------------------------

    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]:
        view: DeliveryView = obs.get("_view")
        if view is None or not view.deliveries:
            return [NoOp()]

        # Treat advisory + route-blocked cells as additional blocked cells (only those in view).
        extra_blocked: set[tuple[int, int]] = set()
        for msg in ctx.receive(self.role, Advisory, RouteBlocked):
            cells = msg.area if isinstance(msg, Advisory) else msg.coords
            for c in cells:
                cc = tuple(c)
                if cc in view.visible_roads:
                    extra_blocked.add(cc)

        # Memory: pull high-risk zones near each delivery and avoid their high-risk cells.
        # Only the most dangerous zones (risk >= 0.85) are treated as blockers — and the BFS
        # below falls back to a memory-free attempt if memory_blocked leaves no route.
        memory_blocked: set[tuple[int, int]] = set()
        if ctx.memory is not None:
            for d in view.deliveries[: self.max_pending]:
                for endpoint in (d.origin, d.dest):
                    zones = ctx.memory.query(
                        kind=MemoryKind.HIGH_RISK_ZONE,
                        near=endpoint,
                        max_distance=self.visibility_pad,
                        min_confidence=0.30,
                        current_tick=ctx.tick,
                        top_k=5,
                    )
                    for z in zones:
                        if not isinstance(z, HighRiskZone) or z.risk_score < 0.85:
                            continue
                        for c in z.coords:
                            cc = tuple(c)
                            if cc in view.visible_roads and cc != d.origin and cc != d.dest:
                                memory_blocked.add(cc)

        # Under EmergencyPriority broadcast, become more conservative: defer low-priority work.
        emergency_active = bool(ctx.receive(self.role, EmergencyPriority))
        blocked_strict = view.blocked_in_view | extra_blocked
        blocked_with_memory = blocked_strict | memory_blocked

        actions: list[Action] = []
        for d in view.deliveries[: self.max_pending]:
            # During an emergency surge, hold low-priority deliveries off the road.
            if emergency_active and d.priority < 2:
                actions.append(DeferDelivery(delivery_id=d.id, until_tick=ctx.tick + 5))
                continue
            # Try memory-aware routing first; fall back to memory-free if it leaves no path.
            path = self._route(d, view, blocked_with_memory)
            if not path and memory_blocked:
                path = self._route(d, view, blocked_strict)
            if path:
                actions.append(AssignRoute(delivery_id=d.id, path=path))
            else:
                actions.append(DeferDelivery(delivery_id=d.id, until_tick=ctx.tick + 5))

        if not actions:
            actions.append(NoOp())
        return actions

    def _route(
        self,
        d,
        view: DeliveryView,
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        o = self._nearest_visible_road(d.origin, view.visible_roads, blocked)
        t = self._nearest_visible_road(d.dest, view.visible_roads, blocked)
        if o is None or t is None:
            return []
        return self._bfs_in_subgraph(o, t, view.visible_roads, blocked)

    # --- routing helpers ---------------------------------------------------

    @staticmethod
    def _nearest_visible_road(
        p: tuple[int, int],
        visible_roads: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if p in visible_roads and p not in blocked:
            return p
        candidates = [c for c in visible_roads if c not in blocked]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1]))

    @staticmethod
    def _bfs_in_subgraph(
        start: tuple[int, int],
        goal: tuple[int, int],
        nodes: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if start in blocked or goal in blocked or start not in nodes or goal not in nodes:
            return []
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            x, y = cur
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in parents or n not in nodes or n in blocked:
                    continue
                parents[n] = cur
                q.append(n)
        if goal not in parents:
            return []
        path: list[tuple[int, int]] = []
        node: tuple[int, int] | None = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path
