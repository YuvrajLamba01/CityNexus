"""MultiAgentCoordinator — orchestrates one full tick.

Per tick:
  1. Stochastic spawn of deliveries / incidents.
  2. Each agent observes + decides.
  3. Actions applied to env (roadblocks, dispatches) and to entities (deliveries, units).
     Each application also emits typed inter-agent messages where relevant.
  4. Bus delivers messages → per-role inboxes (one-tick lifetime).
  5. Mobile units stepped one cell toward target; arrivals resolve.
  6. Env stepped (weather/traffic/accidents advance).
  7. Per-tick bookkeeping (delivery deadlines, incident TTLs).
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

from typing import TYPE_CHECKING

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent

if TYPE_CHECKING:
    from citynexus.memory.store import MemoryStore
from citynexus.agents.delivery import (
    AssignRoute, CancelDelivery, DeferDelivery, RequestPriority,
)
from citynexus.agents.emergency import (
    DispatchUnit as EmergencyDispatch,
    RecallUnit as EmergencyRecall,
    RequestClearance,
)
from citynexus.agents.messages import (
    Advisory,
    Channel,
    ClearanceRequest,
    Directive,
    DispatchNotice,
    EmergencyPriority,
    IncidentReport,
    RouteBlocked,
    SignalChange,
)
from citynexus.agents.planner import BroadcastDirective, SetPriority
from citynexus.agents.police import (
    DispatchUnit as PoliceDispatch,
    EstablishCordon,
    RecallUnit as PoliceRecall,
)
from citynexus.agents.traffic import ClearRoadblock, IssueAdvisory, PlaceRoadblock
from citynexus.city.zones import Zone
from citynexus.entities.delivery import Delivery, DeliveryStatus
from citynexus.entities.incident import Incident, IncidentKind
from citynexus.entities.unit import ResponderUnit, UnitKind, UnitStatus
from citynexus.env.core import CityNexusEnv, StepInfo


@dataclass
class CoordinatorConfig:
    delivery_spawn_rate: float = 0.20      # per tick
    delivery_window: int = 30              # ticks of deadline
    incident_spawn_rate: float = 0.06
    n_ambulances: int = 3
    n_police: int = 3
    n_delivery_vans: int = 4
    seed: int = 0


@dataclass
class TickResult:
    step_info: StepInfo
    actions: dict[AgentRole, list[Action]]
    new_deliveries: list[str]
    new_incidents: list[str]
    completed_deliveries: list[str]
    messages_delivered: int
    messages_by_kind: dict[str, int]


class MultiAgentCoordinator:
    """Drives the multi-agent loop. Holds env + agents + entity registries + bus."""

    def __init__(
        self,
        env: CityNexusEnv,
        agents: Iterable[BaseAgent],
        config: CoordinatorConfig | None = None,
        *,
        memory: "MemoryStore | None" = None,
    ) -> None:
        self.env = env
        self.agents: dict[AgentRole, BaseAgent] = {a.role: a for a in agents}
        self.config = config or CoordinatorConfig()
        self.memory = memory
        self._rng = Random(self.config.seed)
        self.ctx: AgentContext | None = None
        self._next_id = 0

    # ----- lifecycle -------------------------------------------------------

    def reset(self) -> AgentContext:
        self.env.reset()
        self.ctx = AgentContext(tick=0, world=self.env.state, memory=self.memory)
        for role in AgentRole:
            self.ctx.priorities[role] = 1.0
        self._next_id = 0
        self._spawn_initial_units()
        for agent in self.agents.values():
            agent.reset()
        return self.ctx

    def step(self) -> TickResult:
        if self.ctx is None:
            raise RuntimeError("Call reset() first.")

        self.ctx.world = self.env.state
        sent_baseline = dict(self.ctx.bus.stats.get("by_kind", {}))

        # 1. Spawn entities the agents can act on.
        new_deliveries = self._maybe_spawn_deliveries()
        new_incidents = self._maybe_spawn_incidents()
        for iid in new_incidents:
            inc = self.ctx.incidents[iid]
            self.ctx.send(IncidentReport(
                sender=AgentRole.PLANNER, channel=Channel.OPS,
                tick=self.ctx.tick, recipients=(AgentRole.POLICE,),
                location=inc.pos, incident_kind=inc.kind.value, severity=inc.severity,
            ))

        # 2. Agents observe + decide.
        actions: dict[AgentRole, list[Action]] = {}
        for role, agent in self.agents.items():
            actions[role] = agent.act(self.env.state, self.ctx)

        # 3. Apply each role's actions to env / entities (also emits messages).
        for role, role_actions in actions.items():
            self._apply_actions(role, role_actions)

        # 4. Route messages → inboxes (capture per-tick send delta first).
        sent_after = self.ctx.bus.stats.get("by_kind", {})
        by_kind = {
            k: sent_after.get(k, 0) - sent_baseline.get(k, 0)
            for k in sent_after
            if sent_after.get(k, 0) > sent_baseline.get(k, 0)
        }
        delivered = self.ctx.bus.deliver(AgentRole)

        # 5. Move units one cell toward target; resolve arrivals.
        self._update_units()

        # 6. Bookkeeping (deadlines, expiries).
        self._update_deliveries()

        # 7. Step the world.
        info = self.env.step()
        completed = self._finalize_tick()

        # 8. Sync tick.
        self.ctx.tick = info.tick
        self.ctx.world = self.env.state
        return TickResult(
            step_info=info,
            actions=actions,
            new_deliveries=new_deliveries,
            new_incidents=new_incidents,
            completed_deliveries=completed,
            messages_delivered=delivered,
            messages_by_kind=by_kind,
        )

    # ----- spawning --------------------------------------------------------

    def _gen_id(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id:05d}"

    def _maybe_spawn_deliveries(self) -> list[str]:
        spawned: list[str] = []
        if self._rng.random() >= self.config.delivery_spawn_rate:
            return spawned
        grid = self.env.state.grid
        origins = list(grid.cells_of(Zone.COMMERCIAL)) + list(grid.cells_of(Zone.INDUSTRIAL))
        dests = list(grid.cells_of(Zone.RESIDENTIAL)) + list(grid.cells_of(Zone.COMMERCIAL))
        if not origins or not dests:
            return spawned
        o = self._rng.choice(origins)
        d = self._rng.choice(dests)
        if (o.x, o.y) == (d.x, d.y):
            return spawned
        did = self._gen_id("delivery")
        self.ctx.deliveries[did] = Delivery(
            id=did,
            origin=(o.x, o.y),
            dest=(d.x, d.y),
            spawned_tick=self.ctx.tick,
            deadline_tick=self.ctx.tick + self.config.delivery_window,
            priority=self._rng.choice([1, 1, 1, 2, 3]),
        )
        spawned.append(did)
        return spawned

    def _maybe_spawn_incidents(self) -> list[str]:
        spawned: list[str] = []
        if self._rng.random() >= self.config.incident_spawn_rate:
            return spawned
        grid = self.env.state.grid
        pool = list(grid.cells_of(Zone.COMMERCIAL)) + list(grid.cells_of(Zone.RESIDENTIAL))
        if not pool:
            return spawned
        cell = self._rng.choice(pool)
        kind = self._rng.choice(list(IncidentKind))
        iid = self._gen_id("incident")
        self.ctx.incidents[iid] = Incident(
            id=iid,
            kind=kind,
            pos=(cell.x, cell.y),
            severity=self._rng.randint(1, 3),
            spawned_tick=self.ctx.tick,
            ttl=self._rng.randint(8, 20),
        )
        spawned.append(iid)
        return spawned

    def _spawn_initial_units(self) -> None:
        grid = self.env.state.grid
        hospitals = list(grid.cells_of(Zone.HOSPITAL))
        commercials = list(grid.cells_of(Zone.COMMERCIAL))
        industrials = list(grid.cells_of(Zone.INDUSTRIAL))

        ambulance_homes = hospitals if hospitals else (commercials[:1] or industrials[:1])
        for i in range(self.config.n_ambulances):
            home_cell = ambulance_homes[i % len(ambulance_homes)]
            home = self._snap_to_road((home_cell.x, home_cell.y))
            uid = self._gen_id("amb")
            self.ctx.units[uid] = ResponderUnit(
                id=uid, kind=UnitKind.AMBULANCE,
                home=home, pos=home,
            )

        if commercials:
            for i in range(self.config.n_police):
                home_cell = commercials[i % len(commercials)]
                home = self._snap_to_road((home_cell.x, home_cell.y))
                uid = self._gen_id("pol")
                self.ctx.units[uid] = ResponderUnit(
                    id=uid, kind=UnitKind.POLICE_CAR,
                    home=home, pos=home,
                )

        if industrials:
            for i in range(self.config.n_delivery_vans):
                home_cell = industrials[i % len(industrials)]
                home = self._snap_to_road((home_cell.x, home_cell.y))
                uid = self._gen_id("van")
                self.ctx.units[uid] = ResponderUnit(
                    id=uid, kind=UnitKind.DELIVERY_VAN,
                    home=home, pos=home,
                )

    def _snap_to_road(self, point: tuple[int, int]) -> tuple[int, int]:
        """Return the nearest road cell to `point` (or `point` itself if already a road)."""
        from collections import deque
        grid = self.env.state.grid
        if grid.is_road(point):
            return point
        visited = {point}
        queue: deque[tuple[int, int]] = deque([point])
        while queue:
            cur = queue.popleft()
            for n in grid.neighbors4(cur):
                if n in visited:
                    continue
                visited.add(n)
                if grid.is_road(n):
                    return n
                queue.append(n)
            if len(visited) > 400:
                break
        return point

    # ----- action application ---------------------------------------------

    def _apply_actions(self, role: AgentRole, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, AssignRoute):
                self._apply_assign_route(action)
            elif isinstance(action, DeferDelivery):
                d = self.ctx.deliveries.get(action.delivery_id)
                if d:
                    d.status = DeliveryStatus.DELAYED
            elif isinstance(action, CancelDelivery):
                d = self.ctx.deliveries.get(action.delivery_id)
                if d:
                    d.status = DeliveryStatus.FAILED
            elif isinstance(action, RequestPriority):
                d = self.ctx.deliveries.get(action.delivery_id)
                if d:
                    d.priority = min(3, d.priority + 1)

            elif isinstance(action, PlaceRoadblock):
                placed = self.env.add_roadblock(
                    action.x, action.y,
                    ttl=action.ttl, reason=f"{role.value}:{action.reason}",
                )
                if placed:
                    self.ctx.send(SignalChange(
                        sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                        intersection=(action.x, action.y),
                        state="blocked", ttl=action.ttl,
                    ))
            elif isinstance(action, ClearRoadblock):
                cleared = self.env.clear_roadblock(action.x, action.y)
                if cleared:
                    self.ctx.send(SignalChange(
                        sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                        intersection=(action.x, action.y),
                        state="open", ttl=0,
                    ))
            elif isinstance(action, IssueAdvisory):
                self.ctx.send(Advisory(
                    sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                    recipients=(AgentRole.DELIVERY,),
                    area=list(action.area), severity=action.severity,
                ))

            elif isinstance(action, (EmergencyDispatch, PoliceDispatch)):
                self._apply_dispatch(action, role)
            elif isinstance(action, (EmergencyRecall, PoliceRecall)):
                u = self.ctx.units.get(action.unit_id)
                if u and not u.is_idle:
                    u.target = u.home
                    u.status = UnitStatus.RETURNING
                    u.assigned_to = None
            elif isinstance(action, RequestClearance):
                self.ctx.send(ClearanceRequest(
                    sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                    recipients=(AgentRole.TRAFFIC,),
                    location=(action.x, action.y),
                    requesting_for=f"accident@{action.x},{action.y}",
                ))
            elif isinstance(action, EstablishCordon):
                cordon_cells: list[tuple[int, int]] = []
                grid = self.env.state.grid
                for dy in range(-action.radius, action.radius + 1):
                    for dx in range(-action.radius, action.radius + 1):
                        if abs(dx) + abs(dy) <= action.radius:
                            cx, cy = action.x + dx, action.y + dy
                            if grid.is_road((cx, cy)):
                                if self.env.add_roadblock(cx, cy, ttl=action.ttl, reason="police:cordon"):
                                    cordon_cells.append((cx, cy))
                if cordon_cells:
                    self.ctx.send(RouteBlocked(
                        sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                        recipients=(AgentRole.DELIVERY, AgentRole.EMERGENCY),
                        coords=cordon_cells, reason="police_cordon",
                        severity=2, expires_at_tick=self.ctx.tick + action.ttl,
                    ))

            elif isinstance(action, SetPriority):
                self.ctx.priorities[action.role] = action.value
            elif isinstance(action, BroadcastDirective):
                protocol = action.payload.get("protocol", "")
                if protocol in ("surge", "emergency"):
                    self.ctx.send(EmergencyPriority(
                        sender=role, channel=Channel.PLANNING, tick=self.ctx.tick,
                        severity=int(action.payload.get("severity", 2)),
                        protocol=protocol,
                        body=action.body,
                    ))
                else:
                    self.ctx.send(Directive(
                        sender=role, channel=Channel.PLANNING, tick=self.ctx.tick,
                        body=action.body, protocol=protocol, payload=dict(action.payload),
                    ))
            # NoOp is silently skipped.

    def _apply_assign_route(self, action: AssignRoute) -> None:
        d = self.ctx.deliveries.get(action.delivery_id)
        if not d:
            return
        d.assigned_route = list(action.path)
        d.route_progress = 0
        d.status = DeliveryStatus.EN_ROUTE
        d.last_update_tick = self.ctx.tick
        for u in self.ctx.units.values():
            if u.kind == UnitKind.DELIVERY_VAN and u.is_idle:
                u.assigned_to = d.id
                # Snap building destination to nearest road cell — van delivers
                # from the curb, never enters the building.
                u.target = self._snap_to_road(d.dest)
                u.status = UnitStatus.EN_ROUTE
                break

    def _apply_dispatch(self, action, role: AgentRole) -> None:
        u = self.ctx.units.get(action.unit_id)
        if not u or not u.is_idle:
            return
        # Police incidents live at building cells; ambulance accidents already
        # live at road cells, so snap is a no-op for them. Either way, the unit
        # only ever moves on roads.
        u.target = self._snap_to_road((action.x, action.y))
        u.status = UnitStatus.EN_ROUTE
        u.assigned_to = action.assigned_to

        # Notify others that a unit is heading to this target (light traffic notice).
        self.ctx.send(DispatchNotice(
            sender=role, channel=Channel.OPS, tick=self.ctx.tick,
            unit_id=u.id, target=u.target,
        ))
        # Emergency dispatch → mark the destination cell as a route hazard for delivery.
        if role == AgentRole.EMERGENCY:
            self.ctx.send(RouteBlocked(
                sender=role, channel=Channel.OPS, tick=self.ctx.tick,
                recipients=(AgentRole.DELIVERY,),
                coords=[u.target], reason="emergency_scene", severity=2,
            ))

    # ----- per-tick advancement -------------------------------------------

    def _update_units(self) -> None:
        for u in self.ctx.units.values():
            if u.status not in (UnitStatus.EN_ROUTE, UnitStatus.RETURNING):
                continue
            if u.target is None:
                continue
            if u.pos == u.target:
                if u.status == UnitStatus.EN_ROUTE:
                    u.status = UnitStatus.ON_SCENE
                    self._on_unit_arrived(u)
                else:
                    u.status = UnitStatus.IDLE
                    u.target = None
                    u.assigned_to = None
                continue
            # Pathfind THROUGH ROADS — never teleport through buildings.
            # Start (home) and goal (incident / building delivery dest) may be
            # non-road cells; intermediate steps must be roads.
            path = self._find_unit_path(u.pos, u.target)
            if path and len(path) > 1:
                u.pos = path[1]
            # No road path → unit waits this tick.

    def _find_unit_path(
        self, start: tuple[int, int], goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """BFS over road cells; start and goal may be any cell type."""
        from collections import deque
        grid = self.env.state.grid
        if start == goal:
            return [start]
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cur = queue.popleft()
            for n in grid.neighbors4(cur):
                if n in parents:
                    continue
                is_goal = (n == goal)
                is_road = grid.is_road(n)
                if not is_goal and not is_road:
                    continue
                parents[n] = cur
                if is_goal:
                    path: list[tuple[int, int]] = [n]
                    node = parents[n]
                    while node is not None:
                        path.append(node)
                        node = parents[node]
                    path.reverse()
                    return path
                queue.append(n)
        return None

    def _on_unit_arrived(self, u: ResponderUnit) -> None:
        if u.kind == UnitKind.AMBULANCE:
            self.env.clear_accident(*u.pos)
            u.target = u.home
            u.status = UnitStatus.RETURNING
        elif u.kind == UnitKind.POLICE_CAR and u.assigned_to:
            inc = self.ctx.incidents.get(u.assigned_to)
            if inc:
                inc.resolved = True
            u.target = u.home
            u.status = UnitStatus.RETURNING
        elif u.kind == UnitKind.DELIVERY_VAN and u.assigned_to:
            d = self.ctx.deliveries.get(u.assigned_to)
            if d:
                d.status = DeliveryStatus.DELIVERED
                d.last_update_tick = self.ctx.tick
            u.target = u.home
            u.status = UnitStatus.RETURNING

    def _update_deliveries(self) -> None:
        tick = self.ctx.tick
        for d in self.ctx.deliveries.values():
            if d.is_open and tick > d.deadline_tick:
                d.status = DeliveryStatus.FAILED

    def _finalize_tick(self) -> list[str]:
        to_drop = [
            iid for iid, inc in self.ctx.incidents.items()
            if inc.resolved or self.ctx.tick > inc.spawned_tick + inc.ttl
        ]
        for iid in to_drop:
            del self.ctx.incidents[iid]
        return [
            did for did, d in self.ctx.deliveries.items()
            if d.status == DeliveryStatus.DELIVERED and d.last_update_tick == self.ctx.tick
        ]
