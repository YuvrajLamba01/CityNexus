"""Reward / penalty / process-aware components.

Each function takes a `VerificationContext` and returns a `dict[role_value → float]`
mapping per-agent contributions. Functions are pure (the system orchestrator owns
all per-tick state).

Three kinds:
  * outcome rewards        — for terminal events (delivery, accident clear, ...)
  * penalties              — for delays, collisions, inefficiency
  * process-aware rewards  — dense per-tick signals (progress, intent, anticipation)

Attribution rules:
  * delivery completion / failure → DeliveryAgent
  * accident clearance / response → EmergencyAgent
  * incident resolution           → PoliceAgent
  * congestion management         → TrafficAgent
  * priority anticipation         → PlannerAgent
  * planner additionally gets a small share of system-wide gains (handled in system.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from citynexus.agents.base import AgentRole
from citynexus.agents.emergency import DispatchUnit as EmergencyDispatch
from citynexus.agents.police import DispatchUnit as PoliceDispatch
from citynexus.entities.unit import UnitKind
from citynexus.rewards.schemas import CityScore

if TYPE_CHECKING:
    from citynexus.verify.schemas import VerificationContext


# ============================================================================
# Outcome rewards
# ============================================================================

def delivery_completion_reward(ctx: "VerificationContext", weight: float = 1.0) -> dict[str, float]:
    """Per delivery completed this tick → DeliveryAgent."""
    n = len(ctx.completed_deliveries)
    if n == 0:
        return {}
    return {AgentRole.DELIVERY.value: weight * n}


def accident_clearance_reward(ctx: "VerificationContext", weight: float = 0.5) -> dict[str, float]:
    """Each accident cleared this tick (any cause) → EmergencyAgent."""
    if ctx.accidents_cleared == 0:
        return {}
    return {AgentRole.EMERGENCY.value: weight * ctx.accidents_cleared}


def incident_resolution_reward(
    ctx: "VerificationContext",
    *,
    resolved_count: int,
    weight: float = 0.3,
) -> dict[str, float]:
    """Each incident resolved this tick → PoliceAgent.

    `resolved_count` must be supplied by the caller (the system orchestrator
    diffs incident snapshots across ticks)."""
    if resolved_count == 0:
        return {}
    return {AgentRole.POLICE.value: weight * resolved_count}


def congestion_management_reward(ctx: "VerificationContext", weight: float = 0.30) -> dict[str, float]:
    """Drop in congestion (prev → curr) → TrafficAgent. Scaled by magnitude of drop."""
    drop = ctx.prev_state.congestion_ratio() - ctx.curr_state.congestion_ratio()
    if drop <= 0:
        return {}
    return {AgentRole.TRAFFIC.value: weight * drop}


# ============================================================================
# Penalties — delays, collisions, inefficiency
# ============================================================================

def delivery_failure_penalty(
    ctx: "VerificationContext",
    *,
    new_failed: int,
    weight: float = -0.5,
) -> dict[str, float]:
    """Newly-failed deliveries this tick → DeliveryAgent (penalty).

    `new_failed` is supplied by the system orchestrator (diff across ticks)."""
    if new_failed == 0:
        return {}
    return {AgentRole.DELIVERY.value: weight * new_failed}


def deadline_pressure_penalty(ctx: "VerificationContext", weight: float = -0.05) -> dict[str, float]:
    """Each tick a delivery sits past its deadline (still open) → DeliveryAgent.
    Continuous pressure for delays, not just terminal failure."""
    over = sum(
        1 for d in ctx.agent_ctx.deliveries.values()
        if d.is_open and ctx.tick > d.deadline_tick
    )
    if over == 0:
        return {}
    return {AgentRole.DELIVERY.value: weight * over}


def congestion_rise_penalty(ctx: "VerificationContext", weight: float = -0.20) -> dict[str, float]:
    """Rise in congestion → TrafficAgent (penalty proportional to rise)."""
    rise = ctx.curr_state.congestion_ratio() - ctx.prev_state.congestion_ratio()
    if rise <= 0:
        return {}
    return {AgentRole.TRAFFIC.value: weight * rise}


def collision_penalty(ctx: "VerificationContext", weight: float = -0.30) -> dict[str, float]:
    """New accidents that spawned at high-traffic cells → TrafficAgent.

    Rationale: accidents at quiet cells aren't reasonably preventable; accidents
    where density was already high (>0.5) implicate failure to manage flow.
    """
    if ctx.accidents_spawned == 0:
        return {}
    new_accidents = [a for a in ctx.curr_state.accidents if a.spawned_tick == ctx.tick]
    if not new_accidents:
        return {}
    pen = 0.0
    H, W = ctx.prev_state.traffic.shape
    for a in new_accidents:
        if not (0 <= a.y < H and 0 <= a.x < W):
            continue
        density = float(ctx.prev_state.traffic[a.y, a.x])
        if density > 0.5:
            pen += weight * ((density - 0.5) / 0.5)
    if pen == 0.0:
        return {}
    return {AgentRole.TRAFFIC.value: pen}


def idle_unit_inefficiency(
    ctx: "VerificationContext", weight: float = -0.05,
) -> dict[str, float]:
    """Per idle unit when work is available for that role → role penalty."""
    n_acc = len(ctx.curr_state.accidents)
    n_inc = len(ctx.agent_ctx.incidents)
    n_open_unassigned = sum(
        1 for d in ctx.agent_ctx.deliveries.values() if d.is_open and not d.is_assigned
    )

    idle_amb = sum(
        1 for u in ctx.agent_ctx.units.values()
        if u.kind == UnitKind.AMBULANCE and u.is_idle
    )
    idle_pol = sum(
        1 for u in ctx.agent_ctx.units.values()
        if u.kind == UnitKind.POLICE_CAR and u.is_idle
    )
    idle_van = sum(
        1 for u in ctx.agent_ctx.units.values()
        if u.kind == UnitKind.DELIVERY_VAN and u.is_idle
    )

    out: dict[str, float] = {}
    if n_acc > 0 and idle_amb > 0:
        out[AgentRole.EMERGENCY.value] = weight * idle_amb
    if n_inc > 0 and idle_pol > 0:
        out[AgentRole.POLICE.value] = weight * idle_pol
    if n_open_unassigned > 0 and idle_van > 0:
        out[AgentRole.DELIVERY.value] = weight * idle_van
    return out


def redundant_dispatch_penalty(
    ctx: "VerificationContext", weight: float = -0.10,
) -> dict[str, float]:
    """If multiple dispatches target the same cell this tick → small penalty.

    Wasteful coordination; the heuristic shouldn't do it but a learning policy might.
    """
    out: dict[str, float] = {}
    e_targets: dict[tuple[int, int], int] = {}
    p_targets: dict[tuple[int, int], int] = {}
    for action in ctx.actions.get(AgentRole.EMERGENCY, []):
        if isinstance(action, EmergencyDispatch):
            e_targets[(action.x, action.y)] = e_targets.get((action.x, action.y), 0) + 1
    for action in ctx.actions.get(AgentRole.POLICE, []):
        if isinstance(action, PoliceDispatch):
            p_targets[(action.x, action.y)] = p_targets.get((action.x, action.y), 0) + 1

    e_dups = sum(c - 1 for c in e_targets.values() if c > 1)
    p_dups = sum(c - 1 for c in p_targets.values() if c > 1)
    if e_dups:
        out[AgentRole.EMERGENCY.value] = weight * e_dups
    if p_dups:
        out[AgentRole.POLICE.value] = weight * p_dups
    return out


# ============================================================================
# Process-aware rewards (dense per-tick signals)
# ============================================================================

def progress_toward_target_reward(
    ctx: "VerificationContext",
    *,
    prev_positions: dict[str, tuple[int, int]],
    weight: float = 0.05,
) -> dict[str, float]:
    """Per cell of Manhattan progress toward target this tick → owning role.

    A van that moves 1 cell closer to dest gets +weight for the DeliveryAgent.
    Same for ambulances (Emergency) and police cars (Police).
    """
    role_for_kind = {
        UnitKind.AMBULANCE: AgentRole.EMERGENCY.value,
        UnitKind.POLICE_CAR: AgentRole.POLICE.value,
        UnitKind.DELIVERY_VAN: AgentRole.DELIVERY.value,
    }
    out: dict[str, float] = {}
    for u in ctx.agent_ctx.units.values():
        if u.target is None:
            continue
        prev = prev_positions.get(u.id)
        if prev is None:
            continue
        prev_dist = abs(prev[0] - u.target[0]) + abs(prev[1] - u.target[1])
        curr_dist = abs(u.pos[0] - u.target[0]) + abs(u.pos[1] - u.target[1])
        gain = prev_dist - curr_dist
        if gain <= 0:
            continue
        role = role_for_kind.get(u.kind)
        if role is None:
            continue
        out[role] = out.get(role, 0.0) + weight * gain
    return out


def dispatch_intent_reward(
    ctx: "VerificationContext", weight: float = 0.10,
) -> dict[str, float]:
    """Reward an agent for dispatching to a high-severity target. Scales with severity / 3.

    This is a *process* reward — it credits the *decision* (pick the right target)
    independent of whether the eventual arrival succeeds.
    """
    out: dict[str, float] = {}

    for action in ctx.actions.get(AgentRole.EMERGENCY, []):
        if not isinstance(action, EmergencyDispatch):
            continue
        # Match the dispatched cell to a current accident's severity.
        sev = 0
        for a in ctx.curr_state.accidents:
            if a.coord == (action.x, action.y):
                sev = int(a.severity)
                break
        if sev > 0:
            out[AgentRole.EMERGENCY.value] = out.get(AgentRole.EMERGENCY.value, 0.0) + weight * (sev / 3.0)

    for action in ctx.actions.get(AgentRole.POLICE, []):
        if not isinstance(action, PoliceDispatch):
            continue
        inc = ctx.agent_ctx.incidents.get(action.assigned_to)
        if inc is None:
            continue
        out[AgentRole.POLICE.value] = out.get(AgentRole.POLICE.value, 0.0) + weight * (inc.severity / 3.0)

    return out


def planner_anticipation_reward(
    ctx: "VerificationContext",
    *,
    prev_priorities: dict[AgentRole, float],
    weight: float = 0.10,
) -> dict[str, float]:
    """Reward planner when its current priorities align with observed load.

    Two effects:
      (a) holding an elevated priority that matches the load (alignment),
      (b) a fresh raise that just happened (reactivity bonus).
    """
    cur = ctx.agent_ctx.priorities
    n_acc = len(ctx.curr_state.accidents)
    n_inc = len(ctx.agent_ctx.incidents)
    cong = ctx.curr_state.congestion_ratio()

    score = 0.0

    def _aligned(role: AgentRole, condition: bool) -> None:
        nonlocal score
        if condition and cur.get(role, 1.0) > 1.0:
            score += weight
            # Bonus if the raise happened *this tick*.
            if prev_priorities.get(role, 1.0) <= 1.0 and cur.get(role, 1.0) > 1.0:
                score += weight * 0.5

    _aligned(AgentRole.EMERGENCY, n_acc >= 2)
    _aligned(AgentRole.POLICE,    n_inc >= 1)
    _aligned(AgentRole.TRAFFIC,   cong > 0.4)
    _aligned(AgentRole.DELIVERY,
             sum(1 for d in ctx.agent_ctx.deliveries.values() if d.is_open) >= 4)

    if score == 0.0:
        return {}
    return {AgentRole.PLANNER.value: score}


# ============================================================================
# Global city score
# ============================================================================

def compute_city_score(
    ctx: "VerificationContext",
    *,
    weights: dict[str, float] | None = None,
) -> CityScore:
    """Composite per-tick city health metric in [0, 1].

    Sub-scores:
      delivery_health  — completed / (completed + failed) over the running tally
      safety           — 1 − min(1, n_active_accidents / 10)
      mobility         — 1 − congestion_ratio
      coordination     — fraction of planner-priority elevations that match observed load

    Default weights: delivery=0.30, safety=0.30, mobility=0.25, coordination=0.15.
    """
    w = dict(weights or {
        "delivery_health": 0.30,
        "safety":          0.30,
        "mobility":        0.25,
        "coordination":    0.15,
    })
    # Normalise weights defensively.
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    # Delivery health.
    deliveries = ctx.agent_ctx.deliveries.values()
    completed = sum(1 for d in deliveries if d.status.value == "delivered")
    failed = sum(1 for d in deliveries if d.status.value == "failed")
    delivery_h = completed / (completed + failed) if (completed + failed) > 0 else 1.0

    # Safety.
    safety = max(0.0, 1.0 - min(1.0, len(ctx.curr_state.accidents) / 10.0))

    # Mobility.
    mobility = max(0.0, 1.0 - ctx.curr_state.congestion_ratio())

    # Coordination.
    cur_p = ctx.agent_ctx.priorities
    n_acc = len(ctx.curr_state.accidents)
    n_inc = len(ctx.agent_ctx.incidents)
    cong = ctx.curr_state.congestion_ratio()

    expected = aligned = 0
    for role, condition in (
        (AgentRole.EMERGENCY, n_acc >= 2),
        (AgentRole.POLICE,    n_inc >= 1),
        (AgentRole.TRAFFIC,   cong > 0.4),
    ):
        if condition:
            expected += 1
            if cur_p.get(role, 1.0) > 1.0:
                aligned += 1
    coord = aligned / expected if expected > 0 else 1.0

    total = (
        w["delivery_health"] * delivery_h
        + w["safety"]        * safety
        + w["mobility"]      * mobility
        + w["coordination"]  * coord
    )
    return CityScore(
        total=total,
        delivery_health=delivery_h,
        safety=safety,
        mobility=mobility,
        coordination=coord,
        metadata={"weights": w},
    )
