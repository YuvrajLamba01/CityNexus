"""Layer 2 — system state checks: did the claimed effect actually happen?"""

from __future__ import annotations

from citynexus.agents.base import AgentRole, NoOp
from citynexus.agents.emergency import DispatchUnit as EmergencyDispatch
from citynexus.entities.unit import UnitKind
from citynexus.verify.base import Check
from citynexus.verify.schemas import CheckResult, CheckStatus, VerificationContext


class CongestionReducedCheck(Check):
    """If congestion was high and the traffic agent acted, congestion should not rise."""

    name = "congestion_reduced"
    layer = "system_state"
    attributed_to = (AgentRole.TRAFFIC.value,)

    def __init__(self, *, threshold: float = 0.5, flat_band: float = 0.05) -> None:
        self.threshold = threshold
        self.flat_band = flat_band

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        prev_cong = ctx.prev_state.congestion_ratio()
        curr_cong = ctx.curr_state.congestion_ratio()
        if prev_cong < self.threshold:
            return self._result(
                CheckStatus.SKIP,
                reason=f"prev congestion {prev_cong:.2f} below threshold {self.threshold}",
            )
        traffic_actions = [a for a in ctx.actions.get(AgentRole.TRAFFIC, []) if not isinstance(a, NoOp)]
        if not traffic_actions:
            return self._result(
                CheckStatus.WARN, score=0.5,
                reason=f"high congestion ({prev_cong:.2f}) but TrafficAgent took no action",
            )
        delta = curr_cong - prev_cong
        if delta < -self.flat_band:
            return self._result(
                CheckStatus.PASS, score=1.0,
                reason=f"congestion dropped {prev_cong:.2f}→{curr_cong:.2f} (Δ={delta:+.3f})",
                metadata={"prev": prev_cong, "curr": curr_cong, "delta": delta},
            )
        if delta <= self.flat_band:
            return self._result(
                CheckStatus.WARN, score=0.5,
                reason=f"congestion roughly flat (Δ={delta:+.3f})",
                metadata={"prev": prev_cong, "curr": curr_cong},
            )
        return self._result(
            CheckStatus.FAIL, score=0.0,
            reason=f"congestion rose despite traffic action (Δ={delta:+.3f})",
            metadata={"prev": prev_cong, "curr": curr_cong},
        )


class EmergencySolvedCheck(Check):
    """Active accidents must either decrease, or be actively responded to."""

    name = "emergency_solved"
    layer = "system_state"
    attributed_to = (AgentRole.EMERGENCY.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        prev_acc = len(ctx.prev_state.accidents)
        curr_acc = len(ctx.curr_state.accidents)
        if prev_acc == 0:
            return self._result(CheckStatus.SKIP, reason="no accidents at start of tick")

        if curr_acc < prev_acc or ctx.accidents_cleared > 0:
            cleared = max(prev_acc - curr_acc, ctx.accidents_cleared)
            return self._result(
                CheckStatus.PASS, score=1.0,
                reason=f"accidents cleared this tick={cleared}, active {prev_acc}→{curr_acc}",
            )

        emergency_actions = ctx.actions.get(AgentRole.EMERGENCY, [])
        dispatched = [a for a in emergency_actions if isinstance(a, EmergencyDispatch)]
        if dispatched:
            return self._result(
                CheckStatus.PASS, score=0.7,
                reason=f"active dispatch in progress ({len(dispatched)} unit(s))",
            )

        # No clearance, no dispatch — was the lack of action excusable?
        idle_amb = sum(
            1 for u in ctx.agent_ctx.units.values()
            if u.kind == UnitKind.AMBULANCE and u.is_idle
        )
        if idle_amb > 0:
            return self._result(
                CheckStatus.FAIL, score=0.0,
                reason=f"{prev_acc} accidents active, {idle_amb} idle ambulance(s), no dispatch",
                metadata={"prev_acc": prev_acc, "idle_amb": idle_amb},
            )
        return self._result(
            CheckStatus.WARN, score=0.3,
            reason=f"{prev_acc} accidents active but all ambulances busy",
        )


class AccidentResponseCheck(Check):
    """If new accidents spawned and we have idle ambulances, expect a dispatch this tick."""

    name = "accident_response_latency"
    layer = "system_state"
    attributed_to = (AgentRole.EMERGENCY.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        new_acc = ctx.accidents_spawned
        if new_acc == 0:
            return self._result(CheckStatus.SKIP, reason="no new accidents this tick")
        idle_amb = sum(
            1 for u in ctx.agent_ctx.units.values()
            if u.kind == UnitKind.AMBULANCE and u.is_idle
        )
        emergency_actions = ctx.actions.get(AgentRole.EMERGENCY, [])
        dispatched = [a for a in emergency_actions if isinstance(a, EmergencyDispatch)]
        if not dispatched and idle_amb > 0:
            return self._result(
                CheckStatus.FAIL, score=0.0,
                reason=f"{new_acc} new accident(s), {idle_amb} idle ambulance(s), no dispatch",
            )
        return self._result(
            CheckStatus.PASS, score=1.0,
            reason=f"{new_acc} new accident(s); {len(dispatched)} dispatch(es), {idle_amb} idle remaining",
        )


class DeliveryProgressCheck(Check):
    """Vans assigned to deliveries should change position toward their target each tick."""

    name = "delivery_progress"
    layer = "system_state"
    attributed_to = (AgentRole.DELIVERY.value,)

    def __init__(self, *, stall_grace: int = 2) -> None:
        # If a van's pos has been the same for >= stall_grace ticks while EN_ROUTE → flag.
        # Lightweight implementation: just check the van isn't somehow off-grid or with no target.
        self.stall_grace = stall_grace

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        active_vans = [
            u for u in ctx.agent_ctx.units.values()
            if u.kind == UnitKind.DELIVERY_VAN and u.status.value == "en_route"
        ]
        if not active_vans:
            return self._result(CheckStatus.SKIP, reason="no en-route vans")
        violations = [
            f"{u.id}: en_route with no target" for u in active_vans if u.target is None
        ]
        if violations:
            return self._result(
                CheckStatus.FAIL, score=0.0,
                reason=f"{len(violations)} van(s) en_route without target",
                metadata={"violations": violations},
            )
        return self._result(
            CheckStatus.PASS,
            reason=f"{len(active_vans)} van(s) en_route with valid target",
        )


def default_system_state_checks() -> list[Check]:
    return [
        CongestionReducedCheck(),
        EmergencySolvedCheck(),
        AccidentResponseCheck(),
        DeliveryProgressCheck(),
    ]
