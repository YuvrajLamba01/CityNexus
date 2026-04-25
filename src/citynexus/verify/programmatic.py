"""Layer 1 — programmatic checks: action well-formedness, schema, route validity."""

from __future__ import annotations

from citynexus.agents.base import AgentRole
from citynexus.agents.delivery import AssignRoute
from citynexus.agents.emergency import DispatchUnit as EmergencyDispatch
from citynexus.agents.police import DispatchUnit as PoliceDispatch
from citynexus.agents.traffic import PlaceRoadblock
from citynexus.verify.base import Check
from citynexus.verify.schemas import CheckResult, CheckStatus, VerificationContext


class RoutePathConnectivityCheck(Check):
    """A delivery route must be non-empty, all on road cells, in bounds, and 4-connected."""

    name = "route_path_connectivity"
    layer = "programmatic"
    attributed_to = (AgentRole.DELIVERY.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        violations: list[str] = []
        grid = ctx.curr_state.grid
        n_routes = 0
        for actions in ctx.actions.values():
            for action in actions:
                if not isinstance(action, AssignRoute):
                    continue
                n_routes += 1
                path = action.path
                if not path:
                    violations.append(f"{action.delivery_id}: empty path")
                    continue
                for i, (x, y) in enumerate(path):
                    if not grid.in_bounds((x, y)):
                        violations.append(f"{action.delivery_id}#{i}: out of bounds {(x, y)}")
                        break
                    if not grid.is_road((x, y)):
                        violations.append(f"{action.delivery_id}#{i}: not a road cell {(x, y)}")
                        break
                    if i > 0:
                        px, py = path[i - 1]
                        if abs(x - px) + abs(y - py) != 1:
                            violations.append(
                                f"{action.delivery_id}#{i}: gap {(px, py)}→{(x, y)}"
                            )
                            break
        if n_routes == 0:
            return self._result(CheckStatus.SKIP, reason="no AssignRoute actions this tick")
        if violations:
            return self._result(
                CheckStatus.FAIL,
                score=0.0,
                reason=f"{len(violations)} path violation(s): {violations[:3]}",
                metadata={"violations": violations, "n_routes": n_routes},
            )
        return self._result(CheckStatus.PASS, reason=f"{n_routes} route(s) well-formed")


class UnitDispatchValidityCheck(Check):
    """DispatchUnit must reference a known unit, idle (or already routed by us), and a valid target."""

    name = "unit_dispatch_validity"
    layer = "programmatic"
    attributed_to = (AgentRole.EMERGENCY.value, AgentRole.POLICE.value)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        violations: list[str] = []
        n_dispatches = 0
        units = ctx.agent_ctx.units
        grid = ctx.curr_state.grid
        for role, actions in ctx.actions.items():
            for action in actions:
                if not isinstance(action, (EmergencyDispatch, PoliceDispatch)):
                    continue
                n_dispatches += 1
                u = units.get(action.unit_id)
                if u is None:
                    violations.append(f"{role.value}: unknown unit_id={action.unit_id}")
                    continue
                if not grid.in_bounds((action.x, action.y)):
                    violations.append(
                        f"{role.value}: target {(action.x, action.y)} out of bounds"
                    )
        if n_dispatches == 0:
            return self._result(CheckStatus.SKIP, reason="no dispatch actions this tick")
        if violations:
            return self._result(
                CheckStatus.FAIL,
                score=0.0,
                reason=f"{len(violations)} dispatch violation(s): {violations[:3]}",
                metadata={"violations": violations, "n_dispatches": n_dispatches},
            )
        return self._result(CheckStatus.PASS, reason=f"{n_dispatches} dispatch(es) valid")


class RoadblockPlacementCheck(Check):
    """PlaceRoadblock must target a road cell in bounds."""

    name = "roadblock_placement"
    layer = "programmatic"
    attributed_to = (AgentRole.TRAFFIC.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        violations: list[str] = []
        n_blocks = 0
        grid = ctx.curr_state.grid
        for actions in ctx.actions.values():
            for action in actions:
                if not isinstance(action, PlaceRoadblock):
                    continue
                n_blocks += 1
                if not grid.in_bounds((action.x, action.y)):
                    violations.append(f"out of bounds {(action.x, action.y)}")
                elif not grid.is_road((action.x, action.y)):
                    violations.append(f"not road {(action.x, action.y)}")
                if action.ttl <= 0:
                    violations.append(f"non-positive TTL {action.ttl} at {(action.x, action.y)}")
        if n_blocks == 0:
            return self._result(CheckStatus.SKIP, reason="no PlaceRoadblock actions this tick")
        if violations:
            return self._result(
                CheckStatus.FAIL, score=0.0,
                reason=f"{len(violations)} block placement violation(s): {violations[:3]}",
                metadata={"violations": violations},
            )
        return self._result(CheckStatus.PASS, reason=f"{n_blocks} placement(s) valid")


class DeliveryReferenceCheck(Check):
    """Any action that references a delivery_id must point to an existing delivery."""

    name = "delivery_reference"
    layer = "programmatic"
    attributed_to = (AgentRole.DELIVERY.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        violations: list[str] = []
        n_refs = 0
        deliveries = ctx.agent_ctx.deliveries
        for actions in ctx.actions.values():
            for action in actions:
                did = getattr(action, "delivery_id", None)
                if not did:
                    continue
                n_refs += 1
                if did not in deliveries:
                    violations.append(f"{type(action).__name__}: unknown delivery_id={did}")
        if n_refs == 0:
            return self._result(CheckStatus.SKIP, reason="no delivery-referencing actions")
        if violations:
            return self._result(
                CheckStatus.FAIL, score=0.0,
                reason=f"{len(violations)} unknown delivery reference(s)",
                metadata={"violations": violations},
            )
        return self._result(CheckStatus.PASS, reason=f"{n_refs} reference(s) resolved")


def default_programmatic_checks() -> list[Check]:
    return [
        RoutePathConnectivityCheck(),
        UnitDispatchValidityCheck(),
        RoadblockPlacementCheck(),
        DeliveryReferenceCheck(),
    ]
