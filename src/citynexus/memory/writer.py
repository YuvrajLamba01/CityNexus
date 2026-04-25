"""MemoryWriter — observes per-tick events and writes typed memory records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from citynexus.memory.schemas import (
    HighRiskZone,
    MemoryKind,
    PastFailure,
    SuccessfulStrategy,
)
from citynexus.memory.store import MemoryStore

if TYPE_CHECKING:
    from citynexus.rewards.schemas import RewardBreakdown
    from citynexus.verify.schemas import VerificationContext, VerificationReport


_FAILURE_TO_AVOIDANCE: dict[str, list[str]] = {
    "route_path_connectivity": ["assign_route"],
    "delivery_reference":      ["assign_route"],
    "unit_dispatch_validity":  ["dispatch_ambulance", "dispatch_police"],
    "roadblock_placement":     ["place_roadblock"],
    "congestion_reduced":      ["place_roadblock"],
    "emergency_solved":        [],
    "accident_response_latency": [],
    "delivery_progress":       ["assign_route"],
    "priority_coherence":      [],
    "dispatch_severity_order": [],
}


@dataclass
class WriterConfig:
    # High-risk zone tuning
    accident_zone_weight: float = 0.10           # base risk added per accident-severity unit
    incident_zone_weight: float = 0.10
    congestion_zone_weight: float = 0.05         # added when a road cell hits >= threshold density
    congestion_density_threshold: float = 0.85
    zone_max: float = 1.0

    # Failure tuning
    failure_confidence: float = 0.80
    failure_decay: float = 0.985

    # Strategy tuning
    strategy_min_total: float = 0.80             # only record strategies with reward >= this
    strategy_decay: float = 0.99


class MemoryWriter:
    """Observes one tick and emits PastFailure / SuccessfulStrategy / HighRiskZone records.

    Driven externally by the trainer / runner — not auto-wired into the coordinator,
    so users can opt in and choose which signals matter.
    """

    def __init__(self, store: MemoryStore, *, config: WriterConfig | None = None) -> None:
        self.store = store
        self.config = config or WriterConfig()

    # ----- main entry ------------------------------------------------------

    def observe_tick(
        self,
        ctx: "VerificationContext",
        *,
        report: "VerificationReport | None" = None,
        breakdown: "RewardBreakdown | None" = None,
    ) -> dict:
        """Write zero or more records based on this tick's events. Returns counts added."""
        counts = {"failures": 0, "strategies": 0, "zones_new": 0, "zones_updated": 0}

        # 1. High-risk zones from new accidents.
        for a in ctx.curr_state.accidents:
            if a.spawned_tick != ctx.tick:
                continue
            outcome = self._note_zone(
                coord=a.coord,
                risk_factor=f"accident_sev{int(a.severity)}",
                weight=int(a.severity) * self.config.accident_zone_weight,
                tick=ctx.tick,
            )
            counts[outcome] = counts[outcome] + 1

        # 2. High-risk zones from new incidents.
        for iid in ctx.new_incidents:
            inc = ctx.agent_ctx.incidents.get(iid)
            if inc is None:
                continue
            outcome = self._note_zone(
                coord=inc.pos,
                risk_factor=f"incident_{inc.kind.value}",
                weight=inc.severity * self.config.incident_zone_weight,
                tick=ctx.tick,
            )
            counts[outcome] = counts[outcome] + 1

        # 3. Congestion hotspots (road cells above threshold).
        traffic = ctx.curr_state.traffic
        H, W = traffic.shape
        for y in range(H):
            for x in range(W):
                t = float(traffic[y, x])
                if t >= self.config.congestion_density_threshold:
                    outcome = self._note_zone(
                        coord=(x, y),
                        risk_factor="congestion_hotspot",
                        weight=self.config.congestion_zone_weight,
                        tick=ctx.tick,
                    )
                    counts[outcome] = counts[outcome] + 1

        # 4. Failures from the verifier.
        if report is not None and report.has_failures:
            from citynexus.verify.schemas import CheckStatus
            for r in report.results:
                if r.status != CheckStatus.FAIL:
                    continue
                self._note_failure(check_result=r, ctx=ctx)
                counts["failures"] += 1

        # 5. Successful strategies from positive-reward roles.
        if breakdown is not None:
            for role_value, agent_reward in breakdown.per_agent.items():
                if agent_reward.gated:
                    continue
                if agent_reward.total < self.config.strategy_min_total:
                    continue
                if not agent_reward.components:
                    continue
                self._note_strategy(role_value, agent_reward, ctx)
                counts["strategies"] += 1

        return counts

    # ----- emitters --------------------------------------------------------

    def _note_zone(self, *, coord: tuple[int, int], risk_factor: str, weight: float, tick: int) -> str:
        """Increment an existing zone covering `coord` or create a new one."""
        # Find an existing zone that contains this cell.
        for r in self.store.by_kind(MemoryKind.HIGH_RISK_ZONE):
            if isinstance(r, HighRiskZone) and coord in r.coords:
                r.sample_count += 1
                r.risk_score = min(self.config.zone_max, r.risk_score + weight)
                if risk_factor not in r.risk_factors:
                    r.risk_factors.append(risk_factor)
                r.timestamp = tick                   # refresh recency on update
                r.confidence = min(1.0, r.confidence + 0.02)
                return "zones_updated"
        # Else create new.
        new_zone = HighRiskZone(
            coords=[tuple(coord)],
            risk_score=min(self.config.zone_max, weight),
            risk_factors=[risk_factor],
            sample_count=1,
            timestamp=tick,
            confidence=0.5,
            decay_factor=0.997,
        )
        self.store.add(new_zone)
        return "zones_new"

    def _note_failure(self, *, check_result, ctx: "VerificationContext") -> None:
        context = {
            "weather": ctx.curr_state.weather.value,
            "hour_of_day": ctx.curr_state.hour_of_day,
            "congestion": round(ctx.curr_state.congestion_ratio(), 3),
            "n_accidents": len(ctx.curr_state.accidents),
            "n_incidents": len(ctx.agent_ctx.incidents),
            "layer": check_result.layer,
        }
        self.store.add(PastFailure(
            failure_mode=check_result.name,
            description=check_result.reason[:200],
            context=context,
            timestamp=ctx.tick,
            confidence=self.config.failure_confidence,
            decay_factor=self.config.failure_decay,
            suggested_avoidance=list(_FAILURE_TO_AVOIDANCE.get(check_result.name, [])),
        ))

    def _note_strategy(self, role_value: str, agent_reward, ctx: "VerificationContext") -> None:
        from citynexus.agents.base import AgentRole
        try:
            role_enum = AgentRole(role_value)
        except ValueError:
            return
        actions = ctx.actions.get(role_enum, [])
        action_kinds = sorted({type(a).__name__ for a in actions if type(a).__name__ != "NoOp"})
        if not action_kinds:
            return

        triggers = {
            "weather": ctx.curr_state.weather.value,
            "hour_of_day": ctx.curr_state.hour_of_day,
            "congestion": round(ctx.curr_state.congestion_ratio(), 2),
            "n_accidents": len(ctx.curr_state.accidents),
        }
        outcome = round(agent_reward.total, 3)
        self.store.add(SuccessfulStrategy(
            role=role_value,
            actions=action_kinds,
            triggers=triggers,
            outcome_score=outcome,
            pattern=f"{role_value}: {','.join(action_kinds)} → {outcome:+.2f}",
            timestamp=ctx.tick,
            confidence=min(1.0, max(0.5, agent_reward.total / 2.0)),
            decay_factor=self.config.strategy_decay,
        ))
