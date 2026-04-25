"""Reward computation + verification gating.

Pipeline:
    raw_components = RewardCalculator.compute(ctx)
    report         = Verifier.verify(ctx)
    final_reward   = GatedReward.compute(ctx)  → 0 if any check FAILed (strict)
                                                  raw × aggregate_score otherwise (non-strict)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from citynexus.verify.base import Verifier
from citynexus.verify.schemas import (
    CheckStatus,
    VerificationContext,
    VerificationReport,
)


# --- Reward components -----------------------------------------------------

@dataclass
class RewardComponents:
    delivery_completed: float = 0.0
    delivery_failed: float = 0.0
    accident_cleared: float = 0.0
    incident_resolved: float = 0.0
    congestion_drop: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.delivery_completed
            + self.delivery_failed
            + self.accident_cleared
            + self.incident_resolved
            + self.congestion_drop
        )

    def as_dict(self) -> dict:
        return {
            "delivery_completed": round(self.delivery_completed, 3),
            "delivery_failed": round(self.delivery_failed, 3),
            "accident_cleared": round(self.accident_cleared, 3),
            "incident_resolved": round(self.incident_resolved, 3),
            "congestion_drop": round(self.congestion_drop, 3),
            "total": round(self.total, 3),
        }


# --- Reward calculator -----------------------------------------------------

class RewardCalculator:
    """Pure per-tick reward decomposer. No verification gating happens here."""

    def __init__(
        self,
        *,
        r_delivery: float = 1.00,
        p_failed: float = -0.50,
        r_accident_clear: float = 0.50,
        r_incident_resolve: float = 0.30,
        r_congestion_drop: float = 0.20,
    ) -> None:
        self.r_delivery = r_delivery
        self.p_failed = p_failed
        self.r_accident_clear = r_accident_clear
        self.r_incident_resolve = r_incident_resolve
        self.r_congestion_drop = r_congestion_drop

    def compute(self, ctx: VerificationContext) -> RewardComponents:
        c = RewardComponents()

        c.delivery_completed = self.r_delivery * len(ctx.completed_deliveries)

        # Newly-failed deliveries this tick.
        new_failed = sum(
            1 for d in ctx.agent_ctx.deliveries.values()
            if d.status.value == "failed" and d.last_update_tick == ctx.tick
        )
        # Fallback if status flipping doesn't update last_update_tick:
        if new_failed == 0 and len(ctx.prev_state.accidents) > 0:
            # No reliable signal — leave at 0.
            pass
        c.delivery_failed = self.p_failed * new_failed

        c.accident_cleared = self.r_accident_clear * ctx.accidents_cleared

        # Incidents resolved = (prev count) - (curr count) - (newly spawned). Without prev,
        # use the simpler proxy: count incidents whose ttl ran to 0 OR were resolved.
        # ctx.agent_ctx.incidents contains only ACTIVE incidents (coordinator drops resolved/expired),
        # so we can't count them directly here. Heuristic: rely on the EmergencyAgent / PoliceAgent
        # actions that requested resolution. For v1, leave as 0 unless caller pre-fills.
        # (Caller can override RewardCalculator if they want a richer incident-resolve signal.)

        prev_cong = ctx.prev_state.congestion_ratio()
        curr_cong = ctx.curr_state.congestion_ratio()
        delta = prev_cong - curr_cong
        if delta > 0:
            c.congestion_drop = self.r_congestion_drop * delta

        return c


# --- Gated reward ----------------------------------------------------------

@dataclass
class GatedRewardResult:
    total: float                 # final reward after gating
    raw: float                   # raw reward before gating
    components: RewardComponents
    report: VerificationReport
    gated: bool                  # True if reward was zeroed by a FAIL
    aggregate_score: float       # report.aggregate_score()

    def summary(self) -> dict:
        return {
            "total": round(self.total, 3),
            "raw": round(self.raw, 3),
            "gated": self.gated,
            "aggregate_score": round(self.aggregate_score, 3),
            "components": self.components.as_dict(),
            "report": self.report.summary(),
        }


class GatedReward:
    """Combines a Verifier + RewardCalculator with strict-by-default gating.

    `strict=True`  (default): any check at FAIL → reward = 0.
    `strict=False`: reward is multiplied by `report.aggregate_score()`.
    WARN never gates; it can affect aggregate_score only in non-strict mode.
    """

    def __init__(
        self,
        verifier: Verifier,
        calculator: RewardCalculator | None = None,
        *,
        strict: bool = True,
    ) -> None:
        self.verifier = verifier
        self.calculator = calculator or RewardCalculator()
        self.strict = strict

    def compute(self, ctx: VerificationContext) -> GatedRewardResult:
        report = self.verifier.verify(ctx)
        components = self.calculator.compute(ctx)
        raw = components.total
        agg = report.aggregate_score()

        if self.strict and report.has_failures:
            return GatedRewardResult(
                total=0.0, raw=raw, components=components,
                report=report, gated=True, aggregate_score=agg,
            )
        if not self.strict:
            scaled = raw * agg
            return GatedRewardResult(
                total=scaled, raw=raw, components=components,
                report=report, gated=False, aggregate_score=agg,
            )
        return GatedRewardResult(
            total=raw, raw=raw, components=components,
            report=report, gated=False, aggregate_score=agg,
        )
