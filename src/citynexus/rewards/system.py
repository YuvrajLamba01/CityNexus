"""MultiAgentRewardSystem — top-level reward facade.

Stateful: tracks across-tick deltas the components can't compute on their own
(prev unit positions, prev priorities, prev incident snapshot, prev failed-delivery
count). Call `reset()` at episode start.

Wires together:
  * per-agent outcome rewards
  * per-agent penalties (delays, collisions, inefficiency)
  * process-aware rewards (progress, intent, anticipation)
  * a global city score
  * planner's "system share" of every positive reward
  * verifier-driven gating (NONE | ATTRIBUTED | ALL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from citynexus.agents.base import AgentRole
from citynexus.rewards.components import (
    accident_clearance_reward,
    collision_penalty,
    compute_city_score,
    congestion_management_reward,
    congestion_rise_penalty,
    deadline_pressure_penalty,
    delivery_completion_reward,
    delivery_failure_penalty,
    dispatch_intent_reward,
    idle_unit_inefficiency,
    incident_resolution_reward,
    planner_anticipation_reward,
    progress_toward_target_reward,
    redundant_dispatch_penalty,
)
from citynexus.rewards.schemas import (
    CityScore,
    GatingMode,
    PerAgentReward,
    RewardBreakdown,
)
from citynexus.verify.schemas import CheckStatus

if TYPE_CHECKING:
    from citynexus.verify.base import Verifier
    from citynexus.verify.schemas import VerificationContext


@dataclass
class RewardSystemConfig:
    # Outcome rewards
    r_delivery_completion: float = 1.00
    r_accident_clear:      float = 0.50
    r_incident_resolve:    float = 0.30
    r_congestion_drop:     float = 0.30

    # Penalties (negative weights)
    p_delivery_failure:    float = -0.50
    p_deadline_pressure:   float = -0.05    # per-tick while past deadline
    p_congestion_rise:     float = -0.20
    p_collision:           float = -0.30    # per high-traffic-cell accident
    p_idle_unit:           float = -0.05    # per idle unit when work is available
    p_redundant_dispatch:  float = -0.10    # per duplicate dispatch target

    # Process-aware rewards
    r_progress:            float = 0.05    # per cell of Manhattan progress
    r_dispatch_intent:     float = 0.10    # per dispatch, scaled by severity / 3
    r_planner_anticipation: float = 0.10   # per aligned priority (incl. fresh-raise bonus)

    # Coordination — planner's share of system-wide positive contributions
    planner_system_share:  float = 0.15

    # City-score weights (None → use defaults inside compute_city_score)
    city_weights: dict | None = None

    # Verifier gating
    gating_mode: GatingMode = GatingMode.ATTRIBUTED


@dataclass
class _PrevSnapshot:
    """Per-episode rolling state the components can't get from VerificationContext alone."""
    unit_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    priorities: dict[AgentRole, float] = field(default_factory=dict)
    incidents: dict[str, dict] = field(default_factory=dict)   # id → {assigned_unit}
    failed_count: int = 0


class MultiAgentRewardSystem:
    """Compute per-agent + global rewards each tick. Optionally gated by a Verifier."""

    def __init__(
        self,
        *,
        verifier: "Verifier | None" = None,
        config: RewardSystemConfig | None = None,
    ) -> None:
        self.verifier = verifier
        self.config = config or RewardSystemConfig()
        self._prev = _PrevSnapshot()

    # ----- lifecycle ------------------------------------------------------

    def reset(self) -> None:
        self._prev = _PrevSnapshot()

    # ----- main entry -----------------------------------------------------

    def compute(self, ctx: "VerificationContext") -> RewardBreakdown:
        cfg = self.config

        # 0. Verifier (optional).
        report = self.verifier.verify(ctx) if self.verifier else None

        # 1. Pre-compute deltas the components can't see themselves.
        cur_failed = sum(
            1 for d in ctx.agent_ctx.deliveries.values() if d.status.value == "failed"
        )
        new_failed = max(0, cur_failed - self._prev.failed_count)

        cur_incident_ids = set(ctx.agent_ctx.incidents.keys())
        prev_incident_ids = set(self._prev.incidents.keys())
        # Resolved = was active last tick, gone now, AND had been assigned (else: expired).
        resolved_count = sum(
            1 for iid in (prev_incident_ids - cur_incident_ids)
            if self._prev.incidents.get(iid, {}).get("assigned_unit")
        )

        # 2. Initialise per-agent buckets.
        per_agent: dict[str, PerAgentReward] = {
            role.value: PerAgentReward(agent_role=role.value) for role in AgentRole
        }

        # 3. Outcome rewards.
        self._distribute(per_agent, "delivery_completion",
                         delivery_completion_reward(ctx, cfg.r_delivery_completion))
        self._distribute(per_agent, "accident_clearance",
                         accident_clearance_reward(ctx, cfg.r_accident_clear))
        self._distribute(per_agent, "incident_resolution",
                         incident_resolution_reward(ctx, resolved_count=resolved_count,
                                                    weight=cfg.r_incident_resolve))
        self._distribute(per_agent, "congestion_management",
                         congestion_management_reward(ctx, cfg.r_congestion_drop))

        # 4. Penalties.
        self._distribute_penalty(per_agent, "delivery_failure",
                                 delivery_failure_penalty(ctx, new_failed=new_failed,
                                                          weight=cfg.p_delivery_failure))
        self._distribute_penalty(per_agent, "deadline_pressure",
                                 deadline_pressure_penalty(ctx, cfg.p_deadline_pressure))
        self._distribute_penalty(per_agent, "congestion_rise",
                                 congestion_rise_penalty(ctx, cfg.p_congestion_rise))
        self._distribute_penalty(per_agent, "collision",
                                 collision_penalty(ctx, cfg.p_collision))
        self._distribute_penalty(per_agent, "idle_unit",
                                 idle_unit_inefficiency(ctx, cfg.p_idle_unit))
        self._distribute_penalty(per_agent, "redundant_dispatch",
                                 redundant_dispatch_penalty(ctx, cfg.p_redundant_dispatch))

        # 5. Process-aware rewards.
        self._distribute(per_agent, "progress",
                         progress_toward_target_reward(
                             ctx, prev_positions=self._prev.unit_positions,
                             weight=cfg.r_progress,
                         ))
        self._distribute(per_agent, "dispatch_intent",
                         dispatch_intent_reward(ctx, cfg.r_dispatch_intent))
        self._distribute(per_agent, "anticipation",
                         planner_anticipation_reward(
                             ctx, prev_priorities=self._prev.priorities,
                             weight=cfg.r_planner_anticipation,
                         ))

        # 6. Planner's system share — slice of positive contributions from other roles.
        positive_others = sum(
            max(0.0, agent.total)
            for role, agent in per_agent.items()
            if role != AgentRole.PLANNER.value
        )
        share = cfg.planner_system_share * positive_others
        if share > 0:
            per_agent[AgentRole.PLANNER.value].add_component("system_share", share)

        # 7. Global city score.
        city = compute_city_score(ctx, weights=cfg.city_weights)

        # 8. Apply gating.
        gated_any = False
        if self.verifier and report is not None and report.has_failures:
            gated_any = self._apply_gating(per_agent, report, cfg.gating_mode)

        # 9. Update rolling snapshot for next tick.
        self._prev = _PrevSnapshot(
            unit_positions={u.id: u.pos for u in ctx.agent_ctx.units.values()},
            priorities=dict(ctx.agent_ctx.priorities),
            incidents={
                iid: {"assigned_unit": inc.assigned_unit}
                for iid, inc in ctx.agent_ctx.incidents.items()
            },
            failed_count=cur_failed,
        )

        return RewardBreakdown(
            tick=ctx.tick,
            per_agent=per_agent,
            city_score=city,
            gated_any=gated_any,
            verification_report=report,
        )

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _distribute(
        per_agent: dict[str, PerAgentReward],
        name: str,
        contributions: dict[str, float],
    ) -> None:
        for role_value, value in contributions.items():
            agent = per_agent.get(role_value)
            if agent is not None:
                agent.add_component(name, value)

    @staticmethod
    def _distribute_penalty(
        per_agent: dict[str, PerAgentReward],
        name: str,
        contributions: dict[str, float],
    ) -> None:
        for role_value, value in contributions.items():
            agent = per_agent.get(role_value)
            if agent is not None:
                agent.add_penalty(name, value)

    @staticmethod
    def _apply_gating(
        per_agent: dict[str, PerAgentReward],
        report,
        mode: GatingMode,
    ) -> bool:
        if mode == GatingMode.NONE:
            return False

        if mode == GatingMode.ALL:
            for agent in per_agent.values():
                agent.zero()
            return True

        # ATTRIBUTED: zero only roles named in failed checks' attributed_to.
        roles_to_zero: set[str] = set()
        for r in report.results:
            if r.status == CheckStatus.FAIL:
                for role_value in r.attributed_to:
                    roles_to_zero.add(role_value)
        if not roles_to_zero:
            return False
        for role_value in roles_to_zero:
            agent = per_agent.get(role_value)
            if agent is not None:
                agent.zero()
        return True
