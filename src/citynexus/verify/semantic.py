"""Layer 3 — semantic checks (optional).

The base `SemanticCheck` is intended to be overridden with an LLM judge for
richer evaluation. The two concrete classes below are rule-based heuristics
that approximate what a judge would assess; they're enabled by default so the
layer isn't dead, but a user can swap them for an `LLMJudgeCheck` at any time.
"""

from __future__ import annotations

from citynexus.agents.base import AgentRole
from citynexus.agents.emergency import DispatchUnit as EmergencyDispatch
from citynexus.verify.base import Check
from citynexus.verify.schemas import CheckResult, CheckStatus, VerificationContext


class SemanticCheck(Check):
    """Base for the optional semantic layer. Default = SKIP (so it doesn't gate rewards).

    To enable, subclass and implement `evaluate`, or use one of the rule-based
    concrete classes below. To plug in an LLM judge, subclass and call your
    LLM client inside `evaluate`.
    """

    layer = "semantic"

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        return self._result(CheckStatus.SKIP, reason="semantic check not implemented")


class PriorityCoherenceCheck(SemanticCheck):
    """Planner's priorities should reflect the situation (e.g. emergency ↑ when accidents pile up)."""

    name = "priority_coherence"
    attributed_to = (AgentRole.PLANNER.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        priorities = ctx.agent_ctx.priorities
        n_acc = len(ctx.curr_state.accidents)
        n_inc = len(ctx.agent_ctx.incidents)
        cong = ctx.curr_state.congestion_ratio()

        notes: list[str] = []
        score = 1.0
        if n_acc >= 3 and priorities.get(AgentRole.EMERGENCY, 1.0) < 1.5:
            score -= 0.30
            notes.append(
                f"emergency priority {priorities.get(AgentRole.EMERGENCY, 1.0):.1f} "
                f"too low for {n_acc} accidents"
            )
        if n_inc >= 2 and priorities.get(AgentRole.POLICE, 1.0) < 1.5:
            score -= 0.30
            notes.append(
                f"police priority {priorities.get(AgentRole.POLICE, 1.0):.1f} "
                f"too low for {n_inc} incidents"
            )
        if cong > 0.5 and priorities.get(AgentRole.TRAFFIC, 1.0) < 1.5:
            score -= 0.20
            notes.append(
                f"traffic priority {priorities.get(AgentRole.TRAFFIC, 1.0):.1f} "
                f"too low for congestion {cong:.2f}"
            )
        score = max(0.0, score)

        if score >= 0.7:
            status = CheckStatus.PASS
        elif score >= 0.4:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.FAIL
        return self._result(
            status, score=score,
            reason="; ".join(notes) or "priorities aligned with situation",
            metadata={"priorities": {r.value: v for r, v in priorities.items()}},
        )


class DispatchSeverityOrderCheck(SemanticCheck):
    """When dispatching, the highest-severity accidents should be addressed first."""

    name = "dispatch_severity_order"
    attributed_to = (AgentRole.EMERGENCY.value,)

    def evaluate(self, ctx: VerificationContext) -> CheckResult:
        emergency_actions = ctx.actions.get(AgentRole.EMERGENCY, [])
        dispatches = [a for a in emergency_actions if isinstance(a, EmergencyDispatch)]
        if not dispatches:
            return self._result(CheckStatus.SKIP, reason="no dispatches this tick")

        accidents = ctx.curr_state.accidents
        if not accidents:
            return self._result(CheckStatus.SKIP, reason="no accidents to compare against")

        # Severity at each dispatch target (look at curr accident list).
        target_sev: list[int] = []
        for d in dispatches:
            for a in accidents:
                if a.coord == (d.x, d.y):
                    target_sev.append(int(a.severity))
                    break
        if not target_sev:
            return self._result(
                CheckStatus.WARN, score=0.5,
                reason="dispatched but no matching accident at target",
            )

        all_sev = sorted([int(a.severity) for a in accidents], reverse=True)
        top_k = all_sev[: len(target_sev)]
        avg_dispatched = sum(target_sev) / len(target_sev)
        avg_top_k = sum(top_k) / len(top_k)
        ratio = avg_dispatched / max(1.0, avg_top_k)

        if ratio >= 0.85:
            return self._result(
                CheckStatus.PASS, score=min(1.0, ratio),
                reason=f"avg dispatched severity {avg_dispatched:.2f} ≈ top-{len(target_sev)} {avg_top_k:.2f}",
            )
        if ratio >= 0.5:
            return self._result(
                CheckStatus.WARN, score=ratio,
                reason=f"low dispatch severity {avg_dispatched:.2f} vs top {avg_top_k:.2f}",
            )
        return self._result(
            CheckStatus.FAIL, score=ratio,
            reason=f"dispatched low-severity ({avg_dispatched:.2f}) while higher were waiting ({avg_top_k:.2f})",
        )


def default_semantic_checks() -> list[Check]:
    return [
        PriorityCoherenceCheck(),
        DispatchSeverityOrderCheck(),
    ]
