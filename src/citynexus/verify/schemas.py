"""Verification schemas: status, result, report, and the per-tick context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citynexus.agents.base import Action, AgentContext, AgentRole
    from citynexus.env.world_state import WorldState


class CheckStatus(str, Enum):
    PASS = "pass"     # check OK
    FAIL = "fail"     # check failed → reward gating triggers
    SKIP = "skip"     # not applicable this tick
    WARN = "warn"     # weak signal, partial credit; does NOT gate reward


@dataclass
class CheckResult:
    name: str
    layer: str                          # "programmatic" | "system_state" | "semantic"
    status: CheckStatus
    score: float = 1.0                  # 0.0–1.0; for partial credit / non-strict mode
    reason: str = ""
    attributed_to: tuple[str, ...] = ()   # role values this check is about
    metadata: dict = field(default_factory=dict)


@dataclass
class VerificationReport:
    tick: int
    results: list[CheckResult] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return any(r.status == CheckStatus.FAIL for r in self.results)

    @property
    def passed(self) -> bool:
        return not self.has_failures

    def by_layer(self, layer: str) -> list[CheckResult]:
        return [r for r in self.results if r.layer == layer]

    def by_status(self, status: CheckStatus) -> list[CheckResult]:
        return [r for r in self.results if r.status == status]

    def aggregate_score(self) -> float:
        """Mean score across all non-skipped checks. Used for non-strict gating."""
        scored = [r.score for r in self.results if r.status != CheckStatus.SKIP]
        if not scored:
            return 1.0
        return sum(scored) / len(scored)

    def summary(self) -> dict:
        by_status: dict[str, int] = {}
        for r in self.results:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "tick": self.tick,
            "total": len(self.results),
            "by_status": by_status,
            "aggregate_score": round(self.aggregate_score(), 3),
            "passed": self.passed,
            "failures": [
                {"name": r.name, "layer": r.layer, "reason": r.reason}
                for r in self.results if r.status == CheckStatus.FAIL
            ],
        }


@dataclass
class VerificationContext:
    """Snapshot of everything a check might read.

    Built once per tick by the caller (typically right after `coordinator.step()`).
    """
    tick: int
    prev_state: "WorldState"
    curr_state: "WorldState"
    agent_ctx: "AgentContext"
    actions: dict["AgentRole", list["Action"]] = field(default_factory=dict)
    completed_deliveries: list[str] = field(default_factory=list)
    new_deliveries: list[str] = field(default_factory=list)
    new_incidents: list[str] = field(default_factory=list)
    accidents_cleared: int = 0
    accidents_spawned: int = 0
    shocks_fired_this_tick: int = 0
