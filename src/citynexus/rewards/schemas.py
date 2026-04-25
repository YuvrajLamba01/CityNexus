"""Reward schemas: per-agent breakdown, global city score, and the GatingMode enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citynexus.verify.schemas import VerificationReport


class GatingMode(str, Enum):
    """How verifier failures gate rewards."""
    NONE = "none"             # don't gate; raw rewards passthrough
    ATTRIBUTED = "attributed"  # zero only roles named in failed checks' attributed_to
    ALL = "all"                # any FAIL → all per-agent rewards zeroed (matches strict spec)


@dataclass
class PerAgentReward:
    """One agent's per-tick reward, decomposed by component + penalty."""
    agent_role: str
    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)   # positive contributions
    penalties: dict[str, float] = field(default_factory=dict)    # negative contributions
    gated: bool = False                                          # zeroed by verifier this tick

    def add_component(self, name: str, value: float) -> None:
        if value == 0.0:
            return
        self.components[name] = self.components.get(name, 0.0) + value
        self.total += value

    def add_penalty(self, name: str, value: float) -> None:
        if value == 0.0:
            return
        self.penalties[name] = self.penalties.get(name, 0.0) + value
        self.total += value

    def zero(self) -> None:
        self.total = 0.0
        self.components = {}
        # Penalties retained: they stay informational so the agent (or trainer) can see *why*
        # the reward was gated, not just that it was.
        self.gated = True

    def summary(self) -> dict:
        return {
            "role": self.agent_role,
            "total": round(self.total, 3),
            "gated": self.gated,
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "penalties": {k: round(v, 3) for k, v in self.penalties.items()},
        }


@dataclass
class CityScore:
    """Global per-tick health metric in [0, 1]. Composed of weighted sub-scores."""
    total: float = 0.0
    delivery_health: float = 0.0      # completed / (completed + failed)
    safety: float = 0.0               # 1 - normalized active accident burden
    mobility: float = 0.0             # 1 - congestion ratio
    coordination: float = 0.0         # planner priorities aligned with observed load
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "total": round(self.total, 3),
            "delivery_health": round(self.delivery_health, 3),
            "safety": round(self.safety, 3),
            "mobility": round(self.mobility, 3),
            "coordination": round(self.coordination, 3),
        }


@dataclass
class RewardBreakdown:
    """One tick's full reward decomposition."""
    tick: int
    per_agent: dict[str, PerAgentReward]
    city_score: CityScore
    gated_any: bool = False
    verification_report: "VerificationReport | None" = None

    def total_per_agent(self) -> dict[str, float]:
        return {role: agent.total for role, agent in self.per_agent.items()}

    def total_summed(self) -> float:
        return sum(agent.total for agent in self.per_agent.values())

    def summary(self) -> dict:
        return {
            "tick": self.tick,
            "per_agent": {role: a.summary() for role, a in self.per_agent.items()},
            "city_score": self.city_score.summary(),
            "gated_any": self.gated_any,
            "summed": round(self.total_summed(), 3),
        }
