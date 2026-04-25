"""Per-episode trajectory data for RL trainers.

Standard `(obs, action, reward, done, info)` transitions, decomposed per-agent.
A `Trajectory` per episode aggregates these plus the final EpisodeMetrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citynexus.scenarios.schemas import EpisodeMetrics


@dataclass
class Transition:
    """Per-tick, per-agent transition (RL-style)."""
    tick: int
    role: str
    obs: dict                        # the agent's observation BEFORE acting (encoded or raw dict)
    actions: list[Any]               # action(s) emitted this tick
    reward: float                    # per-agent reward this tick (post-gating)
    done: bool                       # True at terminal tick of episode
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    episode_id: str
    scenario_id: str
    role_transitions: dict[str, list[Transition]] = field(default_factory=dict)
    final_metrics: "EpisodeMetrics | None" = None
    cumulative_per_agent: dict[str, float] = field(default_factory=dict)
    avg_city_score: float = 0.0
    n_gated_ticks: int = 0

    def length(self, role: str | None = None) -> int:
        if role is None:
            return max((len(ts) for ts in self.role_transitions.values()), default=0)
        return len(self.role_transitions.get(role, []))

    def total_reward(self, role: str) -> float:
        return sum(t.reward for t in self.role_transitions.get(role, []))

    def total_summed(self) -> float:
        return sum(self.total_reward(r) for r in self.role_transitions)

    def to_jsonable(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "n_transitions_per_role": {r: len(ts) for r, ts in self.role_transitions.items()},
            "cumulative_per_agent": {r: round(v, 3) for r, v in self.cumulative_per_agent.items()},
            "avg_city_score": round(self.avg_city_score, 3),
            "n_gated_ticks": self.n_gated_ticks,
            "final_metrics": self.final_metrics.summary() if self.final_metrics else None,
        }
