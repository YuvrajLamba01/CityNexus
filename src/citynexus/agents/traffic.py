"""TrafficAgent — partial obs: intersection cells only (T-junctions and crossroads)."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent, NoOp
from citynexus.agents.messages import ClearanceRequest
from citynexus.agents.observability import TrafficView, build_traffic_view
from citynexus.agents.spaces import Box, DictSpace, Discrete, Space
from citynexus.env.world_state import WorldState


# --- Action types ------------------------------------------------------------

@dataclass
class PlaceRoadblock(Action):
    x: int = 0
    y: int = 0
    ttl: int = 5
    reason: str = "manage_traffic"
    kind: str = "place_roadblock"


@dataclass
class ClearRoadblock(Action):
    x: int = 0
    y: int = 0
    kind: str = "clear_roadblock"


@dataclass
class IssueAdvisory(Action):
    area: list[tuple[int, int]] = field(default_factory=list)
    severity: int = 1
    kind: str = "issue_advisory"


TRAFFIC_ACTION_TYPES: tuple[type[Action], ...] = (
    PlaceRoadblock,
    ClearRoadblock,
    IssueAdvisory,
    NoOp,
)


# --- Agent -------------------------------------------------------------------

class TrafficAgent(BaseAgent):
    """Sees only intersection cells. Acts on hot intersections + clearance asks."""

    def __init__(
        self,
        *,
        congestion_threshold: float = 0.55,
        advisory_top_n: int = 5,
        rng: Random | None = None,
    ) -> None:
        super().__init__(AgentRole.TRAFFIC, rng=rng)
        self.congestion_threshold = congestion_threshold
        self.advisory_top_n = advisory_top_n

    @property
    def observation_space(self) -> Space:
        return DictSpace({
            "weather": Discrete(3),
            "hour_of_day": Discrete(24),
            "intersection_avg_traffic": Box(0.0, 1.0, (1,)),
            "intersection_max_traffic": Box(0.0, 1.0, (1,)),
            "intersection_congestion": Box(0.0, 1.0, (1,)),
            "n_intersection_blocks": Discrete(64),
        })

    @property
    def action_space(self) -> Space:
        return Discrete(len(TRAFFIC_ACTION_TYPES))

    @property
    def action_types(self) -> tuple[type[Action], ...]:
        return TRAFFIC_ACTION_TYPES

    def observe(self, world: WorldState, ctx: AgentContext) -> dict:
        view = build_traffic_view(world)
        weather_to_idx = {"clear": 0, "rain": 1, "storm": 2}
        return {
            "weather": weather_to_idx[view.weather.value],
            "hour_of_day": view.hour_of_day,
            "intersection_avg_traffic": np.array([view.avg_traffic], dtype=np.float32),
            "intersection_max_traffic": np.array([view.max_traffic], dtype=np.float32),
            "intersection_congestion": np.array([view.congestion_ratio], dtype=np.float32),
            "n_intersection_blocks": min(63, len(view.intersection_blocks)),
            "_view": view,
        }

    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]:
        view: TrafficView = obs.get("_view")
        if view is None or not view.intersections:
            return [NoOp()]

        actions: list[Action] = []

        # 1. Issue advisory listing the hottest intersections.
        if view.congestion_ratio > self.congestion_threshold:
            scored = sorted(view.intersection_traffic.items(), key=lambda kv: kv[1], reverse=True)
            hot = [coord for coord, t in scored if t > 0.5][: self.advisory_top_n]
            if hot:
                severity = int(min(3, max(1, round(view.congestion_ratio * 3))))
                actions.append(IssueAdvisory(area=hot, severity=severity))

        # 2. Honour clearance requests from emergency / police (only at intersections we can see).
        for msg in ctx.receive(self.role, ClearanceRequest):
            x, y = msg.location
            if (int(x), int(y)) in view.intersections:
                actions.append(ClearRoadblock(x=int(x), y=int(y)))

        # 3. Clear our own stale blocks at intersections where traffic has dropped.
        for coord in view.intersection_blocks:
            t = view.intersection_traffic.get(coord, 0.0)
            if t < 0.15:
                actions.append(ClearRoadblock(x=coord[0], y=coord[1]))

        if not actions:
            actions.append(NoOp())
        return actions
