"""PlannerAgent — partial obs: aggregated metrics only (no per-cell, no per-entity)."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent, NoOp
from citynexus.agents.observability import PlannerView, build_planner_view
from citynexus.agents.spaces import Box, DictSpace, Discrete, Space
from citynexus.env.world_state import WorldState
from citynexus.memory.schemas import MemoryKind, PastFailure


# --- Action types ------------------------------------------------------------

@dataclass
class SetPriority(Action):
    role: AgentRole = AgentRole.TRAFFIC
    value: float = 1.0
    kind: str = "set_priority"


@dataclass
class BroadcastDirective(Action):
    body: str = ""
    payload: dict = field(default_factory=dict)
    kind: str = "broadcast"


PLANNER_ACTION_TYPES: tuple[type[Action], ...] = (
    SetPriority,
    BroadcastDirective,
    NoOp,
)


# --- Agent -------------------------------------------------------------------

class PlannerAgent(BaseAgent):
    """Sees nothing but aggregated city metrics. Re-balances per-role priority and
    emits a directive when the situation is acute."""

    def __init__(self, *, rng: Random | None = None) -> None:
        super().__init__(AgentRole.PLANNER, rng=rng)

    @property
    def observation_space(self) -> Space:
        return DictSpace({
            "tick_norm": Box(0.0, 1.0, (1,)),
            "weather": Discrete(3),
            "n_accidents": Discrete(64),
            "n_incidents": Discrete(64),
            "n_open_deliveries": Discrete(64),
            "congestion_ratio": Box(0.0, 1.0, (1,)),
            "n_idle_ambulances": Discrete(16),
            "n_idle_police": Discrete(16),
        })

    @property
    def action_space(self) -> Space:
        return Discrete(len(PLANNER_ACTION_TYPES))

    @property
    def action_types(self) -> tuple[type[Action], ...]:
        return PLANNER_ACTION_TYPES

    def observe(self, world: WorldState, ctx: AgentContext) -> dict:
        view = build_planner_view(world, ctx)
        weather_to_idx = {"clear": 0, "rain": 1, "storm": 2}
        return {
            "tick_norm": np.array([min(1.0, view.tick / 200.0)], dtype=np.float32),
            "weather": weather_to_idx[view.weather.value],
            "n_accidents": min(63, view.n_accidents),
            "n_incidents": min(63, view.n_incidents),
            "n_open_deliveries": min(63, view.n_open_deliveries),
            "congestion_ratio": np.array([view.congestion_ratio], dtype=np.float32),
            "n_idle_ambulances": min(15, view.n_units_idle.get("ambulance", 0)),
            "n_idle_police": min(15, view.n_units_idle.get("police_car", 0)),
            "_view": view,
        }

    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]:
        view: PlannerView = obs.get("_view")
        if view is None:
            return [NoOp()]

        actions: list[Action] = []
        target = {role: 1.0 for role in AgentRole}
        if view.n_accidents >= 3:
            target[AgentRole.EMERGENCY] = 2.0
        if view.n_incidents >= 2:
            target[AgentRole.POLICE] = 1.5
        if view.congestion_ratio > 0.5:
            target[AgentRole.TRAFFIC] = 1.5
        if view.n_open_deliveries >= 5:
            target[AgentRole.DELIVERY] = 1.5

        # Memory: anticipate based on recurrent past failure modes. Raise relevant
        # priority before its trigger condition fires this tick.
        if ctx.memory is not None:
            recent = ctx.memory.query(
                kind=MemoryKind.PAST_FAILURE,
                since=max(0, ctx.tick - 80),
                min_confidence=0.30,
                current_tick=ctx.tick,
            )
            mode_counts: dict[str, int] = {}
            for r in recent:
                if isinstance(r, PastFailure):
                    mode_counts[r.failure_mode] = mode_counts.get(r.failure_mode, 0) + 1
            if mode_counts.get("emergency_solved", 0) >= 2 or mode_counts.get("accident_response_latency", 0) >= 2:
                target[AgentRole.EMERGENCY] = max(target[AgentRole.EMERGENCY], 1.5)
            if mode_counts.get("congestion_reduced", 0) >= 2:
                target[AgentRole.TRAFFIC] = max(target[AgentRole.TRAFFIC], 1.5)
            if mode_counts.get("delivery_progress", 0) >= 2:
                target[AgentRole.DELIVERY] = max(target[AgentRole.DELIVERY], 1.3)

        for role, value in target.items():
            if abs(ctx.priorities.get(role, 1.0) - value) > 1e-6:
                actions.append(SetPriority(role=role, value=value))

        if view.n_accidents >= 5 and view.weather.value == "storm":
            actions.append(BroadcastDirective(
                body="surge protocol: storm + accident cluster",
                payload={"protocol": "surge", "reason": "storm_accidents"},
            ))

        if not actions:
            actions.append(NoOp())
        return actions
