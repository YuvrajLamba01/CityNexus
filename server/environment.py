"""CityNexusEnvironment — OpenEnv-compatible wrapper around the CITYNEXUS engine.

Surface (Gym-style):
    obs = env.reset(seed=42)
    obs = env.step(CityAction(mode="emergency_focus"))
    state = env.state                   # CityNexusEnvState

The wrapper:
  * owns a `MultiAgentCoordinator` driving the existing 5-role engine,
  * installs a Planner policy override that translates `CityAction.mode`
    into per-role `SetPriority` directives (so the action actually steers
    the multi-agent loop),
  * computes per-agent rewards via the existing `MultiAgentRewardSystem`
    with verifier-gated, process-aware components,
  * returns a structured `CityObservation` whose `reward` field is the
    summed per-agent reward for the tick.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from openenv.core.env_server import Environment

from citynexus.agents import (
    CoordinatorConfig,
    DeliveryAgent,
    EmergencyAgent,
    MultiAgentCoordinator,
    PlannerAgent,
    PoliceAgent,
    TrafficAgent,
)
from citynexus.agents.base import AgentRole, NoOp
from citynexus.agents.planner import BroadcastDirective, SetPriority
from citynexus.env.core import CityNexusEnv, EnvConfig
from citynexus.rewards import MultiAgentRewardSystem, RewardSystemConfig
from citynexus.verify import Verifier
from citynexus.verify.schemas import VerificationContext

from server.models import CITY_MODES, CityAction, CityNexusEnvState, CityObservation


# Mode → per-role priority targets.  The Planner override emits SetPriority
# actions that drive the heuristic agents toward this distribution.
_MODE_PRIORITIES: dict[str, dict[AgentRole, float]] = {
    "normal": {
        AgentRole.DELIVERY: 1.0,
        AgentRole.TRAFFIC: 1.0,
        AgentRole.EMERGENCY: 1.0,
        AgentRole.POLICE: 1.0,
        AgentRole.PLANNER: 1.0,
    },
    "emergency_focus": {
        AgentRole.DELIVERY: 0.7,
        AgentRole.TRAFFIC: 1.2,
        AgentRole.EMERGENCY: 2.0,
        AgentRole.POLICE: 1.5,
        AgentRole.PLANNER: 1.0,
    },
    "delivery_focus": {
        AgentRole.DELIVERY: 2.0,
        AgentRole.TRAFFIC: 1.5,
        AgentRole.EMERGENCY: 0.8,
        AgentRole.POLICE: 0.8,
        AgentRole.PLANNER: 1.0,
    },
    "defensive": {
        AgentRole.DELIVERY: 0.7,
        AgentRole.TRAFFIC: 1.5,
        AgentRole.EMERGENCY: 1.2,
        AgentRole.POLICE: 1.2,
        AgentRole.PLANNER: 2.0,
    },
}


class _ModePlannerPolicy:
    """Replaces PlannerAgent.decide() while the OpenEnv wrapper is in control.

    Each tick it reads the current mode + directive off the env and emits
    the matching SetPriority / BroadcastDirective actions.
    """

    def __init__(self, env: "CityNexusEnvironment") -> None:
        self._env = env

    def __call__(self, obs: dict, ctx) -> list:
        mode = self._env._current_mode
        directive = self._env._current_directive
        targets = _MODE_PRIORITIES.get(mode, _MODE_PRIORITIES["normal"])

        actions: list = []
        for role, value in targets.items():
            if abs(ctx.priorities.get(role, 1.0) - value) > 1e-6:
                actions.append(SetPriority(role=role, value=value))

        if directive:
            actions.append(BroadcastDirective(
                body=directive,
                payload={"protocol": "llm_directive", "mode": mode},
            ))

        if not actions:
            actions.append(NoOp())
        return actions


class CityNexusEnvironment(Environment[CityAction, CityObservation, CityNexusEnvState]):
    """OpenEnv `Environment` subclass exposing CITYNEXUS as a Gym-style env."""

    SUPPORTS_CONCURRENT_SESSIONS = False
    REQUIRES_SINGLE_THREAD_EXECUTOR = False

    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        max_ticks: int = 100,
        delivery_spawn_rate: float = 0.30,
        incident_spawn_rate: float = 0.10,
        default_seed: int = 42,
    ) -> None:
        self._cfg_width = width
        self._cfg_height = height
        self._cfg_max_ticks = max_ticks
        self._cfg_delivery_spawn = delivery_spawn_rate
        self._cfg_incident_spawn = incident_spawn_rate
        self._default_seed = default_seed

        self._engine: CityNexusEnv | None = None
        self._coord: MultiAgentCoordinator | None = None
        self._reward_system: MultiAgentRewardSystem | None = None
        self._verifier: Verifier | None = None

        self._state = CityNexusEnvState(
            episode_id=None,
            step_count=0,
            seed=default_seed,
            max_ticks=max_ticks,
            cumulative_reward=0.0,
            mode_history=[],
        )
        self._cumulative_per_agent: dict[str, float] = {}
        self._current_mode: str = "normal"
        self._current_directive: str | None = None

    # ----- OpenEnv API -----------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> CityObservation:
        s = self._default_seed if seed is None else int(seed)
        eid = episode_id or f"ep-{uuid.uuid4().hex[:8]}"

        self._engine = CityNexusEnv(EnvConfig(
            width=self._cfg_width,
            height=self._cfg_height,
            seed=s,
            max_ticks=self._cfg_max_ticks,
        ))
        agents = [
            DeliveryAgent(),
            TrafficAgent(),
            EmergencyAgent(),
            PoliceAgent(),
            PlannerAgent(),
        ]
        self._coord = MultiAgentCoordinator(
            self._engine,
            agents,
            CoordinatorConfig(
                seed=s,
                delivery_spawn_rate=self._cfg_delivery_spawn,
                incident_spawn_rate=self._cfg_incident_spawn,
            ),
        )
        self._coord.reset()

        # Hand the Planner over to the LLM-driven mode policy.
        self._coord.agents[AgentRole.PLANNER].set_policy(_ModePlannerPolicy(self))

        self._verifier = Verifier.default()
        self._reward_system = MultiAgentRewardSystem(
            verifier=self._verifier,
            config=RewardSystemConfig(),
        )
        self._reward_system.reset()

        self._cumulative_per_agent = {role.value: 0.0 for role in AgentRole}
        self._current_mode = "normal"
        self._current_directive = None

        self._state = CityNexusEnvState(
            episode_id=eid,
            step_count=0,
            seed=s,
            max_ticks=self._cfg_max_ticks,
            cumulative_reward=0.0,
            mode_history=[],
        )

        return self._snapshot_observation(
            per_agent_reward={r.value: 0.0 for r in AgentRole},
            tick_reward=0.0,
            last_actions={r.value: [] for r in AgentRole},
            recent_messages=[],
            gated_any=False,
            info={"event": "reset"},
        )

    def step(
        self,
        action: CityAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CityObservation:
        if self._coord is None or self._engine is None or self._reward_system is None:
            raise RuntimeError("Call reset() before step().")

        mode = action.mode if action.mode in CITY_MODES else "normal"
        self._current_mode = mode
        self._current_directive = action.directive
        self._state.mode_history.append(mode)

        # Snapshot the world before the tick so the verifier can diff prev/curr.
        prev_state = copy.copy(self._engine.state)

        # Run one tick of the multi-agent system.
        tick_result = self._coord.step()

        # Compute rewards on the post-tick state.
        ver_ctx = VerificationContext(
            tick=tick_result.step_info.tick,
            prev_state=prev_state,
            curr_state=self._engine.state,
            agent_ctx=self._coord.ctx,
            actions=tick_result.actions,
            completed_deliveries=tick_result.completed_deliveries,
            new_deliveries=tick_result.new_deliveries,
            new_incidents=tick_result.new_incidents,
            accidents_cleared=tick_result.step_info.cleared_accidents,
            accidents_spawned=len(tick_result.step_info.new_accidents),
        )
        breakdown = self._reward_system.compute(ver_ctx)

        per_agent_reward = {
            role: agent.total for role, agent in breakdown.per_agent.items()
        }
        for role, value in per_agent_reward.items():
            self._cumulative_per_agent[role] = (
                self._cumulative_per_agent.get(role, 0.0) + value
            )
        tick_reward = sum(per_agent_reward.values())
        self._state.cumulative_reward += tick_reward
        self._state.step_count = self._coord.ctx.tick

        last_actions: dict[str, list[str]] = {}
        for role, role_actions in tick_result.actions.items():
            last_actions[role.value] = [
                getattr(a, "kind", type(a).__name__) for a in role_actions
            ]

        recent_messages = self._format_recent_messages(tick_result.messages_by_kind)

        info = {
            "city_score": breakdown.city_score.total,
            "n_new_deliveries": len(tick_result.new_deliveries),
            "n_new_incidents": len(tick_result.new_incidents),
            "n_completed_deliveries": len(tick_result.completed_deliveries),
            "messages_delivered": tick_result.messages_delivered,
            "mode": mode,
        }

        return self._snapshot_observation(
            per_agent_reward=per_agent_reward,
            tick_reward=tick_reward,
            last_actions=last_actions,
            recent_messages=recent_messages,
            gated_any=breakdown.gated_any,
            info=info,
            city_score=breakdown.city_score.total,
        )

    @property
    def state(self) -> CityNexusEnvState:
        return self._state

    def close(self) -> None:
        self._engine = None
        self._coord = None
        self._reward_system = None
        self._verifier = None

    # ----- helpers ---------------------------------------------------------

    def _snapshot_observation(
        self,
        *,
        per_agent_reward: dict[str, float],
        tick_reward: float,
        last_actions: dict[str, list[str]],
        recent_messages: list[str],
        gated_any: bool,
        info: dict[str, Any],
        city_score: float | None = None,
    ) -> CityObservation:
        assert self._engine is not None and self._coord is not None
        world = self._engine.state
        ctx = self._coord.ctx

        deliveries = list(ctx.deliveries.values())
        n_open = sum(1 for d in deliveries if d.is_open)
        n_done = sum(1 for d in deliveries if d.status.value == "delivered")
        n_failed = sum(1 for d in deliveries if d.status.value == "failed")

        priorities = {role.value: ctx.priorities.get(role, 1.0) for role in AgentRole}
        done = self._engine.done or self._coord.ctx.tick >= self._cfg_max_ticks

        return CityObservation(
            done=done,
            reward=float(tick_reward),
            tick=self._coord.ctx.tick,
            seed=self._state.seed,
            weather=world.weather.value,
            avg_traffic=float(world.avg_traffic()),
            congestion_ratio=float(world.congestion_ratio()),
            n_active_accidents=len(world.accidents),
            n_active_incidents=len(ctx.incidents),
            n_active_roadblocks=len(world.roadblocks),
            n_open_deliveries=n_open,
            n_completed_deliveries=n_done,
            n_failed_deliveries=n_failed,
            city_score=float(city_score) if city_score is not None else 0.0,
            per_agent_reward=per_agent_reward,
            cumulative_per_agent=dict(self._cumulative_per_agent),
            priorities=priorities,
            last_actions=last_actions,
            recent_messages=recent_messages,
            gated_any=gated_any,
            info=info,
        )

    @staticmethod
    def _format_recent_messages(by_kind: dict[str, int]) -> list[str]:
        if not by_kind:
            return []
        return [f"{k}×{v}" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])[:8]]
