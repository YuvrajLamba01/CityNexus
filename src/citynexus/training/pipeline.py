"""TrainingPipeline — orchestrates env + agents + curriculum + reward + memory + logging.

One episode loop step:
  1. Curriculum picks difficulty + bias modes.
  2. AdversarialGenerator builds a Scenario.
  3. Coordinator is reset; PolicyBundle (if any) overrides agents' decide().
  4. Per-tick: shocks fire → agents act → reward system computes → memory writer
     observes → trajectory transitions are appended.
  5. Curriculum.update(metrics); periodic memory pruning + persistence.
  6. MetricsLogger.log per-episode summary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from random import Random
from typing import Callable

from citynexus.agents.base import AgentRole, BaseAgent
from citynexus.agents.coordinator import CoordinatorConfig, MultiAgentCoordinator
from citynexus.env.core import CityNexusEnv, EnvConfig
from citynexus.memory.store import MemoryStore
from citynexus.memory.writer import MemoryWriter
from citynexus.rewards.schemas import PerAgentReward, RewardBreakdown
from citynexus.rewards.system import MultiAgentRewardSystem, RewardSystemConfig
from citynexus.scenarios.generator import AdversarialGenerator, Curriculum
from citynexus.scenarios.schemas import EpisodeMetrics, Scenario
from citynexus.scenarios.shocks import apply_shock
from citynexus.training.config import TrainingConfig
from citynexus.training.metrics import MetricsLogger
from citynexus.training.policy import PolicyBundle
from citynexus.training.trajectory import Trajectory, Transition
from citynexus.verify.base import Verifier
from citynexus.verify.schemas import VerificationContext


@dataclass
class TrainingSummary:
    n_episodes: int
    final_difficulty: float
    mean_score: float
    last_window_avg_score: float
    mean_delivery_success: float
    last_window_avg_success: float
    cumulative_reward_per_role: dict[str, float]
    n_trajectories: int


class TrainingPipeline:
    """Train a multi-agent policy bundle against an adaptive curriculum.

    Construct with factories (called per-episode for reproducibility) plus an
    optional `policy_bundle` to override one or more agents' decide() with a
    learned model.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[int], CityNexusEnv] | None = None,
        agents_factory: Callable[[], list[BaseAgent]],
        config: TrainingConfig | None = None,
        policy_bundle: PolicyBundle | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.env_factory = env_factory or self._default_env_factory
        self.agents_factory = agents_factory
        self.policy_bundle = policy_bundle
        self.verifier = verifier or (Verifier.default() if self.config.use_verifier else None)

        self.curriculum = Curriculum(
            target_score=self.config.curriculum_target,
            alpha=self.config.curriculum_alpha,
            starting_difficulty=self.config.starting_difficulty,
        )
        self.generator = AdversarialGenerator(
            grid_size=self.config.grid_size,
            episode_length=self.config.max_ticks_per_episode,
            seed=self.config.seed,
        )
        self.memory: MemoryStore | None = (
            MemoryStore(path=self.config.memory_path) if self.config.use_memory else None
        )
        self.memory_writer: MemoryWriter | None = (
            MemoryWriter(self.memory) if self.memory is not None else None
        )
        self.reward_system = MultiAgentRewardSystem(
            verifier=self.verifier,
            config=RewardSystemConfig(gating_mode=self.config.gating_mode),
        )
        self.logger = MetricsLogger(
            log_dir=self.config.log_dir,
            console=self.config.console,
            window=self.config.log_window,
        )
        self.trajectories: list[Trajectory] = []

    # ----- public API -----------------------------------------------------

    def train(self) -> TrainingSummary:
        for ep in range(self.config.n_episodes):
            scenario = self._next_scenario()
            metrics, trajectory = self._run_episode(ep, scenario)
            self.curriculum.update(metrics)
            self.trajectories.append(trajectory)
            self._post_episode(ep, scenario, metrics, trajectory)
        if self.memory is not None and self.config.memory_path is not None:
            self.memory.save()
        return self.summary()

    def summary(self) -> TrainingSummary:
        if not self.trajectories:
            return TrainingSummary(
                n_episodes=0, final_difficulty=self.curriculum.next_difficulty(),
                mean_score=0.0, last_window_avg_score=0.0,
                mean_delivery_success=0.0, last_window_avg_success=0.0,
                cumulative_reward_per_role={}, n_trajectories=0,
            )
        scores = [t.final_metrics.overall_score for t in self.trajectories if t.final_metrics]
        succs = [t.final_metrics.delivery_success_rate for t in self.trajectories if t.final_metrics]
        w = self.config.log_window
        cum: dict[str, float] = {}
        for traj in self.trajectories:
            for role, val in traj.cumulative_per_agent.items():
                cum[role] = cum.get(role, 0.0) + val
        return TrainingSummary(
            n_episodes=len(self.trajectories),
            final_difficulty=self.curriculum.next_difficulty(),
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            last_window_avg_score=sum(scores[-w:]) / max(1, min(w, len(scores))) if scores else 0.0,
            mean_delivery_success=sum(succs) / len(succs) if succs else 0.0,
            last_window_avg_success=sum(succs[-w:]) / max(1, min(w, len(succs))) if succs else 0.0,
            cumulative_reward_per_role=cum,
            n_trajectories=len(self.trajectories),
        )

    # ----- episode internals ---------------------------------------------

    def _next_scenario(self) -> Scenario:
        bias = self.curriculum.top_failure_modes(n=self.config.bias_top_k_modes) if self.config.use_curriculum else None
        return self.generator.generate(self.curriculum.next_difficulty(), bias_toward=bias)

    def _run_episode(self, ep_idx: int, scenario: Scenario) -> tuple[EpisodeMetrics, Trajectory]:
        # Build env + agents fresh per episode.
        env = self.env_factory(self.config.seed + ep_idx)
        agents = self.agents_factory()
        if self.policy_bundle is not None:
            self.policy_bundle.install(agents)

        coord = MultiAgentCoordinator(
            env, agents,
            CoordinatorConfig(
                seed=self.config.seed + ep_idx,
                delivery_spawn_rate=self.config.delivery_spawn_rate,
                incident_spawn_rate=self.config.incident_spawn_rate,
            ),
            memory=self.memory,
        )
        coord.reset()
        coord.env.state.weather = scenario.initial_weather
        self.reward_system.reset()
        if self.policy_bundle is not None:
            self.policy_bundle.reset_all()

        ctx = coord.ctx
        shock_rng = Random(scenario.seed)
        weather_lock: tuple[int, object] | None = None

        per_role_transitions: dict[str, list[Transition]] = {role.value: [] for role in AgentRole}
        cum_reward: dict[str, float] = {role.value: 0.0 for role in AgentRole}
        sum_city = 0.0
        n_gated = 0
        shocks_fired = 0
        peak_cong = 0.0
        peak_acc = 0
        peak_inc = 0
        sum_cong = 0.0
        storm_ticks = 0

        for t in range(self.config.max_ticks_per_episode):
            current_tick = ctx.tick
            for shock in scenario.shocks_at(current_tick):
                report = apply_shock(coord.env, ctx, shock, shock_rng)
                shocks_fired += 1
                if report.weather_lock_until is not None:
                    weather_lock = (report.weather_lock_until, coord.env.state.weather)

            prev_state = copy.copy(coord.env.state)

            # Capture each agent's pre-action observation (for trajectory logging).
            tick_obs: dict[str, dict] = {}
            for role_enum, agent in coord.agents.items():
                tick_obs[role_enum.value] = agent.observe(coord.env.state, ctx)

            res = coord.step()

            if weather_lock is not None:
                if ctx.tick < weather_lock[0]:
                    coord.env.state.weather = weather_lock[1]
                else:
                    weather_lock = None

            v_ctx = VerificationContext(
                tick=res.step_info.tick,
                prev_state=prev_state,
                curr_state=coord.env.state,
                agent_ctx=ctx,
                actions=res.actions,
                completed_deliveries=res.completed_deliveries,
                new_deliveries=res.new_deliveries,
                new_incidents=res.new_incidents,
                accidents_cleared=res.step_info.cleared_accidents,
                accidents_spawned=len(res.step_info.new_accidents),
            )
            breakdown: RewardBreakdown = self.reward_system.compute(v_ctx)
            if self.memory_writer is not None:
                self.memory_writer.observe_tick(
                    v_ctx,
                    report=breakdown.verification_report,
                    breakdown=breakdown,
                )

            done = (t == self.config.max_ticks_per_episode - 1)
            for role_value in (r.value for r in AgentRole):
                actions = res.actions.get(AgentRole(role_value), [])
                agent_reward: PerAgentReward = breakdown.per_agent.get(
                    role_value, PerAgentReward(agent_role=role_value),
                )
                per_role_transitions[role_value].append(Transition(
                    tick=res.step_info.tick,
                    role=role_value,
                    obs=tick_obs[role_value],
                    actions=list(actions),
                    reward=agent_reward.total,
                    done=done,
                    info={"gated": agent_reward.gated},
                ))
                cum_reward[role_value] += agent_reward.total

            sum_city += breakdown.city_score.total
            if breakdown.gated_any:
                n_gated += 1

            world = coord.env.state
            cong = world.congestion_ratio()
            peak_cong = max(peak_cong, cong)
            sum_cong += cong
            if world.weather.value == "storm":
                storm_ticks += 1
            peak_acc = max(peak_acc, len(world.accidents))
            peak_inc = max(peak_inc, len(ctx.incidents))

        # --- finalise EpisodeMetrics --------------------------------------
        deliveries = list(ctx.deliveries.values())
        completed = sum(1 for d in deliveries if d.status.value == "delivered")
        failed = sum(1 for d in deliveries if d.status.value == "failed")
        open_d = sum(1 for d in deliveries if d.is_open)
        ticks_run = self.config.max_ticks_per_episode

        metrics = EpisodeMetrics(
            episode_id=f"ep-{ep_idx:05d}",
            scenario_id=scenario.id,
            difficulty=scenario.difficulty,
            ticks_run=ticks_run,
            deliveries_total=len(deliveries),
            deliveries_completed=completed,
            deliveries_failed=failed,
            deliveries_open=open_d,
            accidents_peak_concurrent=peak_acc,
            accidents_unresolved_at_end=len(coord.env.state.accidents),
            incidents_peak_concurrent=peak_inc,
            incidents_unresolved_at_end=len(ctx.incidents),
            peak_congestion=peak_cong,
            avg_congestion=sum_cong / max(1, ticks_run),
            storm_ticks=storm_ticks,
            messages_sent=ctx.bus.stats.get("sent", 0),
            shocks_fired=shocks_fired,
        )
        trajectory = Trajectory(
            episode_id=metrics.episode_id,
            scenario_id=scenario.id,
            role_transitions=per_role_transitions,
            final_metrics=metrics,
            cumulative_per_agent=cum_reward,
            avg_city_score=sum_city / max(1, ticks_run),
            n_gated_ticks=n_gated,
        )
        return metrics, trajectory

    def _post_episode(
        self,
        ep_idx: int,
        scenario: Scenario,
        metrics: EpisodeMetrics,
        trajectory: Trajectory,
    ) -> None:
        if self.memory is not None and (ep_idx + 1) % max(1, self.config.memory_prune_every) == 0:
            self.memory.prune(current_tick=metrics.ticks_run * (ep_idx + 1))
        self.logger.log(
            episode=ep_idx,
            phase="train",
            scenario_id=scenario.id,
            difficulty=scenario.difficulty,
            score=metrics.overall_score,
            delivery_success_rate=metrics.delivery_success_rate,
            deliveries_completed=metrics.deliveries_completed,
            deliveries_failed=metrics.deliveries_failed,
            accidents_peak=metrics.accidents_peak_concurrent,
            incidents_peak=metrics.incidents_peak_concurrent,
            peak_congestion=metrics.peak_congestion,
            shocks_fired=metrics.shocks_fired,
            messages_sent=metrics.messages_sent,
            summed_reward=trajectory.total_summed(),
            avg_city_score=trajectory.avg_city_score,
            n_gated_ticks=trajectory.n_gated_ticks,
            cumulative_per_agent=trajectory.cumulative_per_agent,
        )

    def _default_env_factory(self, seed: int) -> CityNexusEnv:
        return CityNexusEnv(EnvConfig(
            width=self.config.grid_size[0],
            height=self.config.grid_size[1],
            seed=seed,
            max_ticks=self.config.max_ticks_per_episode,
        ))
