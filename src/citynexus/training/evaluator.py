"""Evaluator — fixed-difficulty episodes for fair baseline-vs-trained comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Callable

from citynexus.agents.base import BaseAgent
from citynexus.agents.coordinator import CoordinatorConfig, MultiAgentCoordinator
from citynexus.env.core import CityNexusEnv, EnvConfig
from citynexus.scenarios.generator import AdversarialGenerator
from citynexus.scenarios.runner import EpisodeRunner, EpisodeRunnerConfig
from citynexus.scenarios.schemas import EpisodeMetrics
from citynexus.training.config import EvalConfig
from citynexus.training.metrics import MetricsLogger
from citynexus.training.policy import PolicyBundle


@dataclass
class EvalResult:
    name: str
    n_episodes: int
    scores: list[float] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    metrics: list[EpisodeMetrics] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return mean(self.scores) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        return stdev(self.scores) if len(self.scores) >= 2 else 0.0

    @property
    def mean_success_rate(self) -> float:
        return mean(self.success_rates) if self.success_rates else 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_episodes": self.n_episodes,
            "mean_score": round(self.mean_score, 3),
            "std_score": round(self.std_score, 3),
            "mean_success_rate": round(self.mean_success_rate, 3),
        }


@dataclass
class ComparisonResult:
    baseline: EvalResult
    trained: EvalResult
    delta_score: float
    delta_success_rate: float

    def summary(self) -> dict:
        return {
            "baseline": self.baseline.summary(),
            "trained": self.trained.summary(),
            "delta_score": round(self.delta_score, 3),
            "delta_success_rate": round(self.delta_success_rate, 3),
        }


class Evaluator:
    """Run held-out evaluation episodes at a fixed difficulty.

    Both arms (baseline / trained) see the SAME seeds and SAME scenarios so the
    comparison isolates policy quality.
    """

    def __init__(
        self,
        *,
        agents_factory: Callable[[], list[BaseAgent]],
        env_factory: Callable[[int], CityNexusEnv] | None = None,
        config: EvalConfig | None = None,
        logger: MetricsLogger | None = None,
    ) -> None:
        self.config = config or EvalConfig()
        self.agents_factory = agents_factory
        self.env_factory = env_factory or self._default_env_factory
        self.logger = logger
        self._generator = AdversarialGenerator(
            grid_size=self.config.grid_size,
            episode_length=self.config.max_ticks_per_episode,
            seed=self.config.base_seed,
        )

    # ----- public API -----------------------------------------------------

    def evaluate(
        self,
        *,
        name: str = "eval",
        policy_bundle: PolicyBundle | None = None,
    ) -> EvalResult:
        seeds = self._seeds()
        result = EvalResult(name=name, n_episodes=len(seeds))
        for i, seed in enumerate(seeds):
            metrics = self._run_one(seed=seed, policy_bundle=policy_bundle)
            result.scores.append(metrics.overall_score)
            result.success_rates.append(metrics.delivery_success_rate)
            result.metrics.append(metrics)
            if self.logger is not None:
                self.logger.log(
                    episode=i,
                    phase=name,
                    scenario_id=metrics.scenario_id,
                    difficulty=metrics.difficulty,
                    score=metrics.overall_score,
                    delivery_success_rate=metrics.delivery_success_rate,
                    deliveries_completed=metrics.deliveries_completed,
                    deliveries_failed=metrics.deliveries_failed,
                    peak_congestion=metrics.peak_congestion,
                    accidents_peak=metrics.accidents_peak_concurrent,
                )
        return result

    def compare(
        self,
        *,
        baseline_bundle: PolicyBundle | None = None,
        trained_bundle: PolicyBundle | None = None,
    ) -> ComparisonResult:
        baseline = self.evaluate(name="baseline", policy_bundle=baseline_bundle)
        trained = self.evaluate(name="trained", policy_bundle=trained_bundle)
        return ComparisonResult(
            baseline=baseline,
            trained=trained,
            delta_score=trained.mean_score - baseline.mean_score,
            delta_success_rate=trained.mean_success_rate - baseline.mean_success_rate,
        )

    # ----- internals ------------------------------------------------------

    def _seeds(self) -> list[int]:
        if self.config.seeds is not None:
            return list(self.config.seeds[: self.config.n_episodes])
        return [self.config.base_seed + i for i in range(self.config.n_episodes)]

    def _run_one(self, *, seed: int, policy_bundle: PolicyBundle | None) -> EpisodeMetrics:
        env = self.env_factory(seed)
        agents = self.agents_factory()
        if policy_bundle is not None:
            policy_bundle.install(agents)
        coord = MultiAgentCoordinator(
            env, agents,
            CoordinatorConfig(
                seed=seed,
                delivery_spawn_rate=self.config.delivery_spawn_rate,
                incident_spawn_rate=self.config.incident_spawn_rate,
            ),
        )
        scenario = self._generator.generate(self.config.fixed_difficulty, seed=seed)
        runner = EpisodeRunner(
            coord, scenario,
            EpisodeRunnerConfig(max_ticks=self.config.max_ticks_per_episode),
        )
        return runner.run()

    def _default_env_factory(self, seed: int) -> CityNexusEnv:
        return CityNexusEnv(EnvConfig(
            width=self.config.grid_size[0],
            height=self.config.grid_size[1],
            seed=seed,
            max_ticks=self.config.max_ticks_per_episode,
        ))
