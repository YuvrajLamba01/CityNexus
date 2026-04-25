"""EpisodeRunner — runs one episode end-to-end and emits EpisodeMetrics."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from citynexus.agents.coordinator import MultiAgentCoordinator
from citynexus.env.events import Weather
from citynexus.scenarios.schemas import EpisodeMetrics, Scenario
from citynexus.scenarios.shocks import ShockKind, ShockReport, apply_shock


@dataclass
class EpisodeRunnerConfig:
    max_ticks: int = 100
    episode_id: str | None = None       # autogen if None


class EpisodeRunner:
    """Drives one episode using a Scenario applied to an existing coordinator.

    Lifecycle:
        runner = EpisodeRunner(coordinator, scenario)
        metrics = runner.run()
    """

    _next_episode_id: int = 0

    def __init__(
        self,
        coordinator: MultiAgentCoordinator,
        scenario: Scenario,
        config: EpisodeRunnerConfig | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.scenario = scenario
        self.config = config or EpisodeRunnerConfig()
        self._rng = Random(scenario.seed)
        # Active weather lock from a WeatherStorm shock: (until_tick, weather).
        self._weather_lock: tuple[int, Weather] | None = None

    def run(self) -> EpisodeMetrics:
        eid = self.config.episode_id or self._auto_episode_id()

        # Reset the coordinator and apply scenario's initial overrides.
        self.coordinator.reset()
        self.coordinator.env.state.weather = self.scenario.initial_weather

        # Per-tick accumulators.
        peak_cong = 0.0
        sum_cong = 0.0
        storm_ticks = 0
        accidents_peak = 0
        incidents_peak = 0
        shocks_fired = 0

        for _ in range(self.config.max_ticks):
            current_tick = self.coordinator.ctx.tick

            # 1. Fire any shocks scheduled for this tick.
            for shock in self.scenario.shocks_at(current_tick):
                report: ShockReport = apply_shock(
                    self.coordinator.env, self.coordinator.ctx, shock, self._rng,
                )
                shocks_fired += 1
                if report.weather_lock_until is not None:
                    target = self.coordinator.env.state.weather
                    self._weather_lock = (report.weather_lock_until, target)

            # 2. Step the multi-agent system one tick.
            self.coordinator.step()

            # 3. Enforce active weather lock (override post-step weather).
            self._enforce_weather_lock()

            # 4. Track per-tick stats.
            world = self.coordinator.env.state
            cong = world.congestion_ratio()
            peak_cong = max(peak_cong, cong)
            sum_cong += cong
            if world.weather == Weather.STORM:
                storm_ticks += 1
            accidents_peak = max(accidents_peak, len(world.accidents))
            incidents_peak = max(incidents_peak, len(self.coordinator.ctx.incidents))

        # Finalize.
        ctx = self.coordinator.ctx
        deliveries = list(ctx.deliveries.values())
        completed = sum(1 for d in deliveries if d.status.value == "delivered")
        failed = sum(1 for d in deliveries if d.status.value == "failed")
        open_d = sum(1 for d in deliveries if d.is_open)

        ticks_run = self.config.max_ticks
        metrics = EpisodeMetrics(
            episode_id=eid,
            scenario_id=self.scenario.id,
            difficulty=self.scenario.difficulty,
            ticks_run=ticks_run,
            deliveries_total=len(deliveries),
            deliveries_completed=completed,
            deliveries_failed=failed,
            deliveries_open=open_d,
            accidents_peak_concurrent=accidents_peak,
            accidents_unresolved_at_end=len(self.coordinator.env.state.accidents),
            incidents_peak_concurrent=incidents_peak,
            incidents_unresolved_at_end=len(ctx.incidents),
            peak_congestion=peak_cong,
            avg_congestion=sum_cong / max(1, ticks_run),
            storm_ticks=storm_ticks,
            messages_sent=ctx.bus.stats.get("sent", 0),
            shocks_fired=shocks_fired,
        )
        return metrics

    def _enforce_weather_lock(self) -> None:
        if self._weather_lock is None:
            return
        until, target = self._weather_lock
        if self.coordinator.ctx.tick >= until:
            self._weather_lock = None
            return
        # Override whatever the env's weather Markov picked.
        self.coordinator.env.state.weather = target

    @classmethod
    def _auto_episode_id(cls) -> str:
        cls._next_episode_id += 1
        return f"ep-{cls._next_episode_id:05d}"
