"""AdversarialGenerator + Curriculum.

The generator turns a difficulty score into a concrete `Scenario` (with a
schedule of shocks). The curriculum tracks performance across episodes and
adjusts both the next difficulty and a *failure-mode bias* — a list of
weaknesses the next scenario should target.

Together they form an adaptive loop:

    scenario = generator.generate(curriculum.next_difficulty(),
                                  bias_toward=curriculum.top_failure_modes())
    metrics  = run_episode(scenario)
    curriculum.update(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from citynexus.env.events import Weather
from citynexus.scenarios.schemas import EpisodeMetrics, FailureMode, Scenario
from citynexus.scenarios.shocks import (
    BlockedRoutes,
    EmergencyCluster,
    IncidentSurge,
    Shock,
    ShockKind,
    TrafficSpike,
    WeatherStorm,
)


# Shock kinds the generator can emit (excludes NOOP).
_ALL_KINDS: tuple[str, ...] = (
    ShockKind.TRAFFIC_SPIKE.value,
    ShockKind.EMERGENCY_CLUSTER.value,
    ShockKind.BLOCKED_ROUTES.value,
    ShockKind.WEATHER_STORM.value,
    ShockKind.INCIDENT_SURGE.value,
)


# --- Generator -------------------------------------------------------------

class AdversarialGenerator:
    """Parametric scenario generator. Difficulty maps to (count, severity, density)
    of shocks; bias re-weights the kind distribution toward known weaknesses."""

    def __init__(
        self,
        *,
        grid_size: tuple[int, int] = (20, 20),
        episode_length: int = 100,
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.episode_length = episode_length
        self._rng = Random(seed)
        self._next_id = 0

    def generate(
        self,
        difficulty: float,
        *,
        bias_toward: list[FailureMode] | None = None,
        seed: int | None = None,
    ) -> Scenario:
        d = max(0.0, min(1.0, difficulty))
        scenario_seed = seed if seed is not None else self._rng.randrange(1 << 30)
        rng = Random(scenario_seed)
        sid = self._gen_id()

        n_shocks = int(round(2 + d * 8))                    # 2 → 10
        warmup = max(2, int(self.episode_length * 0.05))
        # Shocks fire across [warmup, 0.85 * episode_length]
        latest_tick = max(warmup + 1, int(self.episode_length * 0.85))

        kind_weights = self._kind_weights(d, bias_toward)
        kinds = list(kind_weights.keys())
        weights = list(kind_weights.values())

        shocks: list[Shock] = []
        for _ in range(n_shocks):
            kind = rng.choices(kinds, weights=weights, k=1)[0]
            tick = rng.randint(warmup, latest_tick)
            shocks.append(self._build_shock(kind, tick, d, rng))

        # Sort shocks by trigger tick for predictable iteration.
        shocks.sort(key=lambda s: s.trigger_tick)

        # Initial weather: storm probability scales with difficulty.
        if rng.random() < 0.4 * d:
            init_weather = Weather.STORM
        elif rng.random() < 0.3 + 0.3 * d:
            init_weather = Weather.RAIN
        else:
            init_weather = Weather.CLEAR

        meta = {
            "generator": "AdversarialGenerator",
            "n_shocks": len(shocks),
            "kind_weights": {k: round(v, 3) for k, v in kind_weights.items()},
            "biased_toward": [m.name for m in (bias_toward or [])],
        }
        return Scenario(
            id=sid,
            seed=scenario_seed,
            difficulty=d,
            initial_weather=init_weather,
            shocks=shocks,
            metadata=meta,
        )

    # --- internals --------------------------------------------------------

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"sc-{self._next_id:05d}"

    def _kind_weights(
        self,
        difficulty: float,
        bias_toward: list[FailureMode] | None,
    ) -> dict[str, float]:
        # Base distribution: gently shifts toward more severe shock types as difficulty rises.
        base = {
            ShockKind.TRAFFIC_SPIKE.value:     1.0 + 0.5 * difficulty,
            ShockKind.BLOCKED_ROUTES.value:    1.0 + 0.4 * difficulty,
            ShockKind.EMERGENCY_CLUSTER.value: 0.7 + 1.0 * difficulty,
            ShockKind.INCIDENT_SURGE.value:    0.6 + 0.8 * difficulty,
            ShockKind.WEATHER_STORM.value:     0.4 + 0.6 * difficulty,
        }
        # Bias: each suggested kind gets a multiplicative boost proportional to mode severity.
        if bias_toward:
            for mode in bias_toward:
                boost = 1.0 + 1.5 * mode.severity
                for kind in mode.suggested_shock_kinds:
                    if kind in base:
                        base[kind] *= boost
        return base

    def _build_shock(self, kind: str, tick: int, difficulty: float, rng: Random) -> Shock:
        W, H = self.grid_size
        if kind == ShockKind.TRAFFIC_SPIKE.value:
            return TrafficSpike(
                trigger_tick=tick,
                center=(rng.randrange(W), rng.randrange(H)),
                radius=int(2 + difficulty * 3),                # 2 → 5
                magnitude=0.4 + 0.5 * difficulty,              # 0.4 → 0.9
            )
        if kind == ShockKind.EMERGENCY_CLUSTER.value:
            severity = 1 + int(round(difficulty * 2))           # 1 → 3
            return EmergencyCluster(
                trigger_tick=tick,
                center=(rng.randrange(W), rng.randrange(H)),
                radius=int(2 + difficulty * 3),
                count=int(2 + difficulty * 4),                  # 2 → 6
                severity=severity,
            )
        if kind == ShockKind.BLOCKED_ROUTES.value:
            n_blocks = int(2 + difficulty * 6)                  # 2 → 8
            coords = [
                (rng.randrange(W), rng.randrange(H))
                for _ in range(n_blocks)
            ]
            return BlockedRoutes(
                trigger_tick=tick,
                coords=coords,
                ttl=int(8 + difficulty * 12),                   # 8 → 20
            )
        if kind == ShockKind.WEATHER_STORM.value:
            return WeatherStorm(
                trigger_tick=tick,
                force_to="storm" if difficulty > 0.5 else "rain",
                duration=int(5 + difficulty * 15),              # 5 → 20
            )
        if kind == ShockKind.INCIDENT_SURGE.value:
            return IncidentSurge(
                trigger_tick=tick,
                count=int(2 + difficulty * 4),                  # 2 → 6
                severity=1 + int(round(difficulty * 2)),
            )
        # Fallback — shouldn't hit.
        return Shock(trigger_tick=tick)


# --- Curriculum ------------------------------------------------------------

class Curriculum:
    """Adaptive difficulty controller.

    `update(metrics)` runs after each episode. Difficulty is moved toward the
    region where the agent scores `target_score` (proportional control). A
    decaying EMA tracks which failure modes recur, surfaced via `top_failure_modes`.
    """

    def __init__(
        self,
        *,
        target_score: float = 0.55,
        alpha: float = 0.18,
        starting_difficulty: float = 0.20,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
        history_window: int = 8,
        ema_decay: float = 0.6,
    ) -> None:
        self.target_score = target_score
        self.alpha = alpha
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.history_window = history_window
        self.ema_decay = ema_decay
        self.difficulty = starting_difficulty
        self.history: list[EpisodeMetrics] = []
        self._failure_ema: dict[str, float] = {}

    def update(self, metrics: EpisodeMetrics) -> None:
        # Record
        self.history.append(metrics)
        if len(self.history) > self.history_window:
            self.history = self.history[-self.history_window:]

        # Difficulty: P-controller on (score - target).
        gap = metrics.overall_score - self.target_score
        new_d = self.difficulty + self.alpha * gap
        self.difficulty = max(self.min_difficulty, min(self.max_difficulty, new_d))

        # Failure mode EMA: decay then add.
        for k in list(self._failure_ema.keys()):
            self._failure_ema[k] *= self.ema_decay
            if self._failure_ema[k] < 0.05:
                del self._failure_ema[k]
        for mode in self._classify_failures(metrics):
            self._failure_ema[mode.name] = (
                self._failure_ema.get(mode.name, 0.0)
                + (1.0 - self.ema_decay) * mode.severity
            )

    def next_difficulty(self) -> float:
        return self.difficulty

    def top_failure_modes(self, n: int = 3, threshold: float = 0.10) -> list[FailureMode]:
        ranked = sorted(self._failure_ema.items(), key=lambda kv: -kv[1])
        modes: list[FailureMode] = []
        for name, severity in ranked[:n]:
            if severity < threshold:
                continue
            modes.append(self._mode_from_name(name, severity))
        return modes

    def stats(self) -> dict:
        recent_scores = [m.overall_score for m in self.history[-self.history_window:]]
        return {
            "difficulty": round(self.difficulty, 3),
            "target_score": self.target_score,
            "n_episodes": len(self.history),
            "recent_avg_score": round(sum(recent_scores) / len(recent_scores), 3) if recent_scores else None,
            "failure_modes_ema": {k: round(v, 3) for k, v in self._failure_ema.items()},
        }

    # --- failure classification ------------------------------------------

    @staticmethod
    def _classify_failures(m: EpisodeMetrics) -> list[FailureMode]:
        modes: list[FailureMode] = []
        if m.delivery_success_rate < 0.50:
            modes.append(FailureMode(
                name="delivery_failure",
                severity=min(1.0, 1.0 - m.delivery_success_rate),
                description="Delivery success rate dropped below 50%.",
                suggested_shock_kinds=[
                    ShockKind.TRAFFIC_SPIKE.value,
                    ShockKind.BLOCKED_ROUTES.value,
                ],
            ))
        if m.accidents_unresolved_at_end >= 3 or m.accidents_peak_concurrent >= 6:
            modes.append(FailureMode(
                name="accident_pileup",
                severity=min(1.0, max(
                    m.accidents_unresolved_at_end / 8.0,
                    m.accidents_peak_concurrent / 10.0,
                )),
                description="Accidents accumulated beyond emergency capacity.",
                suggested_shock_kinds=[
                    ShockKind.EMERGENCY_CLUSTER.value,
                    ShockKind.WEATHER_STORM.value,
                ],
            ))
        if m.incidents_unresolved_at_end >= 2 or m.incidents_peak_concurrent >= 4:
            modes.append(FailureMode(
                name="incident_pileup",
                severity=min(1.0, max(
                    m.incidents_unresolved_at_end / 5.0,
                    m.incidents_peak_concurrent / 6.0,
                )),
                description="Police incidents outpaced response.",
                suggested_shock_kinds=[ShockKind.INCIDENT_SURGE.value],
            ))
        if m.peak_congestion > 0.75:
            modes.append(FailureMode(
                name="congestion_overload",
                severity=min(1.0, m.peak_congestion),
                description="Peak congestion exceeded 75%.",
                suggested_shock_kinds=[
                    ShockKind.TRAFFIC_SPIKE.value,
                    ShockKind.BLOCKED_ROUTES.value,
                ],
            ))
        return modes

    @staticmethod
    def _mode_from_name(name: str, severity: float) -> FailureMode:
        # Mirror the suggested shock kinds from the classifier.
        suggestions = {
            "delivery_failure": [
                ShockKind.TRAFFIC_SPIKE.value, ShockKind.BLOCKED_ROUTES.value,
            ],
            "accident_pileup": [
                ShockKind.EMERGENCY_CLUSTER.value, ShockKind.WEATHER_STORM.value,
            ],
            "incident_pileup": [ShockKind.INCIDENT_SURGE.value],
            "congestion_overload": [
                ShockKind.TRAFFIC_SPIKE.value, ShockKind.BLOCKED_ROUTES.value,
            ],
        }
        return FailureMode(
            name=name,
            severity=severity,
            description=f"Recurring weakness: {name}",
            suggested_shock_kinds=suggestions.get(name, []),
        )
