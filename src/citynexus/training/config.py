"""Training and evaluation configs."""

from __future__ import annotations

from dataclasses import dataclass

from citynexus.rewards.schemas import GatingMode


@dataclass
class TrainingConfig:
    # Episode loop
    n_episodes: int = 30
    max_ticks_per_episode: int = 100
    grid_size: tuple[int, int] = (20, 20)

    # Curriculum (adversarial scenario generator + difficulty controller)
    use_curriculum: bool = True
    curriculum_target: float = 0.55
    curriculum_alpha: float = 0.18
    starting_difficulty: float = 0.20
    bias_top_k_modes: int = 3

    # Persistent memory
    use_memory: bool = True
    memory_path: str | None = None     # if set, autoload + autosave
    memory_prune_every: int = 10       # prune low-confidence records every N episodes

    # Verifier + reward
    use_verifier: bool = True
    gating_mode: GatingMode = GatingMode.ATTRIBUTED

    # Coordinator
    delivery_spawn_rate: float = 0.30
    incident_spawn_rate: float = 0.10

    # Logging
    log_dir: str | None = None         # if set, write training.jsonl
    console: bool = True
    log_window: int = 10               # rolling-window for console summary

    # RNG
    seed: int = 0


@dataclass
class EvalConfig:
    n_episodes: int = 10
    max_ticks_per_episode: int = 100
    seeds: list[int] | None = None     # None → derive from base_seed
    base_seed: int = 1000
    fixed_difficulty: float = 0.50     # eval at a stable difficulty for fair comparison
    grid_size: tuple[int, int] = (20, 20)
    delivery_spawn_rate: float = 0.30
    incident_spawn_rate: float = 0.10
