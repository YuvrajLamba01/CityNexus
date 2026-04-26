"""CPU-only regenerator for the in-browser Trained Model Playback rollouts.

Runs the OpenEnv wrapper deterministically for two policies — uniform-random
mode picker and the heuristic expert (`expert_mode`) — over ten held-out
seeds and records per-tick mode choices in:

* `runs/llm_rollouts.json`         (canonical artifact)
* `web/data/llm_rollouts.json`     (mirrored for the static demo Space)

Notebook section 6c populates a third track (`trained_llm`) after GRPO
finishes on a Colab T4. The `random_baseline` and `heuristic_expert` tracks
are pure CPU and cheap to regenerate from this script — useful when the
JSON is missing locally, or when changing seed ranges / max_ticks.

Usage:
    python gen_rollouts.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from server.environment import CityNexusEnvironment
from server.models import CityAction
from citynexus.training.llm_planner import MODES, expert_mode


SEEDS = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009]
MAX_TICKS = 80
DIFFICULTY = 0.55  # match the curriculum target the GRPO run was evaluated under


def obs_dict(obs) -> dict:
    """CityObservation -> dict suitable for expert_mode."""
    return {
        "tick": obs.tick,
        "weather": obs.weather,
        "congestion_ratio": float(obs.congestion_ratio),
        "n_active_accidents": int(obs.n_active_accidents),
        "n_active_incidents": int(obs.n_active_incidents),
        "n_open_deliveries": int(obs.n_open_deliveries),
    }


def rollout(env: CityNexusEnvironment, seed: int, mode_picker) -> dict:
    """Run one episode, return modes per tick + final cumulative reward."""
    obs = env.reset(seed=seed)
    cum = 0.0
    modes: list[str] = []
    rng = random.Random(seed)
    for _ in range(MAX_TICKS):
        mode = mode_picker(obs_dict(obs), rng)
        modes.append(mode)
        obs = env.step(CityAction(mode=mode))
        cum += float(getattr(obs, "reward", 0.0) or 0.0)
        if getattr(obs, "done", False):
            break
    while len(modes) < MAX_TICKS:
        modes.append("normal")
    return {"modes": modes, "cumulative_reward": round(cum, 4)}


def main() -> None:
    env = CityNexusEnvironment(max_ticks=MAX_TICKS)

    def random_picker(_obs, rng):
        return rng.choice(MODES)

    def expert_picker(obs, _rng):
        return expert_mode(obs)

    tracks = {
        "random_baseline": {
            "label": "Random mode every tick",
            "description": (
                "Uniform random over the four modes per tick. The cumulative "
                "reward shown here is the JS sim re-evaluation (it differs from "
                "the Python eval in runs/training.jsonl because RNG and physics "
                "are independent implementations)."
            ),
            "available": True,
            "rollouts": {},
        },
        "heuristic_expert": {
            "label": "Heuristic expert (the GRPO training target)",
            "description": (
                "expert_mode() from src/citynexus/training/llm_planner.py. "
                "GRPO trains the Qwen-0.5B Planner to imitate this policy on "
                "the format and correctness reward components."
            ),
            "available": True,
            "rollouts": {},
        },
        "trained_llm": {
            "label": "GRPO-trained Qwen-2.5-0.5B Planner",
            "description": (
                "Per-tick mode chosen by the GRPO-trained LLM during evaluation. "
                "Run notebook section 6f on a Colab T4 to populate this track; "
                "the random + heuristic tracks above ship with the repo."
            ),
            "available": False,
            "note": "Pending: re-run notebook section 6f after Section 6 finishes.",
            "rollouts": {},
        },
    }

    t0 = time.time()
    for seed in SEEDS:
        print(f"  seed {seed}...", end="", flush=True)
        tracks["random_baseline"]["rollouts"][str(seed)] = rollout(env, seed, random_picker)
        tracks["heuristic_expert"]["rollouts"][str(seed)] = rollout(env, seed, expert_picker)
        print(" done")
    print(f"finished in {time.time() - t0:.1f}s")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "max_ticks": MAX_TICKS,
        "difficulty_eval": DIFFICULTY,
        "seeds": SEEDS,
        "modes": list(MODES),
        "tracks": tracks,
    }

    runs_path = Path("runs/llm_rollouts.json")
    web_path = Path("web/data/llm_rollouts.json")
    web_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    web_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {runs_path} ({runs_path.stat().st_size // 1024} KB)")
    print(f"wrote {web_path}  (mirror)")


if __name__ == "__main__":
    main()
