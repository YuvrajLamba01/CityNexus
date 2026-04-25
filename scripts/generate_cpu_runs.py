"""Generate the CPU-only submission artifacts the README embeds.

Mirrors notebook Sections 3-5 verbatim, with `matplotlib` switched to a
non-interactive backend so it works headlessly. Produces:

  * runs/training.jsonl
  * runs/training_curves.png
  * runs/baseline_vs_trained.png

The two GPU-only artifacts (runs/grpo_reward.png, runs/llm_vs_random.png) still
require running notebook Section 6 on a Colab T4 - they are not generated here.

Run with:
    python -m scripts.generate_cpu_runs
or
    python scripts/generate_cpu_runs.py
"""

from __future__ import annotations

import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from citynexus import (  # noqa: E402
    DeliveryAgent,
    EmergencyAgent,
    PlannerAgent,
    PoliceAgent,
    TrafficAgent,
    TrainingConfig,
    TrainingPipeline,
)


def _make_agents():
    return [DeliveryAgent(), TrafficAgent(), EmergencyAgent(), PoliceAgent(), PlannerAgent()]


def main() -> None:
    os.makedirs("runs", exist_ok=True)

    cfg = TrainingConfig(
        n_episodes=40,
        max_ticks_per_episode=80,
        curriculum_target=0.55,
        starting_difficulty=0.20,
        use_memory=True,
        memory_path="runs/memory.json",
        log_dir="runs",
        log_window=5,
        seed=42,
        console=False,
    )
    pipeline = TrainingPipeline(agents_factory=_make_agents, config=cfg)
    summary = pipeline.train()
    print(
        f"training: episodes={summary.n_episodes} "
        f"mean_score={summary.mean_score:.3f} "
        f"last_window_avg_score={summary.last_window_avg_score:.3f} "
        f"mean_delivery_success={summary.mean_delivery_success:.3f} "
        f"last_window_avg_success={summary.last_window_avg_success:.3f} "
        f"final_difficulty={summary.final_difficulty:.3f}"
    )

    records = pipeline.logger.all()
    ep = [r["episode"] for r in records]
    score = [r["score"] for r in records]
    succ = [r["delivery_success_rate"] for r in records]
    diff = [r["difficulty"] for r in records]
    summed_r = [r["summed_reward"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(ep, score, label="score")
    if len(score) >= 5:
        rolling = np.convolve(score, np.ones(5) / 5, mode="valid")
        axes[0, 0].plot(ep[4:], rolling, label="rolling-5")
    axes[0, 0].set_title("Episode score")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, diff, color="tab:red")
    axes[0, 1].set_title("Curriculum difficulty")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("difficulty")

    axes[1, 0].plot(ep, summed_r, color="tab:green")
    axes[1, 0].set_title("Summed reward")
    axes[1, 0].set_xlabel("episode")
    axes[1, 0].set_ylabel("reward")

    axes[1, 1].plot(ep, succ, color="tab:purple")
    axes[1, 1].set_title("Delivery success")
    axes[1, 1].set_xlabel("episode")
    axes[1, 1].set_ylabel("success rate")

    plt.tight_layout()
    plt.savefig("runs/training_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved runs/training_curves.png")

    train_jsonl = pathlib.Path("runs/training.jsonl")
    rows = [
        json.loads(x)
        for x in train_jsonl.read_text(encoding="utf-8").strip().splitlines()
        if x.strip()
    ]
    base_score = rows[0]["score"] if rows else 0.0
    trained_score = (
        sum(r["score"] for r in rows[-5:]) / max(1, len(rows[-5:])) if rows else 0.0
    )
    base_succ = rows[0]["delivery_success_rate"] if rows else 0.0
    trained_succ = (
        sum(r["delivery_success_rate"] for r in rows[-5:]) / max(1, len(rows[-5:]))
        if rows
        else 0.0
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["baseline", "trained"], [base_score, trained_score])
    ax[0].set_title("Score (baseline = ep0, trained = mean of last 5 eps)")
    ax[1].bar(["baseline", "trained"], [base_succ, trained_succ])
    ax[1].set_title("Delivery success")
    for a in ax:
        a.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("runs/baseline_vs_trained.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved runs/baseline_vs_trained.png")
    print(
        json.dumps(
            {
                "baseline_score": base_score,
                "trained_score": trained_score,
                "baseline_success": base_succ,
                "trained_success": trained_succ,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
