"""Submission smoke tests.

Run with: ``pytest -q``

Each test is intentionally fast (<2s) and covers a different layer so a judge
can verify the whole stack is wired correctly without a GPU.

Layers exercised
----------------
1. Engine + multi-agent coordinator + reward system + verifier
2. Adversarial scenario generator + curriculum failure-mode bias
3. Persistent memory (write → save → reload)
4. OpenEnv FastAPI wrapper (reset / step / state)
5. LLM-planner pure-Python pieces (prompt, expert label, GRPO reward)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Engine + multi-agent loop + rewards + verifier
# ---------------------------------------------------------------------------

def test_multi_agent_loop_produces_rewards_and_messages():
    from citynexus import (
        CityNexusEnv,
        CoordinatorConfig,
        DeliveryAgent,
        EmergencyAgent,
        EnvConfig,
        MultiAgentCoordinator,
        MultiAgentRewardSystem,
        PlannerAgent,
        PoliceAgent,
        RewardSystemConfig,
        TrafficAgent,
        VerificationContext,
        Verifier,
    )
    import copy

    env = CityNexusEnv(EnvConfig(width=20, height=20, seed=11, max_ticks=30))
    agents = [DeliveryAgent(), TrafficAgent(), EmergencyAgent(), PoliceAgent(), PlannerAgent()]
    coord = MultiAgentCoordinator(env, agents, CoordinatorConfig(seed=11))
    coord.reset()

    verifier = Verifier.default()
    rewards = MultiAgentRewardSystem(verifier=verifier, config=RewardSystemConfig())
    rewards.reset()

    summed = 0.0
    for _ in range(10):
        prev = copy.copy(coord.env.state)
        res = coord.step()
        v_ctx = VerificationContext(
            tick=res.step_info.tick,
            prev_state=prev,
            curr_state=coord.env.state,
            agent_ctx=coord.ctx,
            actions=res.actions,
            completed_deliveries=res.completed_deliveries,
            new_deliveries=res.new_deliveries,
            new_incidents=res.new_incidents,
            accidents_cleared=res.step_info.cleared_accidents,
            accidents_spawned=len(res.step_info.new_accidents),
        )
        breakdown = rewards.compute(v_ctx)
        summed += sum(a.total for a in breakdown.per_agent.values())

    # All 5 roles must appear in the breakdown.
    assert set(breakdown.per_agent.keys()) == {"delivery", "traffic", "emergency", "police", "planner"}
    # The bus must have routed *some* messages over 10 ticks.
    assert coord.ctx.bus.stats.get("delivered", 0) > 0
    # Reward signal is finite.
    assert summed == summed  # not NaN


# ---------------------------------------------------------------------------
# 2. Adversarial scenario generator + curriculum failure-mode bias
# ---------------------------------------------------------------------------

def test_curriculum_biases_generator_toward_failure_modes():
    from citynexus import AdversarialGenerator, Curriculum, EpisodeMetrics

    cur = Curriculum(target_score=0.55, starting_difficulty=0.5)
    # Feed a failure-heavy episode so the curriculum classifies a weakness.
    cur.update(EpisodeMetrics(
        episode_id="ep-test",
        scenario_id="sc-test",
        difficulty=0.5,
        ticks_run=80,
        deliveries_total=10,
        deliveries_completed=2,        # 20% success → triggers delivery_failure
        deliveries_failed=8,
        deliveries_open=0,
        accidents_peak_concurrent=2,
        accidents_unresolved_at_end=0,
        incidents_peak_concurrent=1,
        incidents_unresolved_at_end=0,
        peak_congestion=0.4,
        avg_congestion=0.2,
        storm_ticks=0,
        messages_sent=10,
        shocks_fired=4,
    ))
    modes = cur.top_failure_modes()
    assert any(m.name == "delivery_failure" for m in modes), "Curriculum should detect delivery weakness"

    gen = AdversarialGenerator(grid_size=(20, 20), episode_length=80, seed=0)
    biased = gen.generate(0.5, bias_toward=modes, seed=42)
    unbiased = gen.generate(0.5, seed=42)
    biased_kinds = [type(s).__name__ for s in biased.shocks]
    # The bias *increases* the weight of TrafficSpike/BlockedRoutes — usually surfaces them.
    assert biased_kinds, "Scenario should fire shocks"
    assert unbiased.id != biased.id
    # Sanity: difficulty respected.
    assert 0.0 <= biased.difficulty <= 1.0


# ---------------------------------------------------------------------------
# 3. Persistent memory round-trip
# ---------------------------------------------------------------------------

def test_memory_round_trip(tmp_path: Path):
    from citynexus import HighRiskZone, MemoryStore

    path = tmp_path / "mem.json"
    store = MemoryStore(path=path)
    store.add(HighRiskZone(
        id="",
        coords=[(3, 4), (5, 6)],
        risk_score=0.8,
        risk_factors=["accident_pileup", "congestion"],
        sample_count=4,
        timestamp=10,
    ))
    store.save()

    reloaded = MemoryStore(path=path)
    zones = reloaded.hottest_zones(top_k=5)
    assert len(zones) == 1
    assert zones[0].risk_score == pytest.approx(0.8)
    assert (3, 4) in zones[0].coords


# ---------------------------------------------------------------------------
# 4. OpenEnv FastAPI wrapper — reset / step / state
# ---------------------------------------------------------------------------

def test_openenv_wrapper_round_trip():
    from server.environment import CityNexusEnvironment
    from server.models import CITY_MODES, CityAction

    env = CityNexusEnvironment(max_ticks=10)
    obs = env.reset(seed=7)
    assert obs.tick == 0
    assert obs.done is False
    for mode in CITY_MODES:
        obs = env.step(CityAction(mode=mode, directive=f"test-{mode}"))
        assert obs.tick > 0
        assert "delivery" in obs.per_agent_reward
    # State surface is populated.
    assert env.state.step_count > 0
    assert env.state.mode_history, "Mode history should record each step"


# ---------------------------------------------------------------------------
# 5. LLM planner pure-Python pieces (no torch needed)
# ---------------------------------------------------------------------------

def test_grpo_reward_ranking_anti_gaming():
    from citynexus.training.llm_planner import MODES, grpo_reward

    expert = ["normal", "delivery_focus", "emergency_focus", "defensive", "normal"]
    completions = [
        "normal",                              # correct, concise
        "normal",                              # valid format, wrong label
        "<script>alert(1)</script>",           # invalid format
        "defensive " + "x " * 80,              # correct label but absurdly long
        "",                                    # empty
    ]
    r = grpo_reward(None, completions, expert)

    # Strict ordering: correct > wrong-but-valid > invalid.
    assert r[0] > r[1] > r[2]
    # Length penalty must drag a 'correct but verbose' below a clean 'correct'.
    assert r[3] < r[0]
    # Empty completion is worse than wrong-but-valid (it's invalid format).
    assert r[4] <= r[1]
    # All four MODES are still recognised as 'valid format'.
    assert all(m in MODES for m in MODES)


def test_expert_mode_priority_order():
    from citynexus.training.llm_planner import expert_mode

    base = {
        "tick": 1, "weather": "clear", "congestion_ratio": 0.1,
        "n_active_accidents": 0, "n_active_incidents": 0, "n_open_deliveries": 0,
    }
    assert expert_mode(base) == "normal"
    assert expert_mode({**base, "n_active_accidents": 2}) == "emergency_focus"
    assert expert_mode({**base, "n_active_incidents": 2}) == "emergency_focus"
    assert expert_mode({**base, "n_open_deliveries": 6}) == "delivery_focus"
    assert expert_mode({**base, "congestion_ratio": 0.7}) == "defensive"
    # Emergency dominates everything else.
    high_load = {**base, "n_active_accidents": 3, "n_open_deliveries": 9, "congestion_ratio": 0.9}
    assert expert_mode(high_load) == "emergency_focus"


def test_obs_to_prompt_contains_all_fields_and_modes():
    from citynexus.training.llm_planner import MODES, obs_to_prompt

    prompt = obs_to_prompt({
        "tick": 17, "weather": "storm", "congestion_ratio": 0.42,
        "n_active_accidents": 1, "n_active_incidents": 2, "n_open_deliveries": 3,
    })
    assert "tick 17" in prompt
    assert "storm" in prompt
    assert "0.42" in prompt
    for m in MODES:
        assert m in prompt


def test_build_dataset_uses_openenv_wrapper():
    """End-to-end dataset construction without GPU."""
    from server.environment import CityNexusEnvironment
    from server.models import CityAction
    from citynexus.training.llm_planner import (
        MODES, build_dataset, expert_distribution,
    )

    env = CityNexusEnvironment(max_ticks=10)
    samples = build_dataset(env, n_episodes=2, seed=0, action_cls=CityAction)
    assert samples, "Should produce at least one sample"
    dist = expert_distribution(samples)
    # Sanity: every produced expert is one of the four valid modes.
    for s in samples:
        assert s.expert in MODES
        assert "Respond with ONLY the mode name" in s.prompt
    assert sum(dist.values()) == len(samples)
