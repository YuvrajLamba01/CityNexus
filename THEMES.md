# CITYNEXUS — Theme Mapping (Code Evidence)

This document gives judges a fast, line-pointed map from each hackathon theme claim to the exact code that backs it. The repo is organised so that every claim is *verifiable*, not aspirational.

> **TL;DR.** CITYNEXUS is primarily a **Theme #1 (Multi-Agent Interactions)** environment, with substantive coverage of **Theme #4 (Self-Improvement)** via an adversarial scenario generator and adaptive curriculum that co-evolve with the agents' weaknesses, plus moderate coverage of **Theme #3.1 (World Modeling)** via partial-observability + persistent cross-episode memory.

---

## Theme #1 — Multi-Agent Interactions  (PRIMARY)

> *"Cooperation, competition, negotiation, and coalition formation… modeling the beliefs and incentives of others in partially observable settings."*

### 5 distinct roles with their own action spaces
- `src/citynexus/agents/base.py` — `AgentRole` enum (DELIVERY, TRAFFIC, EMERGENCY, POLICE, PLANNER) and `BaseAgent` ABC.
- `src/citynexus/agents/{delivery,traffic,emergency,police,planner}.py` — one role per file, each with its own typed action subclasses.

### True partial observability per role
- `src/citynexus/agents/observability.py` — explicit per-role view filters:
  - `DeliveryView` → roads near pending deliveries (corridor-only)
  - `TrafficView` → intersection cells only
  - `EmergencyView` → discs around active accidents
  - `PoliceView` → discs around incidents + smaller hazard discs
  - `PlannerView` → aggregated metrics only, **no per-cell info**
- Each agent's `decide(...)` consumes only its `View`, never the full `WorldState`.

### Typed inter-agent messaging (the substrate for emergent strategy)
- `src/citynexus/agents/messages.py:70-145` — 9 typed message subclasses:
  - `Advisory`, `RouteBlocked`, `EmergencyPriority`, `SignalChange`,
    `ClearanceRequest`, `Directive`, `StatusReport`, `DispatchNotice`,
    `IncidentReport`.
- `src/citynexus/agents/messages.py:149-184` — `MessageBus` with broadcast + addressed routing, per-tick lifetime, and per-kind statistics.

### Coordination via shared priorities (Planner steers others)
- `src/citynexus/agents/coordinator.py:360-361` — Planner emits `SetPriority(role, value)`; coordinator stores it.
- Each role agent reads `ctx.priorities` to weight its own decisions, producing emergent priority cascades.

### Cooperation incentive baked into rewards
- `src/citynexus/rewards/system.py:176-184` — Planner gets a 15% share of every other role's *positive* reward. This makes coordinating the system a Planner-rewarded objective.
- `src/citynexus/rewards/system.py:191-203` — *Per-agent* + *global* breakdown so credit assignment is explicit.

### Verifier-gated, role-attributed reward
- `src/citynexus/verify/base.py:49-101` — 3-layer verifier (programmatic → system_state → semantic). Each `Check` declares which roles it judges.
- `src/citynexus/rewards/system.py:251-263` — `GatingMode.ATTRIBUTED` zeros only the roles named in failed checks. Exactly the credit-assignment signal MARL needs.

### What this enables
- Agents must model *what other roles know and want* (only the Planner sees aggregates; only Traffic sees intersections; only Delivery has pending packages). The right strategy emerges from messaging, not from shared global state.

---

## Theme #4 — Self-Improvement  (SECONDARY, via adaptive curriculum + adversarial generator)

> *"…learn to generate new challenges, escalate difficulty, and improve through self-play **or adaptive curricula**."*

### Adversarial scenario generator (the "challenger")
- `src/citynexus/scenarios/generator.py:46-185` — `AdversarialGenerator`:
  - Difficulty maps to (count, severity, density) of shocks.
  - Five shock kinds: `TrafficSpike`, `BlockedRoutes`, `EmergencyCluster`, `IncidentSurge`, `WeatherStorm`.
  - Severity, radius, count, and TTL all scale with difficulty.

### Adaptive curriculum (the "challenger's brain")
- `src/citynexus/scenarios/generator.py:190-251` — `Curriculum`:
  - P-controller moves difficulty toward `target_score=0.55` after every episode.
  - Decaying-EMA tracks recurring failure modes (`delivery_failure`, `accident_pileup`, `incident_pileup`, `congestion_overload`).
  - `top_failure_modes()` surfaces the agent's top weaknesses.

### Closed loop: weakness → biased generator
- `src/citynexus/scenarios/generator.py:121-141` — `_kind_weights(...)` re-weights the shock distribution toward the kinds that exploit the surfaced weaknesses. So the *environment* literally adapts to attack the *agent's* weak spots.
- `src/citynexus/training/pipeline.py:142-143` — at every episode boundary, the pipeline calls `generator.generate(curriculum.next_difficulty(), bias_toward=curriculum.top_failure_modes())`.

### What this enables
- An agent that masters delivery routing will start seeing more `TrafficSpike` and `BlockedRoutes` shocks. An agent that masters those will start seeing more `EmergencyCluster` and `IncidentSurge`. The pressure point follows competence — exactly the "adaptive curriculum" Theme #4 describes.

### What this is *not*
- Not classical self-play (no agent vs. past-version-of-itself).
- Not auto-generated proofs / coding tasks.

---

## Theme #3.1 — World Modeling / Professional Tasks  (BONUS, unclaimed in pitch but present in code)

> *"Real interaction with… dynamic systems… maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows."*

### Persistent partially-observable world
- `src/citynexus/env/core.py` + `src/citynexus/env/world_state.py` — stateful traffic dynamics, weather, accidents, roadblocks evolving across ticks.
- Causal feedback chain: accidents → congestion → delivery deadline misses → next-episode curriculum bias toward `TrafficSpike`.

### Cross-episode persistent memory (belief updates)
- `src/citynexus/memory/store.py:26-237` — `MemoryStore`: JSON-backed, schema-versioned, with confidence decay and spatial query.
- `src/citynexus/memory/schemas.py` — three record kinds:
  - `PastFailure` — what went wrong and where
  - `SuccessfulStrategy` — what worked and when
  - `HighRiskZone` — accumulating spatial risk hotspots
- `src/citynexus/memory/writer.py` — observes verifier reports and reward breakdowns each tick to update beliefs (`MemoryWriter.observe_tick`).

### Stateful reward system
- `src/citynexus/rewards/system.py:82-89, 195-203` — explicit `_PrevSnapshot` tracks prev unit positions, prev priorities, prev incident assignments across ticks for process-aware components (progress, anticipation, redundant-dispatch).

### What this enables
- Agents that *remember* across episodes route around hotspots they've seen fail, anticipate spikes in zones they've seen overload, and don't re-issue dispatches they know never resolved. This is durable internal world modeling, not just per-episode reactivity.

---

## Theme #2 — Long-Horizon Planning  (NOT CLAIMED)

Episodes are 80–100 ticks. Persistent memory does survive across episodes, which is the only meaningful long-horizon hook. We do not claim this theme.

## Theme #3.2 — Personalized Tasks  (NOT APPLICABLE)
## Theme #5 — Wild Card  (NOT APPLICABLE)

---

## How to verify the theme mapping yourself

```bash
# Smoke tests (3 seconds, CPU only) — exercise multi-agent loop, curriculum,
# memory round-trip, OpenEnv wrapper, and the GRPO reward function.
pip install -e . pytest
pytest -q tests/test_smoke.py
```

```bash
# Run the OpenEnv server locally and curl /metadata.
uvicorn server.app:app --port 8000 &
curl localhost:8000/metadata | python -m json.tool
```

The training notebook (`notebooks/train_citynexus_colab.ipynb`) closes the loop:
the GRPO reward function it trains on (`citynexus.training.llm_planner.grpo_reward`)
is the same one the smoke tests verify, and the same one used during the
trained-vs-random rollout comparison in Section 6.
