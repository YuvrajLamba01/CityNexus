---
title: CITYNEXUS
emoji: "🏙️"
colorFrom: indigo
colorTo: purple
sdk: static
app_file: web/index.html
pinned: false
license: mit
short_description: Self-evolving multi-agent urban simulation
---

# CITYNEXUS

**OpenEnv Hackathon 2026 — Theme #1 (Multi-Agent Interactions) + Theme #4 (Self-Improvement)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuvrajLamba01/CityNexus/blob/main/notebooks/train_citynexus_colab.ipynb)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/smoke%20tests-pytest-brightgreen)](tests/test_smoke.py)

A 20×20 grid city run by **5 LLM-trainable agents** (Delivery, Traffic, Emergency, Police, Planner) under an **adversarial curriculum** that adapts to whichever weakness it finds. Verifier-gated, role-attributed rewards plus persistent cross-episode memory make this an end-to-end environment for training multi-agent policies.

> **The pitch in one sentence.** A scenario generator that gets meaner *exactly where the agents are weakest*, gated rewards that only credit the role responsible, and a Planner whose action space is ergonomic enough to drive with a 0.5B-parameter LLM via GRPO.

---

## Why this environment

Most multi-agent environments give you cooperation **or** an adaptive opponent. CITYNEXUS gives you both, in a tightly-coupled loop:

1. **Five roles, five views, one bus.** Each agent only sees what its job requires (Delivery sees its corridors; Traffic sees only intersections; Planner sees only aggregates). They coordinate by sending typed messages on a per-tick bus, not by reading shared state. Coordination is *learned*, not hardcoded.
2. **A challenger that adapts.** An EMA over recent failure modes biases the next scenario generator toward the agents' weakest shock kinds. Master delivery routing → it starts throwing emergency clusters. Master those → weather storms. The pressure point follows competence.
3. **Rewards that don't get gamed.** A 3-layer verifier (programmatic → system-state → semantic) attributes failures to specific roles and zeros only their reward. Composable, not monolithic.
4. **A planner action surface designed for LLMs.** The Planner picks one of four city postures per tick. That tiny action space is what makes GRPO with a 0.5B model tractable on a Colab T4 — and the bigger heuristic agents do the heavy lifting underneath.

Full code-level evidence per theme: see [`THEMES.md`](THEMES.md).

---

## What's in this repo

```
CITYNEXUS/
├── src/citynexus/             # Engine, agents, scenarios, rewards, memory, training
│   ├── env/                   #   grid + traffic + weather + accidents
│   ├── city/                  #   zone taxonomy + procedural generator
│   ├── agents/                #   5 role agents, partial-obs filters, message bus
│   ├── entities/              #   Delivery, ResponderUnit, Incident
│   ├── scenarios/             #   AdversarialGenerator + Curriculum + EpisodeRunner
│   ├── verify/                #   3-layer verifier
│   ├── rewards/               #   per-agent + global rewards, process-aware
│   ├── memory/                #   JSON-persisted store + writer
│   └── training/              #   pipeline + evaluator + LLMPlannerPolicy + GRPO reward
├── server/                    # OpenEnv FastAPI server (HF Space target)
│   ├── app.py
│   ├── environment.py
│   ├── models.py              #   Pydantic CityAction / CityObservation / CityNexusEnvState
│   ├── Dockerfile
│   └── space_README.md        #   Front-matter for the docker SDK Space
├── web/                       # Static playable demo (separate static SDK Space)
├── notebooks/
│   └── train_citynexus_colab.ipynb   # End-to-end: heuristic + GRPO + evaluation
├── tests/test_smoke.py        # 8 fast tests covering every layer
├── openenv.yaml
├── requirements.txt           # CPU-only env deps
├── requirements-train.txt     # GPU GRPO stack
└── THEMES.md                  # Theme-by-theme code evidence
```

---

## Quick start

### 1. Run the smoke tests (3 seconds, CPU)

```bash
pip install -e . pytest
pytest -q
```

Eight tests exercise: multi-agent loop + reward system, curriculum failure-mode bias, memory round-trip, OpenEnv FastAPI wrapper, GRPO reward ranking + anti-gaming, prompt builder, dataset construction.

### 2. Run the OpenEnv server locally

```bash
pip install -e .
uvicorn server.app:app --port 8000
curl localhost:8000/metadata
```

The same image deploys to Hugging Face Spaces via `server/Dockerfile` and `server/space_README.md`.

### 3. Train an LLM Planner via GRPO (Colab T4)

Open the [Colab notebook](notebooks/train_citynexus_colab.ipynb), switch to a T4 runtime for Section 6, and run top-to-bottom. The notebook:

- Verifies the OpenEnv wrapper end-to-end.
- Trains the heuristic baseline with adaptive curriculum + persistent memory + verifier-gated rewards (Section 3).
- Plots reward / difficulty / success-rate curves (Section 4).
- Fine-tunes Qwen-2.5-0.5B with `trl.GRPOTrainer` on the verifiable reward defined in `citynexus.training.llm_planner` (Section 6).
- Compares the trained LLM planner against a random-mode baseline on the same seeds (Section 6e).

### 4. Play it in the browser

The static web demo (`web/index.html`) runs the simulation entirely client-side: live grid, per-agent reward sparklines, decision log, message stream, persistent memory zones in `localStorage`, weather effects, controls for difficulty / seed / episode length.

```bash
cd web && python -m http.server 8080
# → http://localhost:8080
```

---

## The five agents

| Agent | Sees | Decides | Rewarded for |
|-------|------|---------|--------------|
| **Delivery**  | Roads near pending routes (corridor view)        | BFS-route packages around blocked cells          | Completed deliveries, progress toward goal       |
| **Traffic**   | Intersection cells only                           | Place / clear roadblocks; broadcast advisories   | Congestion drops, traffic flow                   |
| **Emergency** | Discs around active accidents                    | Dispatch nearest ambulance by severity           | Scene clearance, response speed                  |
| **Police**    | Discs around incidents + smaller hazard discs    | Dispatch police; cordon protests                 | Incident resolution, crowd safety                |
| **Planner**   | Aggregate metrics only — no per-cell info        | Set per-role priorities; broadcast directives    | Priority coherence, anticipation, system share   |

The Planner gets a 15% share of every other role's positive reward, making whole-system coordination its objective. This is the role the GRPO notebook fine-tunes — its action surface is ergonomic for an LLM (4 modes), and its reward depends on the other 4 agents performing well.

---

## Reward design (and how it resists gaming)

**Per-tick, per-agent reward** = outcome rewards (deliveries completed, accidents cleared, congestion reduction) **+** process-aware rewards (movement progress, dispatch intent, planner anticipation) **−** penalties (delays, collisions, idle units, redundant dispatches), gated by a **3-layer verifier**:

1. **Programmatic checks** — type, range, schema-level invariants.
2. **System-state checks** — the world's response to the action (delivery actually moved? incident actually resolved?).
3. **Semantic checks** — was the choice *appropriate* given role priorities and visible signals?

Failed checks are attributed to specific roles, and `GatingMode.ATTRIBUTED` zeros only those roles' rewards (`src/citynexus/rewards/system.py:251-263`).

**Anti-gaming for the GRPO planner.** The notebook trains on a verifiable reward with three components — format, correctness, length penalty (`src/citynexus/training/llm_planner.py:grpo_reward`). The smoke tests *verify the ordering*: correct-and-concise > correct-and-verbose > wrong-but-valid > invalid-format. The exact same function is used during training and during evaluation.

---

## Self-improvement loop (Theme #4)

```
              ┌─────────────────────────┐
              │  AdversarialGenerator   │ ← bias_toward = top failure modes
              │  (5 shock kinds)        │
              └────────────┬────────────┘
                           │ Scenario
                           ▼
              ┌─────────────────────────┐
              │  EpisodeRunner          │
              │  (5 agents + verifier)  │
              └────────────┬────────────┘
                           │ EpisodeMetrics
                           ▼
              ┌─────────────────────────┐
              │  Curriculum             │
              │  - P-controller on      │
              │    (score - target)     │
              │  - EMA over failure     │
              │    modes                │
              └────────────┬────────────┘
                           │
                           └──→ feeds next iteration
```

After every episode, the curriculum tracks which failure modes recurred (`delivery_failure`, `accident_pileup`, `incident_pileup`, `congestion_overload`) and biases the generator's shock distribution to attack those weaknesses next. The pressure point literally follows competence.

---

## Results & training evidence

Run the [Colab notebook](notebooks/train_citynexus_colab.ipynb) top-to-bottom; it writes the following artifacts to `runs/` and you can `git add -f` them after.

### Heuristic curriculum baseline (CPU, ~5 min)
![Training curves](runs/training_curves.png)
*4-panel plot: episode score with rolling-5 mean, curriculum difficulty, summed reward, delivery success rate.*

### Baseline vs trained rollout (CPU, ~2 min)
![Baseline vs trained](runs/baseline_vs_trained.png)
*Same seeds, same scenarios; isolates policy quality.*

### GRPO LLM planner training (T4 GPU, ~10 min)
![GRPO reward](runs/grpo_reward.png)
*Mean group reward per training step. Reward function defined in `citynexus.training.llm_planner.grpo_reward`.*

### Trained LLM vs random-mode planner (T4 GPU, ~3 min)
![LLM vs random](runs/llm_vs_random.png)
*Cumulative env reward across 10 held-out seeds. Both rollouts go through the OpenEnv wrapper for a fair comparison.*

> **If you don't see images above**, run the notebook and `git add -f runs/*.png runs/training.jsonl` — the `.gitignore` keeps `runs/` out by default but allow-lists these specific filenames.

### Headline numbers
The notebook's final cell prints a copy-paste block that reads from `runs/training.jsonl` and the in-memory GRPO comparison:

```
### Heuristic Pipeline (Sections 3-5)
- Episodes trained:       40
- Mean score (all):       <fill from notebook>
- Last-5 avg score:       <fill from notebook>
- Mean delivery success:  <fill from notebook>
- Last-5 delivery success:<fill from notebook>

### GRPO LLM RL (Section 6)
- Random-mode baseline:   mean=<...>  stdev=<...>
- GRPO-trained LLM:       mean=<...>  stdev=<...>
- Improvement:            <delta>  (<pct>%)
```

---

## OpenEnv compliance checklist

- [x] Inherits `openenv.core.env_server.Environment` (`server/environment.py:111`)
- [x] Standard Gym-style API: `reset(seed, episode_id) → CityObservation`, `step(CityAction) → CityObservation`, `state` property, `close()`
- [x] Pydantic schemas for action / observation / state (`server/models.py`)
- [x] Valid `openenv.yaml` manifest at repo root
- [x] FastAPI server (`server/app.py`) using `openenv.core.env_server.create_app(...)`
- [x] Dockerfile + Space front-matter for HF deployment (`server/Dockerfile`, `server/space_README.md`)
- [x] Smoke tests covering reset / step / state through the FastAPI wrapper (`tests/test_smoke.py::test_openenv_wrapper_round_trip`)
- [x] No reserved MCP tool names — environment uses the FastAPI Environment path, not MCPEnvironment.
- [x] Training script using HF TRL (`trl.GRPOTrainer`) and Unsloth (`unsloth.FastLanguageModel`) in a Colab notebook.

---

## Submission links

| Material                              | Status                              | Link                                                                                                              |
| ------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **OpenEnv server (HF Space, docker)** | ⏳ Deploy after running notebook     | _Add the docker-SDK Space URL here once deployed (use `server/space_README.md` as that Space's README)._         |
| **Static demo (HF Space, static)**    | ⏳ Deploy after running notebook     | _Add the static-SDK Space URL here once deployed (this repo's root README is set up for it)._                   |
| **Training notebook (Colab)**         | ✓ Ready                             | [`notebooks/train_citynexus_colab.ipynb`](notebooks/train_citynexus_colab.ipynb) — Open in Colab via badge above. |
| **Mini-blog or <2-min YouTube demo**  | ⏳ Add link                          | _Drop the URL here once recorded._                                                                                |
| **Theme evidence**                    | ✓ Ready                             | [`THEMES.md`](THEMES.md)                                                                                          |
| **Smoke tests**                       | ✓ 8 passing in 3 s                  | `pytest -q`                                                                                                      |

---

## Architecture deep-dive

For the full per-component breakdown — `MultiAgentCoordinator` tick loop, `MessageBus` semantics, `Curriculum` failure-mode classification, `MemoryStore` decay queries, the `LLMPlannerPolicy` inference path — see [`THEMES.md`](THEMES.md). It's organised by theme, with file/line citations into the source.

---

## License

MIT.
