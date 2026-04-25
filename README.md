---
title: CITYNEXUS
emoji: "\U0001F3D9"
colorFrom: indigo
colorTo: purple
sdk: static
app_file: web/index.html
pinned: false
license: mit
short_description: Self-evolving multi-agent urban simulation
---

# CITYNEXUS

**OpenEnv Hackathon 2026 Submission** — Multi-agent urban simulation with LLM training.

Self-evolving multi-agent urban simulation. Five LLM-trainable agents (Delivery,
Traffic, Emergency, Police, Planner) manage a 20×20 grid city under adversarial
scenarios produced by a curriculum-driven generator. Verifier-gated, process-aware
rewards plus persistent memory make this an end-to-end environment for training
multi-agent policies.

---

## 🎯 Problem & Themes

**Problem:** Train LLMs to coordinate multi-agent urban operations under adversarial scenarios, where agents must cooperate despite partial observability, learn persistent spatial memory, and adapt to a curriculum of increasing difficulty.

**Hackathon Themes:**
- **Theme #1 (Multi-Agent Interactions):** Five roles (Delivery, Traffic, Emergency, Police, Planner) cooperate and compete for resources. Agents model incentives of others via the message bus and priority system.
- **Theme #4 (Self-Improvement):** Adaptive curriculum automatically scales difficulty toward agent competence. High-risk zones persist across episodes, driving memory-based learning.

---

## 📋 Submission Materials

| Material | Status | Link |
|----------|--------|------|
| **OpenEnv Environment (HF Space)** | ✓ Ready | `[ADD URL]` |
| **Training Notebook (Colab)** | ✓ Ready | [View locally: `notebooks/train_citynexus_colab.ipynb`](notebooks/train_citynexus_colab.ipynb) |
| **Demo Video / Blog Post** | ⏳ Add | `[ADD YOUTUBE <2min or HF blog link]` |
| **Training Evidence** | ⏳ Add | Reward curves, baseline-vs-trained comparison (see Results below) |

---

## 📊 Results & Training Evidence

**Status:** Run the training pipeline and commit plots here.

### Reward Curves
**Plot:** `runs/training_curves.png` (4-panel: score, difficulty, summed reward, success rate)
```
[PLACEHOLDER: After running training, copy runs/training_curves.png here]
```

### Baseline vs Trained Comparison  
**Plot:** `runs/baseline_vs_trained.png` (bar chart: random agent vs memory-trained agent)
```
[PLACEHOLDER: After running training, copy runs/baseline_vs_trained.png here]
```

### Key Numbers
- **Episodes trained:** [ADD]
- **Final city score:** [ADD] (composite health, 0–1 scale)
- **Delivery success improvement:** [ADD baseline %] → [ADD trained %]
- **Avg congestion:** [ADD baseline] → [ADD trained]

---

## 🌐 Quick Start

### Play in Browser (No Install)

A pure HTML / CSS / JavaScript canvas frontend that runs the full simulation in
your browser — no Python install, no server, no network.

```bash
# Option 1: double-click
open web/index.html

# Option 2: serve over HTTP
cd web && python -m http.server 8080
# → http://localhost:8080
```

**Features:**
- **Live grid** — zones, traffic overlay, routed units (ambulance / police / delivery van), accidents, roadblocks, incidents
- **Per-agent reward sparklines** + cumulative-reward chart + composite city score
- **Decision log**, **inter-agent message stream**, **memory zones**
- **Controls** — reset, step, play / pause, speed, difficulty, episode length, seed, toggles for persistent memory + weather effects + unit trails
- **`localStorage` memory** — agents query past failures and high-risk hotspots across browser sessions

---

## 📁 Project Layout

```
CITYNEXUS/
├── web/                       # static frontend (deployed here)
│   ├── index.html
│   ├── styles.css
│   └── js/
│       ├── sim.js             # engine + 5 agents + adversary + rewards + memory (JS port)
│       ├── render.js          # canvas drawing + interpolation + charts
│       └── main.js            # control wiring + animation loop
├── src/citynexus/             # full Python implementation
│   ├── env/                   #   grid + traffic dynamics + weather + accidents
│   ├── city/                  #   zone taxonomy + procedural generator
│   ├── agents/                #   5 role agents + base ABC + observability + bus
│   ├── entities/              #   Delivery, ResponderUnit, Incident
│   ├── scenarios/             #   adversarial generator + curriculum + episode runner
│   ├── verify/                #   3-layer verifier
│   ├── rewards/               #   per-agent + global rewards + process-aware components
│   ├── memory/                #   JSON-persisted store + writer
│   └── training/              #   pipeline + evaluator + Policy ABC for RL plug-in
├── server/                    # OpenEnv FastAPI server
│   ├── app.py                 #   entry point for HF Spaces
│   ├── environment.py         #   CityNexusEnvironment (Gym-style wrapper)
│   └── models.py              #   Pydantic action/observation/state schemas
├── notebooks/                 # Training & evaluation notebooks
│   ├── train_citynexus_colab.ipynb   # Full pipeline for Colab
│   └── README.md
├── openenv.yaml               # OpenEnv manifest
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture: The Five Agents

| Agent | Observes | Decides | Rewards For |
|-------|----------|---------|-------------|
| **Delivery** | Roads near pending routes | BFS-route packages around blocked cells | Completed deliveries, progress toward goal |
| **Traffic** | Intersection congestion | Place / clear roadblocks; broadcast advisories | Congestion reduction, traffic flow |
| **Emergency** | Discs around accidents | Dispatch nearest ambulance by severity | Scene clearance, response speed |
| **Police** | Incidents + hazards | Dispatch police cars; cordon protests | Incident resolution, crowd safety |
| **Planner** | Aggregate metrics only | Set per-role priorities; broadcast directives | Priority coherence, anticipation of load spikes |

**Reward System:**
- **Per-tick signal:** movement progress, dispatch intent, planner anticipation, penalties for delays/collisions
- **3-layer verifier gating:** programmatic → system-state → semantic (failures zero that agent's tick reward)
- **Per-agent + global:** summed into composite city score (0–1 scale)

---

## 💾 Persistent Memory

High-risk zones (accident hotspots, congestion clusters) and past failures accumulate in `localStorage` (browser) or JSON file (Python).

- **DeliveryAgent** queries: finds alternate routes avoiding known trouble spots
- **PlannerAgent** queries: anticipates load in familiar problem zones
- **Clear memory** button resets the store for ablation studies

---

## 📈 Evaluation & Comparison

The `Evaluator` class runs baseline (no memory, fixed difficulty) vs trained (memory-warmed, curriculum-adapted) on identical seeds:

```python
evaluator = Evaluator(..., config=EvalConfig(n_episodes=20))
cmp = evaluator.compare(
    baseline_bundle=None,  # use heuristic agents
    trained_bundle=PolicyBundle({...LLM policies...}),
)
print(cmp.summary())  # → avg scores, success rates, improvements
```

Output: side-by-side reward curves, delivery success rates, congestion metrics.

## 🔬 Train an LLM Policy

### Quick Start: Heuristic Baseline

Run the heuristic agents (no LLM) to see the environment in action and understand rewards:

```python
from citynexus import (
    DeliveryAgent, TrafficAgent, EmergencyAgent, PoliceAgent, PlannerAgent,
    TrainingPipeline, TrainingConfig, Evaluator, EvalConfig,
    PolicyBundle, CallablePolicy,
)

def make_agents():
    return [DeliveryAgent(), TrafficAgent(), EmergencyAgent(), PoliceAgent(), PlannerAgent()]

# 1. Train heuristic baseline with curriculum + memory
pipeline = TrainingPipeline(
    agents_factory=make_agents,
    config=TrainingConfig(
        n_episodes=50, 
        log_dir="runs", 
        memory_path="runs/mem.json",
        curriculum_target=0.55,
    ),
)
summary = pipeline.train()
print(f"Baseline: avg score {summary.mean_score:.3f}, success rate {summary.mean_delivery_success:.3f}")

# 2. Collect trajectories for LLM fine-tuning
trajectories = pipeline.trajectories  # → convert to SFT dataset for TRL/Unsloth

# 3. (Optional) Evaluate a trained LLM policy
def my_llm_policy(obs, ctx): ...  # your trained model inference

evaluator = Evaluator(agents_factory=make_agents, config=EvalConfig(n_episodes=20))
cmp = evaluator.compare(
    trained_bundle=PolicyBundle({"delivery": CallablePolicy("delivery", my_llm_policy)}),
)
print(f"Trained: avg score {cmp.trained_score:.3f}, improvement {cmp.improvement_pct:.1f}%")
```

### Full Training Pipeline (Colab)

**See [`notebooks/train_citynexus_colab.ipynb`](notebooks/train_citynexus_colab.ipynb)** for:
- Setup (install + imports)
- Quick env demo (30 ticks)
- Full training (heuristic + curriculum + memory + verifier-gated rewards)
- Plotting (reward curves, city score, per-agent breakdown)
- Evaluator (baseline vs trained comparison)
- **Optional:** Unsloth + HF TRL scaffolding for LLM fine-tuning

**To run:** Upload to Colab, edit the `GITHUB_URL` cell to point at your fork, and run top-to-bottom.

---

## 🛠️ Technical Details

**OpenEnv Compliance:**
- ✅ Inherits from `openenv.core.env_server.Environment`
- ✅ Implements `reset(seed, episode_id) → observation`, `step(action) → observation`, `state` property
- ✅ Valid `openenv.yaml` manifest
- ✅ Pydantic schemas for `CityAction`, `CityObservation`, `CityNexusEnvState`
- ✅ FastAPI server in `server/app.py` for HF Spaces deployment

**Training Pipeline:**
- Adaptive curriculum (difficulty drifts toward agent's target score)
- Persistent memory (JSON-backed, agents query in `decide()`)
- Verifier-gated rewards (composable 3-layer checks)
- MetricsLogger with JSONL persistence for re-plotting
- Trajectory collection for LLM fine-tuning (Unsloth/TRL)

**Performance:**
- Pure JS simulation in browser: ~60 FPS on modern machines
- Python pipeline: ~200 ticks/sec on CPU

---

## 📚 References

- **OpenEnv:** https://github.com/huggingface/openenv
- **Colab notebook:** `notebooks/train_citynexus_colab.ipynb`
- **HF Spaces:** `[ADD URL]`
- **Blog / Video:** `[ADD LINK]`

---

## License

MIT
