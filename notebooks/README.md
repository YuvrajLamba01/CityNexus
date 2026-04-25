# Notebooks

## `train_citynexus_colab.ipynb`

End-to-end training notebook for Google Colab. 26 cells, 6 sections:

1. **Setup** — install + bring in the `citynexus` package (clone from GitHub *or* upload zip)
2. **Quick env demo** — 30 ticks, sanity check
3. **Train** — `TrainingPipeline` with adaptive curriculum + persistent memory + verifier-gated rewards
4. **Plots** — episode score, curriculum difficulty, summed reward, per-agent breakdown (saved as PNG)
5. **Evaluate** — same-seed baseline vs trained (memory-warmed) comparison + bar chart
6. **Optional** — Unsloth + TRL scaffolding for fine-tuning a small LLM as the policy

## How to use

### Option A — upload the notebook
1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → pick `train_citynexus_colab.ipynb`
3. Edit cell 2 (`GITHUB_URL`) to point at your fork after pushing CITYNEXUS to GitHub
4. Run cells top-to-bottom

### Option B — paste cell by cell
1. Open a fresh Colab notebook
2. Open `train_citynexus_colab.ipynb` here in the repo (or in JupyterLab)
3. Copy each code cell into a Colab cell and run

### Option C — no GitHub
1. From your local repo: `zip -r citynexus.zip CITYNEXUS/` (or use 7-Zip on Windows)
2. In Colab cell 4, comment-out the GitHub clone block, uncomment the `files.upload()` block
3. Drag `citynexus.zip` into the upload prompt

## What gets produced

The training run writes to `runs/`:

| File | Contents |
|---|---|
| `training.jsonl` | One line per episode — score, difficulty, success rate, per-agent reward, etc. |
| `memory.json` | Persistent memory store — high-risk zones + past failures (survives reruns) |
| `training_curves.png` | 4-panel plot: score, difficulty, summed reward, success rate |
| `per_agent_reward.png` | Per-role cumulative reward across episodes |
| `baseline_vs_trained.png` | Bar chart — baseline vs memory-trained, same seeds |

All three PNGs are publication-ready (140 dpi, dark theme matching the canvas frontend).

## Hardware

Sections 1–5 run on **CPU** (Colab's free tier is fine).
Section 6 (Unsloth + TRL fine-tuning) needs a **GPU runtime** — switch via *Runtime → Change runtime type → T4 GPU* before running those cells.

## What "training" means here

Even before the LLM scaffolding, the system has real adaptive behaviour:

- **Curriculum** — difficulty drifts toward the agent's competence (target score 0.55).
- **Memory** — across episodes, agents accumulate spatial knowledge (high-risk zones) that the `DeliveryAgent` consults during routing and the `PlannerAgent` consults for anticipatory priority shifts.
- **Verifier-gated rewards** — per-agent reward is zeroed for ticks where that role's actions fail one of the 3-layer checks. This produces the gradient signal an LLM RL trainer would consume.

Section 6 shows how to plug a learned LLM policy (via `CallablePolicy`) into the same `Evaluator.compare(...)` harness.
