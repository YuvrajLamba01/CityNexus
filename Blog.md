# Building CITYNEXUS: A Self-Evolving Multi-Agent City for the OpenEnv Hackathon 2026

> *A 20×20 grid run by five LLM-trainable agents under a curriculum that actively hunts for whichever agent is currently the weakest, with verifier-gated rewards that only credit the role responsible. Trains end-to-end on a Colab T4 in about an hour. This is the story of why I built it that way.*

---

## The setup

The OpenEnv 2026 hackathon brief asked for environments that exercise multi-agent interactions and self-improvement. Most multi-agent RL environments I've used either give you cooperation **or** an adaptive opponent. CITYNEXUS gives you both, in a tightly-coupled loop, on a board small enough to train an actual LLM on a free Colab GPU.

The whole thing is one repo:

- A grid city with traffic, weather, deliveries, accidents, civic incidents
- Five role agents (Delivery, Traffic, Emergency, Police, Planner), each with its own partial observation
- A 3-layer verifier that gates per-agent rewards
- An adversarial scenario generator + adaptive curriculum
- A persistent memory store that survives episode boundaries
- An OpenEnv-compatible FastAPI server
- A static playable browser demo with **click-to-replay of recorded policies** (random baseline, heuristic expert, and the GRPO-trained LLM) on the same held-out seeds the headline metrics measure
- A Colab notebook that fine-tunes Qwen-2.5-0.5B as the Planner via GRPO

By the time I'm done writing this, the entire stack passes 9 smoke tests in ~5 seconds, the heuristic agents cope with the curriculum's escalating pressure, and the GRPO-trained 0.5B-parameter Planner beats a random-mode controller by +6.9 % on held-out seeds.

But none of that was the interesting part. The interesting parts were the design decisions that made all of it possible on a single Colab free-tier T4 session. Let me walk through the four that mattered most.

---

## Decision 1: Five views, five rewards, one bus

I almost made the classic mistake of giving every agent access to the full world state. It's tempting because it's easy: one big `WorldState` object, every agent reads what it needs.

The problem is that you don't actually learn coordination that way. You learn five copies of "look at everything, do something". Coordination only becomes meaningful when agents *can't* see what each other are doing, and have to communicate explicitly to align their plans.

So in CITYNEXUS, every agent's `observe()` method gets only the slice of the world its job needs:

- **Delivery** sees roads inside the bounding box of its pending packages.
- **Traffic** only sees intersections.
- **Emergency** sees discs around active accidents.
- **Police** sees discs around incidents plus smaller hazard discs around accidents.
- **Planner** doesn't see *any* per-cell state — only aggregates.

To do anything together, agents publish typed messages on a per-tick `MessageBus` and read them on the next observation. The bus is reset every tick, so messages are commitments, not gossip. Cooperation is a learned skill that has to fight against partial observation noise.

The Planner deserves a special note. Its observation is *intentionally* the most impoverished — no per-cell, no per-entity, just aggregate metrics. That made it the perfect seat for an LLM, because:

1. The action space is tiny (4 city postures: `normal`, `emergency_focus`, `delivery_focus`, `defensive`).
2. The observation can be serialized into a short prompt.
3. Every other agent's positive reward gives the Planner a 15% kickback, so its objective is *whole-system coordination* — not "do my own job well".

Make the LLM the conductor, not the percussionist. That single decision is what made the whole training story tractable.

---

## Decision 2: A curriculum that hunts for weakness

Static curricula are boring. Linear difficulty schedules are slightly less boring. What I actually wanted was: *the world should attack whatever the agents just failed at*.

So the `Curriculum` class tracks two things after every episode:

1. A P-controller on `(score - target)` that nudges difficulty up or down.
2. An exponential moving average over four classified failure modes: `delivery_failure`, `accident_pileup`, `incident_pileup`, `congestion_overload`.

After every episode, `top_failure_modes(n=3)` returns the most pressing weaknesses. The `AdversarialGenerator` consumes that list and *re-weights its shock distribution*: each failure mode carries a list of `suggested_shock_kinds` whose weights get multiplied by `1.0 + 1.5 * severity`.

The result is what I think of as the "shifting pressure point". Master delivery routing in the early episodes? The curriculum starts throwing emergency clusters. Master those? Weather storms. Master those? Civic protests that block intersections. The agents never get to settle.

There's a plot in the repo (`runs/curriculum_failure_modes.png`) that replays each episode's metrics through a fresh `Curriculum` and snapshots the failure-mode EMA after every update. You can literally watch the curriculum's pressure point shift from `accident_pileup` to `incident_pileup` to `congestion_overload` as the agents adapt to each in turn.

That's the self-improvement loop, and it's the core conceptual contribution of the project. Everything else exists to make the loop *trainable*.

---

## Decision 3: Verifier-gated, role-attributed rewards

If you train multi-agent systems with monolithic rewards — one number for the whole team — you'll spend half your time debugging why your gradient signal is noisy. Police screws up, everyone gets punished, Delivery learns nothing useful that episode. The signal-to-noise ratio is brutal.

CITYNEXUS uses **attributed gating**. Every action goes through a 3-layer verifier:

1. **Programmatic checks** — type, range, schema-level invariants. Does the dispatched unit exist? Is the roadblock placement on a road cell?
2. **System-state checks** — the world's response. Did the delivery actually move? Did the incident actually resolve?
3. **Semantic checks** — was the choice *appropriate* given role priorities and visible signals?

Each `Check` returns a `CheckResult` with `attributed_to: list[AgentRole]`. The `MultiAgentRewardSystem` then runs `_apply_gating` (see `src/citynexus/rewards/system.py:238-263`), which under `GatingMode.ATTRIBUTED` zeros only those roles' rewards. Everyone else still earns. Delivery keeps learning while Police gets punished.

This single decision is what makes the gradient signal stay clean as the curriculum gets meaner. Without attribution, training would collapse the moment the curriculum found a weakness — every agent would get blamed for one agent's mistake.

---

## Decision 4: Train the LLM with a verifiable reward, not a value head

I had a single Colab free-tier T4 session as my training budget — about an hour before I had to worry about being timed out. That rules out actor-critic methods on a small model — value heads need a lot of samples to stabilize and GPU minutes I didn't have.

GRPO (Group-Relative Policy Optimization) is essentially policy gradient with a learned baseline computed *per-prompt-group*. No value head, no critic, just rank completions within a group and push toward the high-reward ones. Hugging Face's `trl.GRPOTrainer` handles the implementation; all you have to provide is a reward function.

This is where RLVR (Reinforcement Learning with Verifiable Rewards) shines. The reward function I wrote (`src/citynexus/training/llm_planner.py:97`) has three components by design:

1. **Format**: completion must start with one of the four valid modes.
2. **Correctness**: first token must match the heuristic expert label.
3. **Length**: completions longer than 32 chars get a stacking penalty.

Anti-gaming properties (verified in smoke tests):

- A syntactically-valid wrong mode beats junk text but loses to a correct mode.
- Length penalty stacks: "correct but verbose" beats "verbose junk" but loses to "correct and concise".
- The same function is used during training (via `trl.GRPOTrainer.reward_funcs`) and during evaluation, so reward and eval can't drift apart silently.

That last point is critical. I've seen too many RL projects where the eval metric and the training reward drift apart over weeks of tweaking, leading to unfalsifiable claims about "learning". Using one function for both forces honesty.

---

## What actually happened during training

I trained the heuristic stack for 40 episodes on a Colab T4, then ran 1600 GRPO steps on Qwen-2.5-0.5B with LoRA + 4-bit quantization (Unsloth's `FastLanguageModel`). Total wall clock: about an hour, mostly the GRPO step.

The numbers (verbatim from notebook Section 8):

```
### Heuristic Pipeline (Sections 2-5)
- Episodes trained:               40
- Mean score (all):               0.553
- Last-5 avg score:               0.556
- Mean delivery success:          0.532
- Last-5 delivery success:        0.616

### GRPO LLM RL (Section 6)
- Random-mode baseline cumulative reward: mean=+72.25, stdev=4.71
- GRPO-trained LLM cumulative reward:     mean=+77.20, stdev=5.45
- Improvement (trained - baseline):       +4.95 (+6.9%)
- Welch's t (one-sided):                  t≈+2.17, p≈0.044
- Cohen's d:                              d≈0.97 (large effect)
```

A few things to call out about these numbers:

**Score holds.** The headline isn't "score went up". It's "score *retained* its level while the world got harder around the agents". The curriculum P-controller climbs difficulty as the agents demonstrate competence, so a stable score under increasing difficulty is the success condition, not a flat curve. The score in the last 5 episodes (0.556) is essentially the same as the mean across all 40 (0.553), but the *difficulty* in those last 5 episodes is meaningfully higher than at the start.

**Delivery success climbs.** Last-5 delivery success (0.616) is meaningfully higher than the all-episodes mean (0.532). The persistent `MemoryStore` is doing real work here — it remembers high-risk zones across episodes, and the `DeliveryAgent` consults it during routing. By the last 5 episodes, the store has accumulated useful spatial knowledge.

**GRPO improvement is statistically significant, not just numerically larger.** `+72.25 ± 4.71` (random) vs `+77.20 ± 5.45` (trained) on 10 held-out seeds. A one-sided **Welch's t-test** on the per-seed cumulative rewards gives **t ≈ +2.17, p ≈ 0.044** — under the null hypothesis that random and GRPO-trained planners are drawn from the same distribution, the observed gap (or larger) shows up only ~4.4 % of the time. **Cohen's d ≈ 0.97**, which is a "large effect" by Cohen's convention (d ≥ 0.8). For a 0.5B-parameter model with 1600 GRPO steps on a 4-action problem trained inside one Colab T4 hour, that's a clean signal — and notebook §6b prints both numbers directly from the rollout cumulative-reward arrays so the test isn't post-hoc.

**Variance is high per-episode.** Single-episode score swings between 0.33 and 0.71. Always show rolling-5 alongside raw — looking at single-episode numbers alone is misleading.

---

## Making the trained policy clickable

A statistical test is the right way to *prove* the model learned something. It is not the right way to *show* it.

So the static demo Space ships a **Trained Model Playback** panel right under the live city. Three recorded policies (`Random baseline`, `Heuristic expert`, `GRPO-trained Qwen-2.5-0.5B Planner`), ten held-out seeds (3000–3009), one button. Click `Play recorded policy` and the recorded per-tick mode choices drive the live in-browser city — mode pill updates every tick, progress bar fills, env-reward HUD ticks live, and the Python-side recorded reward is shown for comparison after the run finishes.

The mechanism: notebook §6c dumps the trained LLM's per-tick mode trace to `runs/llm_rollouts.json` (mirror at `web/data/llm_rollouts.json`). The in-browser sim shares the same `_MODE_PRIORITIES` table as the Python OpenEnv server, so replaying the trace through the JS engine is a faithful visualization of *what the trained policy actually did*. The cumulative reward shown live is an independent JS-sim measurement; it doesn't have to match the Python eval to the decimal — the point is the per-tick policy behavior, which it does match.

Why bother? Because the difference between a hackathon submission that reads as "complete" and one that reads as "alive" is whether a reviewer with five spare minutes can *see* the model behave differently from random. p < 0.05 says it learned. The playback panel lets you watch it.

---

## What I learned

A few things surprised me along the way:

**The Planner's tiny action space matters more than its model size.** I originally tried letting the LLM emit free-text directives. That trained slowly and never converged inside the T4 budget. Collapsing the action space to four enum choices is what made GRPO converge inside one Colab session at all. The lesson: for time-budgeted RL on small LLMs, *constrain the output space first, scale the model second*.

**Attributed gating made debugging tractable.** When a training run misbehaved, I could look at per-agent reward traces and see exactly which role's policy was failing. Monolithic rewards would have made this nearly impossible — you'd be guessing at causal attribution from aggregate signals.

**The curriculum surprised me.** I'd designed it to bias toward whatever failure mode was highest. What I didn't expect was that this naturally produced *coordination failures* in particular — `accident_pileup` happens when Emergency, Police, and Traffic don't sequence their dispatches. The curriculum and the multi-agent design ended up amplifying each other in ways I didn't plan.

**Verifiable rewards beat learned reward models for hackathon timeframes.** I considered training a small reward model. It would have taken longer than the actual policy training. A few hundred lines of Python checking format/correctness/length got me 90% of what I'd need from an LLM-based evaluator with none of the wall-clock cost or unfalsifiability problems.

---

## What I'd do next

The repo is hackathon-complete, but here's what I'd build with another week:

- **More agent roles.** Add a `MaintenanceAgent` for road-quality decay and a `WeatherAgent` for proactive forecasts that other agents can subscribe to via the message bus.
- **Train all five seats with GRPO.** Currently only the Planner is LLM-trainable. The other four are heuristics that the verifier checks. Making each role's heuristic an LLM call would multiply the coordination story by 5.
- **Curriculum dropout.** Right now the curriculum only adds pressure. A real adaptive curriculum should also *relax* in domains the agents have mastered, to prevent overfitting to the most recent failure mode.
- **Cross-episode reward shaping from memory.** The `MemoryStore` currently drives observation behavior. Letting it also shape rewards (e.g. extra credit for revisiting a zone the agents previously failed in and now succeed) would close another loop.

---

## Where to go from here

If you want to play with CITYNEXUS, three entry points:

1. **Fastest path to seeing it work**: clone the repo, `pip install -e .`, `pytest -q`. 9 smoke tests cover every layer of the stack.
2. **See the actual training run**: open the [Colab notebook](https://colab.research.google.com/drive/1sJHVtNQIVvzBuynGe6rokwIs8Cf5yJdA?usp=sharing) — the cell outputs are preserved from the T4 run that produced the numbers above.
3. **Play with it in the browser**: the static demo (`web/index.html`) runs the simulation client-side with live charts, agent message stream, and persistent memory in `localStorage`. The Trained Model Playback panel underneath the live city lets you watch the GRPO-trained policy's actual mode choices animate the same held-out seeds. No install needed.
4. **Open the hosted Space**: visit the [Hugging Face Space](https://huggingface.co/spaces/yuvraaj23/CityNexus) for the deployed browser demo and playback experience.

The full architecture deep-dive with file:line citations into every claim is in the repository's `README.md`.

Thanks for reading. — *Yuvraj Lamba*
