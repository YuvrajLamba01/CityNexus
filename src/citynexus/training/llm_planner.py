"""LLM-driven Planner policy + verifiable reward for CITYNEXUS.

This module is the single source of truth for everything the GRPO notebook
needs. The notebook imports from here so the prompt, reward, dataset shape,
and inference path stay in lock-step with what gets trained.

Surface
-------
* ``MODES``                — the four city postures the Planner can pick.
* ``expert_mode(obs)``     — heuristic best mode for a CityObservation dict.
* ``obs_to_prompt(obs)``   — prompt format the LLM is trained on.
* ``grpo_reward(...)``     — multi-component verifiable reward used by GRPO.
* ``build_dataset(...)``   — roll the OpenEnv wrapper to produce (prompt, expert) pairs.
* ``LLMPlannerPolicy``     — wraps a HF model so it can be installed via ``PolicyBundle``.
* ``run_llm_episode(...)`` — convenience evaluator that drives an episode with the LLM.

Why this lives in the package
-----------------------------
1. The notebook used to redefine ``expert_mode`` / ``obs_to_prompt`` twice. That
   is a footgun: drift between the two copies silently corrupts evaluation.
2. Everything here is import-safe on CPU (no torch / unsloth / trl required at
   module load). Heavy deps are imported lazily inside ``LLMPlannerPolicy``.
3. Smoke tests can exercise the reward, prompt builder, and dataset constructor
   without a GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES: tuple[str, ...] = (
    "normal",
    "emergency_focus",
    "delivery_focus",
    "defensive",
)


# ---------------------------------------------------------------------------
# Heuristic expert label
# ---------------------------------------------------------------------------

def expert_mode(obs: dict) -> str:
    """Heuristic best-mode label for a `CityObservation` dump.

    Priority order: emergency load → delivery backlog → congestion → default.
    Used both as the GRPO reward target and as a teacher-forcing baseline.
    """
    if obs.get("n_active_accidents", 0) >= 2 or obs.get("n_active_incidents", 0) >= 2:
        return "emergency_focus"
    if obs.get("n_open_deliveries", 0) >= 5:
        return "delivery_focus"
    if obs.get("congestion_ratio", 0.0) > 0.5:
        return "defensive"
    return "normal"


# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------

def obs_to_prompt(obs: dict) -> str:
    """Compact, model-readable prompt describing the current city state."""
    return (
        "You are the city planner for CITYNEXUS. Pick ONE city posture mode "
        "for the next tick.\n"
        f"\nState (tick {obs.get('tick', 0)}):\n"
        f"  weather: {obs.get('weather', 'clear')}\n"
        f"  congestion: {obs.get('congestion_ratio', 0.0):.2f}\n"
        f"  active accidents: {obs.get('n_active_accidents', 0)}\n"
        f"  active incidents: {obs.get('n_active_incidents', 0)}\n"
        f"  open deliveries: {obs.get('n_open_deliveries', 0)}\n"
        "\nModes:\n"
        "  normal           - balanced priorities\n"
        "  emergency_focus  - prioritize ambulances + police\n"
        "  delivery_focus   - prioritize logistics + traffic\n"
        "  defensive        - planner-led, cautious\n"
        "\nRespond with ONLY the mode name."
    )


# ---------------------------------------------------------------------------
# GRPO reward (verifiable, hard-to-game)
# ---------------------------------------------------------------------------

def _first_token(text: str) -> str:
    text = text.strip().lower() if text else ""
    return text.split()[0] if text else ""


def grpo_reward(
    prompts: Sequence[str] | None,
    completions: Sequence[str],
    expert: Sequence[str],
    *,
    correct_reward: float = 1.0,
    wrong_valid_reward: float = -0.3,
    invalid_reward: float = -0.7,
    length_penalty: float = -0.2,
    length_threshold: int = 32,
    **_: Any,
) -> list[float]:
    """Multi-component verifiable reward used by ``trl.GRPOTrainer``.

    Components
    ----------
    1. **Format**       — completion must start with one of the four valid modes.
    2. **Correctness**  — first token must match the heuristic expert label.
    3. **Length**       — completions longer than ``length_threshold`` chars are
       penalised to discourage runaway generations.

    The scoring is anti-gameable in two ways:
      * a syntactically-valid wrong mode is *worse* than a no-op blank but
        *better* than a junk string,
      * length penalty stacks on top so "correct but verbose" beats "verbose junk"
        but loses to "correct and concise".

    Returns one float per completion. The ``prompts`` arg is unused but
    accepted because TRL passes it positionally.
    """
    out: list[float] = []
    for comp, exp in zip(completions, expert):
        first = _first_token(comp)
        if first == exp:
            r = correct_reward
        elif first in MODES:
            r = wrong_valid_reward
        else:
            r = invalid_reward
        if isinstance(comp, str) and len(comp) > length_threshold:
            r += length_penalty
        out.append(float(r))
    return out


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

@dataclass
class PromptSample:
    prompt: str
    expert: str

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "expert": self.expert}


def build_dataset(
    env: Any,
    *,
    n_episodes: int = 40,
    seed: int = 0,
    base_seed: int = 1000,
    action_cls: Callable[..., Any] | None = None,
) -> list[PromptSample]:
    """Roll the OpenEnv wrapper with random actions to build (prompt, expert) pairs.

    `env` is expected to expose `reset(seed=...) -> CityObservation` and
    `step(action) -> CityObservation` and the observation must support
    `.model_dump()` / be a Pydantic model. Pass `action_cls` to keep this
    module independent from the FastAPI server's import path (CPU-only smoke
    tests can pass a stub).
    """
    if action_cls is None:
        # Late import — avoids forcing FastAPI deps on consumers of this module.
        from server.models import CityAction as action_cls  # type: ignore

    rng = random.Random(seed)
    samples: list[PromptSample] = []
    for ep in range(n_episodes):
        obs = env.reset(seed=base_seed + ep)
        while not getattr(obs, "done", False):
            d = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            samples.append(PromptSample(prompt=obs_to_prompt(d), expert=expert_mode(d)))
            obs = env.step(action_cls(mode=rng.choice(MODES)))
    return samples


def expert_distribution(samples: Iterable[PromptSample]) -> dict[str, int]:
    counts = {m: 0 for m in MODES}
    for s in samples:
        counts[s.expert] = counts.get(s.expert, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Inference policy
# ---------------------------------------------------------------------------

class LLMPlannerPolicy:
    """Wraps a HF causal LM as a Planner posture chooser.

    Designed so the same object can be called as:
        ``policy(obs_dict) -> str (one of MODES)``
    or installed via the OpenEnv loop:
        ``mode = policy.pick_mode(obs_dict)``

    Heavy deps (`torch`, `transformers`, `unsloth`) are accessed lazily so a
    test or CPU environment that imports this module doesn't crash.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str | None = None,
        max_new_tokens: int = 8,
        deterministic: bool = True,
        chat_template: bool = True,
        fallback: str = "normal",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.deterministic = deterministic
        self.chat_template = chat_template
        self.fallback = fallback if fallback in MODES else MODES[0]
        self._device = device

    # -------- public --------------------------------------------------------

    def pick_mode(self, obs: dict) -> str:
        """Return the model's chosen mode for ``obs``. Falls back to ``self.fallback``
        when the model emits anything unparseable."""
        prompt = self._format_prompt(obs_to_prompt(obs))
        text = self._generate(prompt)
        first = _first_token(text)
        if first in MODES:
            return first
        for m in MODES:
            if m in text.lower():
                return m
        return self.fallback

    def __call__(self, obs: dict) -> str:
        return self.pick_mode(obs)

    # -------- internals -----------------------------------------------------

    def _format_prompt(self, raw_prompt: str) -> str:
        if not self.chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return raw_prompt
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return raw_prompt

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _generate(self, prompt: str) -> str:
        import torch  # noqa: PLC0415

        device = self._resolve_device()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=not self.deterministic,
                temperature=0.0 if self.deterministic else 0.7,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def run_llm_episode(
    env: Any,
    pick_mode_fn: Callable[[dict], str],
    *,
    seed: int,
    action_cls: Callable[..., Any] | None = None,
) -> tuple[float, list[str]]:
    """Drive one full episode of the OpenEnv wrapper using ``pick_mode_fn``.

    Returns ``(cumulative_reward, mode_history)``. Useful for both the
    LLM-vs-baseline notebook chart and the smoke-test trained-policy harness.
    """
    if action_cls is None:
        from server.models import CityAction as action_cls  # type: ignore

    obs = env.reset(seed=seed)
    cum = 0.0
    history: list[str] = []
    while not obs.done:
        d = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        mode = pick_mode_fn(d)
        history.append(mode)
        obs = env.step(action_cls(mode=mode))
        cum += float(obs.reward)
    return cum, history


__all__ = [
    "MODES",
    "expert_mode",
    "obs_to_prompt",
    "grpo_reward",
    "PromptSample",
    "build_dataset",
    "expert_distribution",
    "LLMPlannerPolicy",
    "run_llm_episode",
]
