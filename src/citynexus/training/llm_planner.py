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

    .. note::
       This is the legacy single-function form retained for back-compat.
       New training runs should pass the three independent components
       (``reward_correctness``, ``reward_format``, ``reward_length``) so
       TRL logs each curve separately — matches hackathon guide §7.
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
# Decomposed reward functions — pass these to ``GRPOTrainer.reward_funcs``
# as a list so TRL logs each component separately. Matches the hackathon
# guide's §7 recommendation for multiple independent reward signals.
# ---------------------------------------------------------------------------

def reward_correctness(
    prompts: Sequence[str] | None,
    completions: Sequence[str],
    expert: Sequence[str],
    *,
    match_reward: float = 1.0,
    miss_reward: float = 0.0,
    **_: Any,
) -> list[float]:
    """+1 if the completion's first token matches the heuristic expert mode,
    else 0. This is the verifiable RLVR signal — the model can only earn it
    by producing the right label."""
    return [
        float(match_reward if _first_token(c) == e else miss_reward)
        for c, e in zip(completions, expert)
    ]


def reward_format(
    prompts: Sequence[str] | None,
    completions: Sequence[str],
    expert: Sequence[str] | None = None,
    *,
    valid_reward: float = 0.5,
    invalid_reward: float = -0.5,
    **_: Any,
) -> list[float]:
    """Format check: completion's first token must be one of the four mode
    names. On its own this is gameable (always emit ``"normal"`` to score),
    but combined with ``reward_correctness`` it forces well-formed *correct*
    output — which is the actual training target."""
    return [
        float(valid_reward if _first_token(c) in MODES else invalid_reward)
        for c in completions
    ]


def reward_length(
    prompts: Sequence[str] | None,
    completions: Sequence[str],
    expert: Sequence[str] | None = None,
    *,
    soft_cap: int = 16,
    hard_cap: int = 64,
    over_soft_penalty: float = -0.1,
    over_hard_penalty: float = -0.5,
    **_: Any,
) -> list[float]:
    """Length penalty: discourages essay-style completions. Returns 0 for
    short completions, ``over_soft_penalty`` past ``soft_cap``, and
    ``over_hard_penalty`` past ``hard_cap``."""
    out: list[float] = []
    for c in completions:
        n = len(c) if isinstance(c, str) else 0
        if n > hard_cap:
            out.append(float(over_hard_penalty))
        elif n > soft_cap:
            out.append(float(over_soft_penalty))
        else:
            out.append(0.0)
    return out


def reward_env_lookahead(
    prompts: Sequence[str] | None,
    completions: Sequence[str],
    rewards_by_mode: Sequence[dict[str, float]] | None = None,
    *,
    invalid_penalty: float = -1.0,
    **_: Any,
) -> list[float]:
    """**The env-driven RLVR signal.**

    Returns the actual next-tick env reward of applying the LLM's chosen mode
    at the current observation. ``rewards_by_mode`` is precomputed by
    ``build_dataset(..., with_env_rewards=True)`` — for each observation we
    replay the env from ``(seed, history)`` and record what each of the four
    modes would have earned.

    Why this matters: the other reward functions (``reward_correctness``,
    ``reward_format``) compare the model to a *hand-coded heuristic* — the
    LLM caps at heuristic quality. This function compares the model to *real
    env outcomes*, so the LLM can learn modes the heuristic gets wrong.

    Off-format completions (not one of the four modes) get ``invalid_penalty``.
    If ``rewards_by_mode`` is missing (e.g. the dataset was built without
    ``with_env_rewards``), returns zeros — back-compat-safe.
    """
    if rewards_by_mode is None:
        return [0.0] * len(completions)
    out: list[float] = []
    for comp, lookup in zip(completions, rewards_by_mode):
        first = _first_token(comp)
        if isinstance(lookup, dict) and first in lookup:
            out.append(float(lookup[first]))
        else:
            out.append(float(invalid_penalty))
    return out


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

@dataclass
class PromptSample:
    prompt: str
    expert: str
    # Optional env-driven lookup: ``rewards_by_mode[m]`` is the next-tick env
    # reward you'd receive if mode ``m`` were applied at this observation.
    # Populated by ``build_dataset(..., with_env_rewards=True)``. Lets the GRPO
    # trainer score completions against actual env outcomes (RLVR), not just
    # against a hand-coded heuristic — so the LLM can exceed the heuristic.
    rewards_by_mode: dict[str, float] | None = None
    # Episode seed and the mode history that led to this observation. Optional
    # but useful for re-derivation / debugging.
    seed: int | None = None
    history: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d: dict[str, Any] = {"prompt": self.prompt, "expert": self.expert}
        if self.rewards_by_mode is not None:
            d["rewards_by_mode"] = dict(self.rewards_by_mode)
        return d


def _evaluate_modes_at_state(
    env_factory: Callable[[], Any],
    action_cls: Callable[..., Any],
    seed: int,
    history: Sequence[str],
    modes: Sequence[str] = MODES,
) -> dict[str, float]:
    """Replay a fresh env from ``seed`` + ``history`` for each mode in ``modes``,
    apply the mode for one step, and return the resulting tick reward.

    The env is deterministic given (seed, action_history), so this faithfully
    reproduces what would have happened had the agent picked each mode at this
    decision point. Cost: ``len(modes)`` replays of ``len(history)+1`` env
    steps each. With history ≤ 80 and modes = 4, this is ~320 env steps.
    """
    out: dict[str, float] = {}
    for mode in modes:
        env = env_factory()
        env.reset(seed=seed)
        for past in history:
            env.step(action_cls(mode=past))
        next_obs = env.step(action_cls(mode=mode))
        out[mode] = float(getattr(next_obs, "reward", 0.0) or 0.0)
    return out


def build_dataset(
    env: Any,
    *,
    n_episodes: int = 40,
    seed: int = 0,
    base_seed: int = 1000,
    action_cls: Callable[..., Any] | None = None,
    env_factory: Callable[[], Any] | None = None,
    with_env_rewards: bool = False,
) -> list[PromptSample]:
    """Roll the OpenEnv wrapper with random actions to build (prompt, expert) pairs.

    Parameters
    ----------
    env :
        Active OpenEnv-compliant env. Must expose ``reset(seed=...) ->
        Observation`` and ``step(action) -> Observation``; the observation must
        support ``.model_dump()`` (Pydantic v2 model).
    n_episodes :
        Number of full episodes to roll.
    seed, base_seed :
        Top-level RNG seed for action sampling, and the per-episode reset seed.
    action_cls :
        The OpenEnv ``Action`` class to instantiate (defaults to
        ``server.models.CityAction``).
    env_factory :
        Callable returning a fresh env instance. Required when
        ``with_env_rewards=True`` so we can replay the env independently for
        each candidate mode at each observation. Defaults to ``type(env)()``.
    with_env_rewards :
        When True, each ``PromptSample`` is augmented with
        ``rewards_by_mode[m] = next-tick env reward if mode m is applied``.
        This is the verifiable env-driven signal that lets the GRPO-trained
        LLM exceed the hand-coded heuristic ceiling. Adds ~5–10 minutes of
        precomputation for a 40-episode dataset on CPU; **disabled by default**
        for back-compat with smoke tests and CPU-only callers.
    """
    if action_cls is None:
        # Late import — avoids forcing FastAPI deps on consumers of this module.
        from server.models import CityAction as action_cls  # type: ignore
    if with_env_rewards and env_factory is None:
        # Best-effort default: instantiate the same class with no args.
        env_factory = lambda: type(env)()  # noqa: E731

    rng = random.Random(seed)
    samples: list[PromptSample] = []
    for ep in range(n_episodes):
        ep_seed = base_seed + ep
        obs = env.reset(seed=ep_seed)
        history: list[str] = []
        while not getattr(obs, "done", False):
            d = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
            sample = PromptSample(
                prompt=obs_to_prompt(d),
                expert=expert_mode(d),
                seed=ep_seed,
                history=tuple(history),
            )
            if with_env_rewards:
                sample.rewards_by_mode = _evaluate_modes_at_state(
                    env_factory, action_cls, ep_seed, list(history), MODES,
                )
            samples.append(sample)
            chosen = rng.choice(MODES)
            history.append(chosen)
            obs = env.step(action_cls(mode=chosen))
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
    "reward_correctness",
    "reward_format",
    "reward_length",
    "reward_env_lookahead",
    "PromptSample",
    "build_dataset",
    "expert_distribution",
    "LLMPlannerPolicy",
    "run_llm_episode",
]
