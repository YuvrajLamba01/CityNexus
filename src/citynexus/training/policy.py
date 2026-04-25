"""Policy abstraction — RL-friendly wrapper for swappable per-agent decision policies.

Any callable `(obs, ctx) -> list[Action]` works as a policy. The provided classes
are convenience wrappers:

  HeuristicPolicy(agent)           — uses the agent's existing decide()
  CallablePolicy(role, fn)          — wraps a bare function (e.g. an RL model adapter)
  PolicyBundle({role: policy, ...}) — install on a coordinator's agents in one call
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from citynexus.agents.base import Action, AgentContext, BaseAgent


class Policy(ABC):
    """Per-agent policy. The trainer swaps these without touching the agent class."""

    role: str

    @abstractmethod
    def act(self, obs: dict, ctx: "AgentContext") -> list["Action"]: ...

    def __call__(self, obs: dict, ctx: "AgentContext") -> list["Action"]:
        # So a Policy instance is also a plain callable — installs cleanly via set_policy().
        return self.act(obs, ctx)

    def reset(self) -> None:
        """Optional: clear per-episode internal state."""

    def update(self, transitions) -> None:
        """Optional: online-learning hook. RL trainers may call this between episodes."""


class HeuristicPolicy(Policy):
    """Wraps an agent's heuristic `decide()` so it can be passed where a Policy is expected."""

    def __init__(self, agent: "BaseAgent") -> None:
        self.agent = agent
        self.role = agent.role.value

    def act(self, obs: dict, ctx: "AgentContext") -> list["Action"]:
        return self.agent.decide(obs, ctx)

    def reset(self) -> None:
        self.agent.reset()


class CallablePolicy(Policy):
    """Wraps a bare callable. Intended adapter for trained models."""

    def __init__(
        self,
        role: str,
        fn: Callable[[dict, "AgentContext"], list["Action"]],
        *,
        reset_fn: Callable[[], None] | None = None,
    ) -> None:
        self.role = role
        self.fn = fn
        self._reset_fn = reset_fn

    def act(self, obs: dict, ctx: "AgentContext") -> list["Action"]:
        return self.fn(obs, ctx)

    def reset(self) -> None:
        if self._reset_fn is not None:
            self._reset_fn()


class PolicyBundle:
    """Per-role policy mapping. Apply to a coordinator's agents in one call."""

    def __init__(self, policies: dict[str, Policy] | None = None) -> None:
        self.policies: dict[str, Policy] = dict(policies or {})

    def set(self, role: str, policy: Policy) -> None:
        self.policies[role] = policy

    def get(self, role: str) -> Policy | None:
        return self.policies.get(role)

    def install(self, agents: Iterable["BaseAgent"]) -> None:
        """Override `decide()` for each matching agent via `set_policy()`."""
        for agent in agents:
            policy = self.policies.get(agent.role.value)
            if policy is not None:
                agent.set_policy(policy)

    def uninstall(self, agents: Iterable["BaseAgent"]) -> None:
        """Revert all overrides → agents fall back to their heuristic decide()."""
        for agent in agents:
            agent.set_policy(None)

    def reset_all(self) -> None:
        for p in self.policies.values():
            p.reset()
