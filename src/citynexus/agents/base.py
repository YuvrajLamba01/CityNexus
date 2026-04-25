"""Multi-agent framework: roles, base agent ABC, shared context, action marker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, Any

from citynexus.agents.messages import Message, MessageBus
from citynexus.agents.spaces import Space
from citynexus.env.world_state import WorldState

if TYPE_CHECKING:
    from citynexus.entities.delivery import Delivery
    from citynexus.entities.incident import Incident
    from citynexus.entities.unit import ResponderUnit
    from citynexus.memory.store import MemoryStore


class AgentRole(str, Enum):
    DELIVERY = "delivery"
    TRAFFIC = "traffic"
    EMERGENCY = "emergency"
    POLICE = "police"
    PLANNER = "planner"


@dataclass
class Action:
    """Marker base for typed actions. Subclasses add their own fields and override `kind`."""
    kind: str = "noop"


@dataclass
class NoOp(Action):
    """Universal no-op action."""
    kind: str = "noop"


@dataclass
class AgentContext:
    """Per-tick state shared between coordinator and agents."""
    tick: int = 0
    world: WorldState | None = None
    deliveries: dict[str, "Delivery"] = field(default_factory=dict)
    units: dict[str, "ResponderUnit"] = field(default_factory=dict)
    incidents: dict[str, "Incident"] = field(default_factory=dict)
    bus: MessageBus = field(default_factory=MessageBus)
    priorities: dict[AgentRole, float] = field(default_factory=dict)
    memory: "MemoryStore | None" = None

    # ----- bus pass-throughs (ergonomic shortcuts) ------------------------

    def send(self, msg: Message) -> None:
        self.bus.send(msg)

    def receive(self, role: AgentRole, *types: type[Message]) -> list[Message]:
        return self.bus.receive(role, *types)

    def messages_for(self, role: AgentRole) -> list[Message]:
        """Backwards-compatible alias for `receive(role)`."""
        return self.bus.receive(role)


class BaseAgent(ABC):
    """Abstract base for all agents.

    Subclass this per role. To swap heuristic policy for an LLM/RL policy,
    override `decide()` (and optionally `encode_observation()`/`decode_action()`).
    """

    role: AgentRole

    def __init__(self, role: AgentRole, *, rng: Random | None = None) -> None:
        self.role = role
        self.rng = rng or Random()
        self._policy_override = None  # callable (obs, ctx) -> list[Action]; bypasses decide()

    def set_policy(self, fn) -> None:
        """Install a policy override that replaces `decide()` for this agent.

        `fn` is any callable `(obs, ctx) -> list[Action]` — e.g. a trained model
        wrapped in a Policy. Pass `None` to revert to the heuristic decide().
        """
        self._policy_override = fn

    # --- spaces ------------------------------------------------------------

    @property
    @abstractmethod
    def observation_space(self) -> Space: ...

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Discrete head over `action_types`. RL trainers can wire parameter heads on top."""

    @property
    @abstractmethod
    def action_types(self) -> tuple[type[Action], ...]: ...

    # --- core loop ---------------------------------------------------------

    @abstractmethod
    def observe(self, world: WorldState, ctx: AgentContext) -> dict: ...

    @abstractmethod
    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]: ...

    def act(self, world: WorldState, ctx: AgentContext) -> list[Action]:
        obs = self.observe(world, ctx)
        if self._policy_override is not None:
            return self._policy_override(obs, ctx)
        return self.decide(obs, ctx)

    def reset(self) -> None:
        """Optional hook for clearing per-episode internal state."""

    # --- RL extensibility hooks (default no-ops; override per role) --------

    def encode_observation(self, obs: dict) -> Any:
        return obs

    def decode_action(self, raw: Any, obs: dict, ctx: AgentContext) -> list[Action]:
        raise NotImplementedError(
            f"{type(self).__name__}.decode_action() must be implemented for RL training."
        )
