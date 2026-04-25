"""Pydantic models that define the OpenEnv surface for CITYNEXUS.

Action — one tick of high-level control over the city.
Observation — post-tick snapshot of the world + per-agent reward decomposition.
State — episode-level metadata exposed to the OpenEnv harness.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


CITY_MODES: tuple[str, ...] = (
    "normal",
    "emergency_focus",
    "delivery_focus",
    "defensive",
)


class CityAction(Action):
    """High-level per-tick control. Maps onto the Planner's priority mechanism.

    The LLM picks one of four city-wide postures each tick. The env installs a
    Planner policy override that emits the matching `SetPriority` actions, and
    the four heuristic role agents (Delivery, Traffic, Emergency, Police)
    react to the new priorities inside the same tick.
    """

    mode: str = Field(
        default="normal",
        description=(
            "City posture for this tick. One of: normal, emergency_focus, "
            "delivery_focus, defensive."
        ),
    )
    directive: str | None = Field(
        default=None,
        description=(
            "Optional free-form broadcast emitted by the planner this tick. "
            "Useful for chain-of-thought-style RL training."
        ),
    )


class CityObservation(Observation):
    """Post-tick snapshot.

    `done` and `reward` are inherited from the OpenEnv base. `reward` is set
    to the summed per-agent reward for the tick (positive contributions minus
    penalties, after verifier gating).
    """

    tick: int = 0
    seed: int = 0
    weather: str = "clear"
    avg_traffic: float = 0.0
    congestion_ratio: float = 0.0

    n_active_accidents: int = 0
    n_active_incidents: int = 0
    n_active_roadblocks: int = 0
    n_open_deliveries: int = 0
    n_completed_deliveries: int = 0
    n_failed_deliveries: int = 0

    city_score: float = 0.0
    per_agent_reward: dict[str, float] = Field(default_factory=dict)
    cumulative_per_agent: dict[str, float] = Field(default_factory=dict)
    priorities: dict[str, float] = Field(default_factory=dict)
    last_actions: dict[str, list[str]] = Field(default_factory=dict)
    recent_messages: list[str] = Field(default_factory=list)
    gated_any: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class CityNexusEnvState(State):
    """Episode metadata visible to the OpenEnv harness."""

    seed: int = 42
    max_ticks: int = 100
    cumulative_reward: float = 0.0
    mode_history: list[str] = Field(default_factory=list)
