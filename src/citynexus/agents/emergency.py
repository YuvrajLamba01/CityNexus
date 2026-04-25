"""EmergencyAgent — partial obs: discs around active accidents (incident zones)."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent, NoOp
from citynexus.agents.observability import EmergencyView, build_emergency_view
from citynexus.agents.spaces import Box, DictSpace, Discrete, Space
from citynexus.env.world_state import WorldState


MAX_ACCIDENTS_OBS = 8
FEAT_PER_ACCIDENT = 5    # [x, y, severity_norm, ttl_norm, assigned_flag]
MAX_UNITS_OBS = 6
FEAT_PER_UNIT = 5        # [x, y, status_norm, target_x, target_y]


# --- Action types ------------------------------------------------------------

@dataclass
class DispatchUnit(Action):
    unit_id: str = ""
    x: int = 0
    y: int = 0
    assigned_to: str = ""    # accident@x,y
    kind: str = "dispatch_ambulance"


@dataclass
class RecallUnit(Action):
    unit_id: str = ""
    kind: str = "recall_unit"


@dataclass
class RequestClearance(Action):
    x: int = 0
    y: int = 0
    kind: str = "request_clearance"


EMERGENCY_ACTION_TYPES: tuple[type[Action], ...] = (
    DispatchUnit,
    RecallUnit,
    RequestClearance,
    NoOp,
)


# --- Agent -------------------------------------------------------------------

class EmergencyAgent(BaseAgent):
    """Sees only zones around active accidents. Dispatches nearest idle ambulance."""

    def __init__(
        self,
        *,
        zone_radius: int = 5,
        rng: Random | None = None,
    ) -> None:
        super().__init__(AgentRole.EMERGENCY, rng=rng)
        self.zone_radius = zone_radius

    @property
    def observation_space(self) -> Space:
        return DictSpace({
            "weather": Discrete(3),
            "n_visible_accidents": Discrete(64),
            "n_idle_ambulances": Discrete(16),
            "avg_zone_traffic": Box(0.0, 1.0, (1,)),
            "n_zone_blocks": Discrete(64),
            "accidents": Box(-1.0, 1.0, (MAX_ACCIDENTS_OBS, FEAT_PER_ACCIDENT)),
            "units": Box(-1.0, 1.0, (MAX_UNITS_OBS, FEAT_PER_UNIT)),
        })

    @property
    def action_space(self) -> Space:
        return Discrete(len(EMERGENCY_ACTION_TYPES))

    @property
    def action_types(self) -> tuple[type[Action], ...]:
        return EMERGENCY_ACTION_TYPES

    def observe(self, world: WorldState, ctx: AgentContext) -> dict:
        view = build_emergency_view(world, ctx, radius=self.zone_radius)
        W, H = world.grid.width, world.grid.height
        wnorm, hnorm = max(1, W - 1), max(1, H - 1)

        accs = sorted(view.accidents, key=lambda a: (-int(a.severity), a.ttl))[:MAX_ACCIDENTS_OBS]
        targeted = {u.target for u in view.ambulances if u.target is not None}

        acc_feats = np.full((MAX_ACCIDENTS_OBS, FEAT_PER_ACCIDENT), -1.0, dtype=np.float32)
        for i, a in enumerate(accs):
            acc_feats[i, 0] = a.x / wnorm
            acc_feats[i, 1] = a.y / hnorm
            acc_feats[i, 2] = int(a.severity) / 3.0
            acc_feats[i, 3] = min(1.0, a.ttl / 12.0)
            acc_feats[i, 4] = 1.0 if a.coord in targeted else 0.0

        ambulances = view.ambulances[:MAX_UNITS_OBS]
        status_to_idx = {"idle": 0, "en_route": 1, "on_scene": 2, "returning": 3}
        unit_feats = np.full((MAX_UNITS_OBS, FEAT_PER_UNIT), -1.0, dtype=np.float32)
        for i, u in enumerate(ambulances):
            unit_feats[i, 0] = u.pos[0] / wnorm
            unit_feats[i, 1] = u.pos[1] / hnorm
            unit_feats[i, 2] = status_to_idx.get(u.status.value, 0) / 3.0
            if u.target is not None:
                unit_feats[i, 3] = u.target[0] / wnorm
                unit_feats[i, 4] = u.target[1] / hnorm

        weather_to_idx = {"clear": 0, "rain": 1, "storm": 2}
        idle = sum(1 for u in view.ambulances if u.is_idle)
        avg_zt = float(np.mean(list(view.traffic_in_zones.values()))) if view.traffic_in_zones else 0.0
        return {
            "weather": weather_to_idx[view.weather.value],
            "n_visible_accidents": min(63, len(view.accidents)),
            "n_idle_ambulances": min(15, idle),
            "avg_zone_traffic": np.array([avg_zt], dtype=np.float32),
            "n_zone_blocks": min(63, len(view.blocks_in_zones)),
            "accidents": acc_feats,
            "units": unit_feats,
            "_view": view,
            "_accidents_ranked": accs,
        }

    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]:
        view: EmergencyView = obs.get("_view")
        if view is None:
            return [NoOp()]

        actions: list[Action] = []
        accs = obs.get("_accidents_ranked", [])
        idle = [u for u in view.ambulances if u.is_idle]
        targeted = {u.target for u in view.ambulances if u.target is not None}

        for a in accs:
            if a.coord in targeted:
                continue
            if not idle:
                break
            best = min(idle, key=lambda u: abs(u.pos[0] - a.x) + abs(u.pos[1] - a.y))
            actions.append(
                DispatchUnit(unit_id=best.id, x=a.x, y=a.y, assigned_to=f"accident@{a.x},{a.y}")
            )
            actions.append(RequestClearance(x=a.x, y=a.y))
            idle = [u for u in idle if u.id != best.id]

        if not actions:
            actions.append(NoOp())
        return actions
