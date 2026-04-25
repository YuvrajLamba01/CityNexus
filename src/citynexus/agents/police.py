"""PoliceAgent — partial obs: hazard areas (discs around incidents + accidents)."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from citynexus.agents.base import Action, AgentContext, AgentRole, BaseAgent, NoOp
from citynexus.agents.observability import PoliceView, build_police_view
from citynexus.agents.spaces import Box, DictSpace, Discrete, Space
from citynexus.env.world_state import WorldState


MAX_INCIDENTS_OBS = 6
FEAT_PER_INCIDENT = 5    # [x, y, kind_norm, severity_norm, assigned_flag]


# --- Action types ------------------------------------------------------------

@dataclass
class DispatchUnit(Action):
    unit_id: str = ""
    x: int = 0
    y: int = 0
    assigned_to: str = ""    # incident id
    kind: str = "dispatch_police"


@dataclass
class RecallUnit(Action):
    unit_id: str = ""
    kind: str = "recall_unit"


@dataclass
class EstablishCordon(Action):
    x: int = 0
    y: int = 0
    radius: int = 1
    ttl: int = 5
    kind: str = "establish_cordon"


POLICE_ACTION_TYPES: tuple[type[Action], ...] = (
    DispatchUnit,
    RecallUnit,
    EstablishCordon,
    NoOp,
)


# --- Agent -------------------------------------------------------------------

class PoliceAgent(BaseAgent):
    """Sees discs around active incidents (R_inc) plus smaller discs around
    accidents (R_haz) as traffic hazards."""

    def __init__(
        self,
        *,
        cordon_min_severity: int = 3,
        incident_radius: int = 4,
        hazard_radius: int = 2,
        rng: Random | None = None,
    ) -> None:
        super().__init__(AgentRole.POLICE, rng=rng)
        self.cordon_min_severity = cordon_min_severity
        self.incident_radius = incident_radius
        self.hazard_radius = hazard_radius

    @property
    def observation_space(self) -> Space:
        return DictSpace({
            "n_visible_incidents": Discrete(64),
            "n_idle_police": Discrete(16),
            "n_visible_hazards": Discrete(64),
            "avg_zone_traffic": Box(0.0, 1.0, (1,)),
            "incidents": Box(-1.0, 1.0, (MAX_INCIDENTS_OBS, FEAT_PER_INCIDENT)),
            "max_severity": Discrete(4),
        })

    @property
    def action_space(self) -> Space:
        return Discrete(len(POLICE_ACTION_TYPES))

    @property
    def action_types(self) -> tuple[type[Action], ...]:
        return POLICE_ACTION_TYPES

    def observe(self, world: WorldState, ctx: AgentContext) -> dict:
        view = build_police_view(
            world, ctx,
            incident_radius=self.incident_radius,
            hazard_radius=self.hazard_radius,
        )
        W, H = world.grid.width, world.grid.height
        wnorm, hnorm = max(1, W - 1), max(1, H - 1)

        incs = sorted(
            view.incidents,
            key=lambda i: (-i.severity, i.spawned_tick),
        )[:MAX_INCIDENTS_OBS]

        kind_to_idx = {"disturbance": 0, "theft": 1, "protest": 2}
        feats = np.full((MAX_INCIDENTS_OBS, FEAT_PER_INCIDENT), -1.0, dtype=np.float32)
        for i, inc in enumerate(incs):
            feats[i, 0] = inc.pos[0] / wnorm
            feats[i, 1] = inc.pos[1] / hnorm
            feats[i, 2] = kind_to_idx.get(inc.kind.value, 0) / 2.0
            feats[i, 3] = inc.severity / 3.0
            feats[i, 4] = 1.0 if inc.assigned_unit else 0.0

        idle = sum(1 for u in view.police if u.is_idle)
        max_sev = max((i.severity for i in incs), default=0)
        avg_zt = float(np.mean(list(view.traffic_in_zones.values()))) if view.traffic_in_zones else 0.0
        return {
            "n_visible_incidents": min(63, len(view.incidents)),
            "n_idle_police": min(15, idle),
            "n_visible_hazards": min(63, len(view.accident_hazards)),
            "avg_zone_traffic": np.array([avg_zt], dtype=np.float32),
            "incidents": feats,
            "max_severity": min(3, max_sev),
            "_view": view,
            "_incidents_ranked": incs,
        }

    def decide(self, obs: dict, ctx: AgentContext) -> list[Action]:
        view: PoliceView = obs.get("_view")
        if view is None:
            return [NoOp()]

        actions: list[Action] = []
        incs = obs.get("_incidents_ranked", [])
        idle = [u for u in view.police if u.is_idle]

        for inc in incs:
            if inc.assigned_unit:
                continue
            if not idle:
                break
            best = min(idle, key=lambda u: abs(u.pos[0] - inc.pos[0]) + abs(u.pos[1] - inc.pos[1]))
            actions.append(
                DispatchUnit(unit_id=best.id, x=inc.pos[0], y=inc.pos[1], assigned_to=inc.id)
            )
            inc.assigned_unit = best.id
            idle = [u for u in idle if u.id != best.id]
            if inc.severity >= self.cordon_min_severity and inc.kind.value == "protest":
                actions.append(
                    EstablishCordon(x=inc.pos[0], y=inc.pos[1], radius=2, ttl=10)
                )

        if not actions:
            actions.append(NoOp())
        return actions
