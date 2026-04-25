"""Mobile units that agents dispatch (ambulances, police cars, delivery vans)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UnitKind(str, Enum):
    AMBULANCE = "ambulance"
    POLICE_CAR = "police_car"
    DELIVERY_VAN = "delivery_van"


class UnitStatus(str, Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    RETURNING = "returning"


@dataclass
class ResponderUnit:
    id: str
    kind: UnitKind
    home: tuple[int, int]
    pos: tuple[int, int]
    status: UnitStatus = UnitStatus.IDLE
    target: tuple[int, int] | None = None
    eta_tick: int | None = None
    assigned_to: str | None = None      # id of accident / incident / delivery

    @property
    def is_idle(self) -> bool:
        return self.status == UnitStatus.IDLE
