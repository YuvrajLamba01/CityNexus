"""Delivery missions — owned by the world, acted on by DeliveryAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    EN_ROUTE = "en_route"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class Delivery:
    id: str
    origin: tuple[int, int]
    dest: tuple[int, int]
    spawned_tick: int
    deadline_tick: int
    priority: int = 1
    status: DeliveryStatus = DeliveryStatus.PENDING
    assigned_route: list[tuple[int, int]] = field(default_factory=list)
    route_progress: int = 0
    last_update_tick: int = 0

    @property
    def is_open(self) -> bool:
        return self.status in (
            DeliveryStatus.PENDING,
            DeliveryStatus.EN_ROUTE,
            DeliveryStatus.DELAYED,
        )

    @property
    def is_assigned(self) -> bool:
        return bool(self.assigned_route)

    def time_remaining(self, current_tick: int) -> int:
        return self.deadline_tick - current_tick
