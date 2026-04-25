"""Inter-agent communication: typed message hierarchy + lightweight MessageBus.

Sending:
    ctx.send(RouteBlocked(sender=self.role, coords=[(5, 7)], reason="cordon"))

Receiving (type-filtered):
    for msg in ctx.receive(self.role, Advisory, RouteBlocked):
        ...

Lifecycle:
    Coordinator calls `bus.deliver(all_roles)` once per tick: pending messages
    are routed to per-role inboxes (broadcast or addressed), and inboxes are
    cleared at the start of each delivery so messages live exactly one tick.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Iterable

if TYPE_CHECKING:
    from citynexus.agents.base import AgentRole


# --- Channel & kind enums --------------------------------------------------

class Channel(str, Enum):
    BROADCAST = "broadcast"
    OPS = "ops"               # operational chatter (traffic, emergency, police)
    PLANNING = "planning"     # planner directives
    PUBLIC = "public"         # public-facing announcements


class MessageKind(str, Enum):
    GENERIC = "generic"
    ADVISORY = "advisory"
    ROUTE_BLOCKED = "route_blocked"
    EMERGENCY_PRIORITY = "emergency_priority"
    SIGNAL_CHANGE = "signal_change"
    CLEARANCE_REQUEST = "clearance_request"
    DIRECTIVE = "directive"
    STATUS_REPORT = "status_report"
    DISPATCH_NOTICE = "dispatch_notice"
    INCIDENT_REPORT = "incident_report"


# --- Base envelope ---------------------------------------------------------

@dataclass(kw_only=True)
class Message:
    """Base envelope. All inter-agent messages inherit from this.

    Subclasses set their own `kind` ClassVar and add typed payload fields.
    `recipients=None` means broadcast to every other role on the channel.
    """
    sender: "AgentRole"
    channel: Channel = Channel.OPS
    tick: int = 0
    recipients: tuple["AgentRole", ...] | None = None
    correlation_id: str | None = None
    body: str = ""

    kind: ClassVar[MessageKind] = MessageKind.GENERIC


# --- Typed message subclasses ----------------------------------------------

@dataclass(kw_only=True)
class Advisory(Message):
    """Traffic → Delivery: avoid these cells (congestion advisory)."""
    kind: ClassVar[MessageKind] = MessageKind.ADVISORY
    area: list[tuple[int, int]] = field(default_factory=list)
    severity: int = 1


@dataclass(kw_only=True)
class RouteBlocked(Message):
    """Inform other agents that road cells are blocked (accident, cordon, etc.)."""
    kind: ClassVar[MessageKind] = MessageKind.ROUTE_BLOCKED
    coords: list[tuple[int, int]] = field(default_factory=list)
    reason: str = ""
    severity: int = 1
    expires_at_tick: int | None = None


@dataclass(kw_only=True)
class EmergencyPriority(Message):
    """Planner → all: elevate emergency response priority (e.g. surge protocol)."""
    kind: ClassVar[MessageKind] = MessageKind.EMERGENCY_PRIORITY
    target: tuple[int, int] | None = None
    severity: int = 1
    protocol: str = "surge"


@dataclass(kw_only=True)
class SignalChange(Message):
    """Traffic → others: a road/intersection cell changed control state."""
    kind: ClassVar[MessageKind] = MessageKind.SIGNAL_CHANGE
    intersection: tuple[int, int] = (0, 0)
    state: str = "open"           # "open" | "blocked" | "advisory"
    ttl: int = 5


@dataclass(kw_only=True)
class ClearanceRequest(Message):
    """Emergency / Police → Traffic: please clear this cell."""
    kind: ClassVar[MessageKind] = MessageKind.CLEARANCE_REQUEST
    location: tuple[int, int] = (0, 0)
    requesting_for: str = ""        # accident id, incident id, …


@dataclass(kw_only=True)
class Directive(Message):
    """Planner → all: high-level command (free-form payload)."""
    kind: ClassVar[MessageKind] = MessageKind.DIRECTIVE
    protocol: str = ""
    payload: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class StatusReport(Message):
    """Any → Planner: status / capacity / completion summary."""
    kind: ClassVar[MessageKind] = MessageKind.STATUS_REPORT
    metrics: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class DispatchNotice(Message):
    """Emergency / Police → others: a unit is being dispatched to a target."""
    kind: ClassVar[MessageKind] = MessageKind.DISPATCH_NOTICE
    unit_id: str = ""
    target: tuple[int, int] = (0, 0)
    eta: int | None = None


@dataclass(kw_only=True)
class IncidentReport(Message):
    """Police → all: incident detected at a location."""
    kind: ClassVar[MessageKind] = MessageKind.INCIDENT_REPORT
    location: tuple[int, int] = (0, 0)
    incident_kind: str = ""
    severity: int = 1


# --- The bus ---------------------------------------------------------------

class MessageBus:
    """Per-tick publish/route bus. Lightweight: defaultdict + list under the hood."""

    def __init__(self) -> None:
        self._inbox: dict["AgentRole", list[Message]] = defaultdict(list)
        self._pending: list[Message] = []
        self._stats: dict[str, int] = {"sent": 0, "delivered": 0, "by_kind": {}}

    # ----- send -----------------------------------------------------------

    def send(self, msg: Message) -> None:
        self._pending.append(msg)
        self._stats["sent"] += 1
        by_kind = self._stats["by_kind"]
        by_kind[msg.kind.value] = by_kind.get(msg.kind.value, 0) + 1

    # ----- deliver --------------------------------------------------------

    def deliver(self, all_roles: Iterable["AgentRole"]) -> int:
        """Route pending messages → per-role inboxes. Old inboxes are cleared first.
        Returns the count of (recipient, message) pairs delivered.
        """
        roles = tuple(all_roles)
        for role in roles:
            self._inbox[role] = []
        delivered = 0
        for msg in self._pending:
            targets = msg.recipients if msg.recipients is not None else roles
            for r in targets:
                if r == msg.sender:
                    continue
                self._inbox[r].append(msg)
                delivered += 1
        self._pending.clear()
        self._stats["delivered"] += delivered
        return delivered

    # ----- receive --------------------------------------------------------

    def receive(self, role: "AgentRole", *types: type[Message]) -> list[Message]:
        """Get this tick's messages for `role`, optionally filtered by subclass."""
        msgs = self._inbox.get(role, [])
        if not types:
            return list(msgs)
        return [m for m in msgs if isinstance(m, types)]

    def receive_first(self, role: "AgentRole", *types: type[Message]) -> Message | None:
        for m in self.receive(role, *types):
            return m
        return None

    # ----- maintenance ----------------------------------------------------

    def clear(self) -> None:
        self._pending.clear()
        self._inbox.clear()
        self._stats = {"sent": 0, "delivered": 0, "by_kind": {}}

    @property
    def stats(self) -> dict:
        return {
            "sent": self._stats["sent"],
            "delivered": self._stats["delivered"],
            "by_kind": dict(self._stats["by_kind"]),
            "pending": len(self._pending),
        }
