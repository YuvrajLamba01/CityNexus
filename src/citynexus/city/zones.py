"""Zone taxonomy for the CITYNEXUS grid."""

from __future__ import annotations

from enum import Enum


class Zone(str, Enum):
    EMPTY = "empty"
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    HOSPITAL = "hospital"
    INDUSTRIAL = "industrial"
    ROAD = "road"


BUILDING_ZONES: frozenset[Zone] = frozenset(
    {Zone.RESIDENTIAL, Zone.COMMERCIAL, Zone.HOSPITAL, Zone.INDUSTRIAL}
)
