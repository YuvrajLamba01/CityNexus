"""2D grid of zoned cells with a procedural city generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from citynexus.city.zones import Zone


Coord = tuple[int, int]


@dataclass(frozen=True)
class Cell:
    x: int
    y: int
    zone: Zone


class Grid:
    """Static spatial layout. Zones do not change after generation."""

    def __init__(self, width: int, height: int, cells: list[list[Cell]]) -> None:
        self.width = width
        self.height = height
        self._cells = cells

    def __getitem__(self, c: Coord) -> Cell:
        x, y = c
        return self._cells[y][x]

    def in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors4(self, c: Coord) -> list[Coord]:
        x, y = c
        return [n for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)) if self.in_bounds(n)]

    def is_road(self, c: Coord) -> bool:
        return self.in_bounds(c) and self[c].zone == Zone.ROAD

    def cells_of(self, zone: Zone) -> Iterator[Cell]:
        for row in self._cells:
            for cell in row:
                if cell.zone == zone:
                    yield cell

    def road_cells(self) -> Iterator[Cell]:
        return self.cells_of(Zone.ROAD)

    @classmethod
    def generate(
        cls,
        width: int,
        height: int,
        *,
        seed: int = 0,
        road_spacing: int = 4,
        hospital_count: int = 1,
        zone_weights: dict[Zone, float] | None = None,
    ) -> "Grid":
        """Procedurally generate a city: grid roads + zoned blocks.

        Roads are laid every `road_spacing` cells in both axes. The remaining
        non-road cells receive hospitals first, then the rest are sampled from
        residential / commercial / industrial by `zone_weights`.
        """
        rng = random.Random(seed)
        cells: list[list[Cell]] = [
            [Cell(x, y, Zone.EMPTY) for x in range(width)] for y in range(height)
        ]

        # Lay grid road network
        for y in range(height):
            for x in range(width):
                if x % road_spacing == 0 or y % road_spacing == 0:
                    cells[y][x] = Cell(x, y, Zone.ROAD)

        # Collect non-road blocks
        blocks: list[Coord] = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if cells[y][x].zone == Zone.EMPTY
        ]
        rng.shuffle(blocks)

        # Place hospitals first
        for i in range(min(hospital_count, len(blocks))):
            x, y = blocks[i]
            cells[y][x] = Cell(x, y, Zone.HOSPITAL)

        # Fill remaining blocks
        weights = zone_weights or {
            Zone.RESIDENTIAL: 0.55,
            Zone.COMMERCIAL: 0.30,
            Zone.INDUSTRIAL: 0.15,
        }
        choices = list(weights.keys())
        probs = list(weights.values())
        for x, y in blocks[hospital_count:]:
            zone = rng.choices(choices, weights=probs, k=1)[0]
            cells[y][x] = Cell(x, y, zone)

        return cls(width, height, cells)

    def render_ascii(self) -> str:
        glyphs = {
            Zone.EMPTY: ".",
            Zone.RESIDENTIAL: "R",
            Zone.COMMERCIAL: "C",
            Zone.HOSPITAL: "H",
            Zone.INDUSTRIAL: "I",
            Zone.ROAD: "#",
        }
        return "\n".join(
            "".join(glyphs[c.zone] for c in row) for row in self._cells
        )
