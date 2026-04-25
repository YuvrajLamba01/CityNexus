"""Minimal gym-style space descriptions. Avoids a hard gym dependency.

Wrap with `gymnasium.spaces` if you want full gym compatibility for RL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from random import Random
from typing import Any

import numpy as np


class Space(ABC):
    @abstractmethod
    def sample(self, rng: Random) -> Any: ...

    @abstractmethod
    def contains(self, x: Any) -> bool: ...


class Discrete(Space):
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self, rng: Random) -> int:
        return rng.randrange(self.n)

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class MultiDiscrete(Space):
    def __init__(self, nvec: list[int]) -> None:
        self.nvec = [int(n) for n in nvec]

    def sample(self, rng: Random) -> list[int]:
        return [rng.randrange(n) for n in self.nvec]

    def contains(self, x: Any) -> bool:
        return (
            hasattr(x, "__len__")
            and len(x) == len(self.nvec)
            and all(0 <= int(xi) < n for xi, n in zip(x, self.nvec))
        )

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.nvec})"


class Box(Space):
    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> None:
        self.low = float(low)
        self.high = float(high)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: Random) -> np.ndarray:
        n = int(np.prod(self.shape))
        flat = np.array([rng.uniform(self.low, self.high) for _ in range(n)], dtype=self.dtype)
        return flat.reshape(self.shape)

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and bool(np.all((x >= self.low) & (x <= self.high)))
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


class DictSpace(Space):
    def __init__(self, spaces: dict[str, Space]) -> None:
        self.spaces = dict(spaces)

    def sample(self, rng: Random) -> dict[str, Any]:
        return {k: s.sample(rng) for k, s in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        return (
            isinstance(x, dict)
            and set(x.keys()) >= set(self.spaces.keys())
            and all(self.spaces[k].contains(x[k]) for k in self.spaces)
        )

    def __repr__(self) -> str:
        return f"DictSpace({list(self.spaces.keys())})"
