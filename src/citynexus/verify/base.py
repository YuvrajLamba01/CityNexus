"""Check ABC and the layered Verifier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from citynexus.verify.schemas import (
    CheckResult,
    CheckStatus,
    VerificationContext,
    VerificationReport,
)


class Check(ABC):
    """One verification rule. Subclass and implement `evaluate`.

    `name`/`layer` are class-level identifiers; `attributed_to` (optional) lists
    the agent role values this check is judging.
    """

    name: str = "unnamed"
    layer: str = "programmatic"
    attributed_to: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, ctx: VerificationContext) -> CheckResult: ...

    def _result(
        self,
        status: CheckStatus,
        *,
        score: float = 1.0,
        reason: str = "",
        metadata: dict | None = None,
    ) -> CheckResult:
        return CheckResult(
            name=self.name,
            layer=self.layer,
            status=status,
            score=score,
            reason=reason,
            attributed_to=self.attributed_to,
            metadata=metadata or {},
        )


class Verifier:
    """Composes the three verification layers. Returns one VerificationReport per tick.

    Layers run in order: programmatic → system_state → semantic. By default all
    three are evaluated every tick; pass empty lists to disable any layer.
    """

    def __init__(
        self,
        *,
        programmatic: Iterable[Check] | None = None,
        system_state: Iterable[Check] | None = None,
        semantic: Iterable[Check] | None = None,
    ) -> None:
        self.programmatic: list[Check] = list(programmatic or [])
        self.system_state: list[Check] = list(system_state or [])
        self.semantic: list[Check] = list(semantic or [])

    def verify(self, ctx: VerificationContext) -> VerificationReport:
        results: list[CheckResult] = []
        for check in self.programmatic:
            results.append(self._safe_eval(check, ctx))
        for check in self.system_state:
            results.append(self._safe_eval(check, ctx))
        for check in self.semantic:
            results.append(self._safe_eval(check, ctx))
        return VerificationReport(tick=ctx.tick, results=results)

    @staticmethod
    def _safe_eval(check: Check, ctx: VerificationContext) -> CheckResult:
        try:
            return check.evaluate(ctx)
        except Exception as e:
            # A check raising is itself a failure — don't let it crash training.
            return CheckResult(
                name=check.name, layer=check.layer,
                status=CheckStatus.FAIL,
                score=0.0,
                reason=f"check raised {type(e).__name__}: {e}",
                attributed_to=check.attributed_to,
            )

    @classmethod
    def default(cls) -> "Verifier":
        """Standard verifier with the canonical check set for all three layers."""
        from citynexus.verify.programmatic import default_programmatic_checks
        from citynexus.verify.semantic import default_semantic_checks
        from citynexus.verify.system_state import default_system_state_checks
        return cls(
            programmatic=default_programmatic_checks(),
            system_state=default_system_state_checks(),
            semantic=default_semantic_checks(),
        )
