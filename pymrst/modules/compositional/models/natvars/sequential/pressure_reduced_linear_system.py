"""Port surface for MRST ``PressureReducedLinearSystem.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

from pymrst.modules.compositional.models.natvars.reduced_linearized_system import (
    ReducedLinearizedSystem,
)


@dataclass
class PressureReducedLinearSystem(ReducedLinearizedSystem):
    """Reduced pressure system with MRST pressure-step bookkeeping fields."""

    accumulationTerms: object = None
    flashEquations: object = None
    equations0: object = None
    assembled: bool = False
    wellVarIndices: object = None
    nwellvar: int = 0
    wellvars: object = None
    wellvarNames: object = None
    w: object = None
    model: object = None
    _base_assembled: bool = field(default=False, init=False)

    def assembleSystem(self):
        if not self.assembled:
            super().assembleSystem()
            self.assembled = True
            self._base_assembled = True
        return self

    assemble_system = assembleSystem


__all__ = ["PressureReducedLinearSystem"]
