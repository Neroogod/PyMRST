"""Port surface for MRST ``PressureReducedLinearSystem.m``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from pymrst.modules.compositional.models.natvars.reduced_linearized_system import (
    ReducedLinearizedSystem,
)
from pymrst.modules.compositional.utils.get_partial_volumes import getPartialVolumes
from pymrst.modules.compositional.models.natvars.sequential.utils.get_updates_from_pressure import (
    get_updates_from_pressure,
)
from pymrst.utils.adi import combine_equations, value


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
    state: object = None
    iterationNo: float = np.nan
    equations: object = None
    types: object = None
    primaryVariables: object = None
    _base_assembled: bool = field(default=False, init=False)

    def assembleSystem(self):
        if not self.assembled:
            super().assembleSystem()
            self._base_assembled = True
            if (
                self.A is not None
                and self.b is not None
                and self.accumulationTerms is not None
                and self.model is not None
                and self.state is not None
            ):
                self._assemble_pressure_reduction()
            self.assembled = True
        return self

    def _assemble_pressure_reduction(self) -> None:
        """Collapse component balances to MRST's weighted pressure equation."""
        model = self.model
        well_indices = np.asarray(
            [] if self.wellVarIndices is None else self.wellVarIndices,
            dtype=np.int64,
        ).reshape(-1)
        weights, dwdp = getPartialVolumes(
            model,
            self.state,
            self.accumulationTerms,
            reorder=self.reorder,
            wellVarIndices=well_indices,
            iteration=self.iterationNo,
            singlePhaseStrategy=getattr(model, "singlePhaseStrategy", "numerical"),
            twoPhaseStrategy=getattr(model, "twoPhaseStrategy", "numerical"),
            singlePhaseDifferentiation=getattr(
                model, "singlePhaseDifferentiation", "numerical"
            ),
            twoPhaseDifferentiation=getattr(
                model, "twoPhaseDifferentiation", "numerical"
            ),
        )
        self.w = weights
        A0 = sp.csr_matrix(self.A)
        b0 = np.asarray(self.b, dtype=float).reshape(-1)
        ncell, ncomp = weights.shape
        if A0.shape[0] < ncell * ncomp or b0.size < ncell * ncomp:
            raise ValueError("reduced system does not contain every component balance row")
        if well_indices.size and (
            well_indices.min() < 0
            or well_indices.max() >= min(A0.shape[0], A0.shape[1])
        ):
            raise IndexError("wellVarIndices is outside the reduced pressure system")
        retained = np.concatenate([np.arange(ncell, dtype=np.int64), well_indices])
        pressure_matrix = sp.csr_matrix((ncell, retained.size), dtype=float)
        pressure_rhs = np.zeros(ncell, dtype=float)
        for component in range(ncomp):
            rows = np.arange(ncell, dtype=np.int64) + component * ncell
            wi = np.asarray(weights[:, component], dtype=float)
            pressure_matrix = pressure_matrix + sp.diags(wi) @ A0[rows][:, retained]
            pressure_rhs += wi * b0[rows]
            if dwdp is not None:
                derivative = np.asarray(dwdp[:, component], dtype=float)
                derivative_matrix = sp.hstack(
                    [
                        sp.diags(derivative),
                        sp.csr_matrix((ncell, well_indices.size)),
                    ],
                    format="csr",
                )
                pressure_matrix = pressure_matrix - sp.diags(b0[rows]) @ derivative_matrix
        if well_indices.size:
            self.A = sp.vstack(
                [pressure_matrix, A0[well_indices][:, retained]], format="csr"
            )
            self.b = np.concatenate([pressure_rhs, b0[well_indices]])
        else:
            self.A = pressure_matrix.tocsr()
            self.b = pressure_rhs
        self.B = self.A
        self.C = self.D = self.E = self.h = None
        self.f = self.b

    assemble_system = assembleSystem

    def processResultAfterSolve(self, pressure_update, report=None):
        """Reconstruct natural-variable increments after the pressure solve."""
        if (
            self.model is None
            or self.state is None
            or self.primaryVariables is None
            or self.equations is None
        ):
            return super().processResultAfterSolve(pressure_update, report)
        model = self.model
        nc = np.asarray(self.state.pressure).size
        reduced = np.asarray(pressure_update, dtype=float).reshape(-1)
        if reduced.size < nc:
            raise ValueError("pressure update has fewer entries than reservoir cells")
        dpressure = reduced[:nc]
        dx, dy, ds, _dL, two_phase, _weight = get_updates_from_pressure(
            model, self.state, dpressure
        )

        first_ad = next(
            (eq for eq in self.equations if hasattr(eq, "jac") and eq.jac),
            None,
        )
        if first_ad is None:
            raise ValueError("natural-variable reconstruction requires AD equations")
        block_sizes = [block.shape[1] for block in first_ad.jac]
        if len(block_sizes) != len(self.primaryVariables):
            raise ValueError("primary variable names do not match AD Jacobian blocks")
        offsets = np.cumsum([0, *block_sizes])
        full = np.zeros(int(offsets[-1]), dtype=float)
        names = tuple(str(name) for name in model.EOSModel.getComponentNames())
        vapor_names = {f"v_{name}": i for i, name in enumerate(names)}
        liquid_names = {f"w_{name}": i for i, name in enumerate(names)}
        water = bool(getattr(model, "water", False))

        for index, raw_name in enumerate(self.primaryVariables):
            name = str(raw_name)
            target = slice(int(offsets[index]), int(offsets[index + 1]))
            size = block_sizes[index]
            if name == "pressure":
                update = dpressure
            elif name in {"sl", "sato"}:
                update = ds[:, 1 if water else 0]
            elif name in {"sv", "satg"}:
                update = ds[:, 2 if water else 1]
            elif name in {"sw", "satw"}:
                update = np.zeros(nc)
                if np.any(two_phase):
                    update[two_phase] = ds[:, 0]
            elif name in liquid_names:
                component = liquid_names[name]
                update = dy[:, component] if np.any(two_phase) else np.empty(0)
            elif name in vapor_names:
                component = vapor_names[name]
                update = np.zeros(nc)
                if np.any(two_phase):
                    update[two_phase] = dx[:, component]
            else:
                continue
            update = np.asarray(update, dtype=float).reshape(-1)
            if update.size != size:
                raise ValueError(
                    f"reconstructed {name!r} update has {update.size} values, expected {size}"
                )
            full[target] = update

        well_indices = np.asarray(
            [] if self.wellVarIndices is None else self.wellVarIndices,
            dtype=np.int64,
        ).reshape(-1)
        equation_types = [] if self.types is None else list(self.types)
        well_equations = [
            eq
            for eq, kind in zip(self.equations, equation_types)
            if str(kind).lower() in {"perf", "well"} and hasattr(eq, "jac")
        ]
        if well_indices.size and well_equations:
            combined = combine_equations(*well_equations)
            Aw = -combined.jac[0].tocsr()
            residual = np.asarray(value(combined), dtype=float).reshape(-1)
            rhs = residual - Aw @ full
            well_matrix = Aw[:, well_indices]
            if well_matrix.shape[0] == well_matrix.shape[1]:
                well_update = np.asarray(
                    sp.linalg.spsolve(well_matrix.tocsc(), rhs), dtype=float
                ).reshape(-1)
            else:
                well_update = np.linalg.lstsq(
                    well_matrix.toarray(), rhs, rcond=None
                )[0]
            full[well_indices] = well_update
        return full, report

    process_result_after_solve = processResultAfterSolve


__all__ = ["PressureReducedLinearSystem"]
