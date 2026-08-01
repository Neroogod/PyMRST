"""Linear compressibility flow model from MRST `vemmech`.

MRST's `LinearCompressibilityModel` is a small `ReservoirModel` subclass whose
step function delegates to `lincompTPFA`: a water-only, weakly compressible,
linear TPFA pressure solve. This Python counterpart keeps that standalone class
surface instead of routing users to the poroelastic `BiotModel`.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well
from pymrst.solvers.state import State, WellSol
from pymrst.solvers.trans import compute_trans, get_face_transmissibility


@dataclass
class LinearCompressibilityReport:
    """MRST-style step report for a linear step."""

    Failure: bool = False
    Converged: bool = True
    Residuals: float = 0.0
    ResidualsConverged: bool = True


class LinearCompressibilityModel:
    """Water-only weakly-compressible TPFA model.

    Parameters mirror MRST's constructor `(G, rock, fluid, ...)` as closely as
    Python allows. The fluid must expose a single-phase viscosity as `mu[0]`;
    optional compressibility is read from `fluid.cf`, and optional rock
    compressibility from `rock.cr`. Both default to zero, matching MRST.
    """

    def __init__(self, g: Grid, rock: Rock, fluid: Any, **kwargs):
        self.G = g
        self.rock = rock
        self.fluid = fluid
        self.thermal = bool(kwargs.pop("thermal", False))
        self.oil = False
        self.gas = False
        self.water = True
        self.useCNVConvergence = False
        self.stepFunctionIsLinear = True
        self.operators = SimpleNamespace(
            T_all=get_face_transmissibility(g, rock),
            T_half=compute_trans(g, rock),
            pv=pore_volume(g, rock),
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

    def make_step_report(self, **kwargs) -> LinearCompressibilityReport:
        data = LinearCompressibilityReport()
        for key, value in kwargs.items():
            setattr(data, key, value)
        return data

    def step_function(
        self,
        state: State,
        state0: State,
        dt: float,
        driving_forces: Optional[Any] = None,
        linsolve: Optional[Any] = None,
        nonlinsolve: Optional[Any] = None,
        iteration: int = 1,
        **kwargs,
    ) -> tuple[State, LinearCompressibilityReport]:
        """MRST-style `stepFunction` entry point."""

        if self.thermal:
            raise NotImplementedError(
                "LinearCompressibilityModel thermal linearTransport branch is "
                "not ported; set thermal=False for the validated flow branch."
            )
        forces = _driving_forces(driving_forces)
        lin_solve = _linear_solver(linsolve)
        out = lincomp_tpfa(
            dt,
            state0,
            self.G,
            self.operators.T_half,
            self.operators.pv,
            self.fluid,
            self.rock,
            bc=forces.bc,
            src=forces.src,
            wells=forces.W,
            linear_solver=lin_solve,
            matrix_output=bool(kwargs.get("MatrixOutput", False)),
        )
        report = self.make_step_report(
            Failure=False,
            Converged=True,
            Residuals=0.0,
            ResidualsConverged=True,
        )
        return out, report

    def step(
        self,
        state0: State,
        dt: float,
        driving_forces: Optional[Any] = None,
        **kwargs,
    ) -> State:
        """Convenience Python entry point returning only the updated state."""

        state, _ = self.step_function(
            state0, state0, dt, driving_forces=driving_forces, **kwargs
        )
        return state


def lincomp_tpfa(
    dt: float,
    state0: State,
    g: Grid,
    t_half: np.ndarray,
    pv: np.ndarray,
    fluid: Any,
    rock: Rock,
    *,
    bc: Optional[BoundaryCondition] = None,
    src: Optional[Source] = None,
    wells: Optional[Sequence[Well]] = None,
    linear_solver=None,
    matrix_output: bool = False,
) -> State:
    """Weakly-compressible single-phase TPFA solve.

    This is the implemented subset of MRST `lincompTPFA` used by
    `LinearCompressibilityModel`: pressure/rate sources, pressure/flux
    boundaries, bhp/rate wells, and the linear accumulation term
    `(rock.cr + fluid.cf) * pv / dt`.
    """

    wells = list(wells or [])
    mu = _scalar_attr(fluid, "mu", default=1.0)
    face_t = _face_trans(g, t_half) / mu
    nc = g.cells.num
    nw = len(wells)
    n = nc + nw
    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n)
    diag = np.zeros(nc)

    ni = neighbors[internal]
    tf = face_t[internal]
    diag += np.bincount(ni.ravel(order="F"), weights=np.tile(tf, 2), minlength=nc)
    rows.extend(ni[:, 0].tolist())
    cols.extend(ni[:, 1].tolist())
    vals.extend((-tf).tolist())
    rows.extend(ni[:, 1].tolist())
    cols.extend(ni[:, 0].tolist())
    vals.extend((-tf).tolist())

    if src is not None and src.cell.size:
        np.add.at(rhs, src.cell, src.rate)

    pressure_bc = np.zeros(g.faces.num, dtype=bool)
    pressure_val = np.zeros(g.faces.num)
    flux_bc = np.zeros(g.faces.num, dtype=bool)
    flux_val = np.zeros(g.faces.num)
    if bc is not None and bc.face.size:
        types = np.asarray(bc.type)
        pressure_bc[bc.face[types == "pressure"]] = True
        pressure_val[bc.face[types == "pressure"]] = bc.value[types == "pressure"]
        flux_bc[bc.face[types == "flux"]] = True
        flux_val[bc.face[types == "flux"]] = bc.value[types == "flux"]

    boundary = np.any(neighbors < 0, axis=1)
    bc_faces = np.flatnonzero(boundary & pressure_bc)
    if bc_faces.size:
        cells = _boundary_cells(g, bc_faces)
        tb = face_t[bc_faces]
        np.add.at(diag, cells, tb)
        np.add.at(rhs, cells, tb * pressure_val[bc_faces])

    flux_faces = np.flatnonzero(boundary & flux_bc)
    if flux_faces.size:
        cells = _boundary_cells(g, flux_faces)
        np.add.at(rhs, cells, flux_val[flux_faces])

    d_well = np.zeros(nw)
    c_rows: list[int] = []
    c_cols: list[int] = []
    c_vals: list[float] = []
    for k, well in enumerate(wells):
        wc = np.asarray(well.cells, dtype=np.int64)
        wi = np.asarray(well.WI, dtype=float).ravel() / mu
        wrow = nc + k
        np.add.at(diag, wc, wi)
        if well.type == "bhp":
            ww = float(np.max(wi))
            rhs[wrow] += ww * well.val
            rhs[wc] += wi * well.val
            d_well[k] = ww
        elif well.type == "rate":
            rhs[wrow] += well.val
            c_rows.extend([k] * wc.size)
            c_cols.extend(wc.tolist())
            c_vals.extend((-wi).tolist())
            d_well[k] = float(np.sum(wi))
        else:
            raise ValueError(f"unsupported well control '{well.type}'")

    cr = _cell_compressibility(rock, "cr", nc) + _cell_compressibility(fluid, "cf", nc)
    ct = cr * np.asarray(pv, dtype=float).ravel() / float(dt)
    diag += ct
    rhs[:nc] += ct * np.asarray(state0.pressure, dtype=float).ravel()

    rows.extend(np.arange(nc).tolist())
    cols.extend(np.arange(nc).tolist())
    vals.extend(diag.tolist())
    A = sp.coo_array((vals, (rows, cols)), shape=(nc, nc)).tocsr()
    if nw:
        C = sp.coo_array((c_vals, (c_rows, c_cols)), shape=(nw, nc)).tocsr()
        A = sp.block_array(
            [[A, C.T], [C, sp.diags_array(d_well, format="csr")]],
            format="csr",
        )

    if not (bc_faces.size or any(w.type == "bhp" for w in wells) or np.any(ct > 0)):
        A = A.tolil()
        j = int(np.argmax(A.diagonal()))
        A[j, j] = 2.0 * A[j, j] if A[j, j] > 0 else 1.0
        A = A.tocsr()

    p = linear_solver(A, rhs) if linear_solver is not None else spla.spsolve(A.tocsc(), rhs)
    out = State(
        pressure=np.asarray(p[:nc], dtype=float),
        flux=np.zeros(g.faces.num),
        s=np.asarray(state0.s, dtype=float).copy(),
        wellSol=[],
    )
    out.flux[internal] = -face_t[internal] * (out.pressure[ni[:, 1]] - out.pressure[ni[:, 0]])
    if bc_faces.size:
        cells = _boundary_cells(g, bc_faces)
        bsgn = np.where(g.faces.neighbors[bc_faces, 1] < 0, 1.0, -1.0)
        out.flux[bc_faces] = -bsgn * face_t[bc_faces] * (
            pressure_val[bc_faces] - out.pressure[cells]
        )
    if flux_faces.size:
        out.flux[flux_faces] = flux_val[flux_faces]
    out.facePressure = _face_pressure(g, out.pressure, pressure_bc, pressure_val)
    for k, well in enumerate(wells):
        wc = np.asarray(well.cells, dtype=np.int64)
        wi = np.asarray(well.WI, dtype=float).ravel() / mu
        out.wellSol.append(WellSol(
            flux=wi * (p[nc + k] - out.pressure[wc]),
            pressure=float(p[nc + k]),
        ))
    if matrix_output:
        out.A = A
        out.rhs = rhs
        # the system carries one row per well control on top of the cells, and
        # those rows hold no accumulation, so the diagonal is padded to match
        accumulation = np.zeros(A.shape[0])
        accumulation[:nc] = cr * np.asarray(pv, dtype=float).ravel()
        out.ct = sp.diags_array(accumulation, shape=A.shape, format="csr")
    return out


def _driving_forces(driving_forces: Optional[Any]) -> SimpleNamespace:
    if driving_forces is None:
        return SimpleNamespace(src=None, bc=None, W=[])
    if isinstance(driving_forces, dict):
        return SimpleNamespace(
            src=driving_forces.get("src"),
            bc=driving_forces.get("bc"),
            W=driving_forces.get("W", driving_forces.get("wells", [])),
        )
    return SimpleNamespace(
        src=getattr(driving_forces, "src", None),
        bc=getattr(driving_forces, "bc", None),
        W=getattr(driving_forces, "W", getattr(driving_forces, "wells", [])),
    )


def _linear_solver(linsolve):
    if linsolve is None:
        return None
    if callable(linsolve):
        return linsolve
    if hasattr(linsolve, "solveLinearSystem"):
        return linsolve.solveLinearSystem
    if hasattr(linsolve, "solve_linear_system"):
        return linsolve.solve_linear_system
    return None


def _scalar_attr(obj: Any, name: str, default: float) -> float:
    val = getattr(obj, name, default)
    arr = np.asarray(val, dtype=float).ravel()
    return float(arr[0]) if arr.size else float(default)


def _cell_compressibility(obj: Any, name: str, nc: int) -> np.ndarray:
    val = getattr(obj, name, 0.0)
    arr = np.asarray(val, dtype=float).ravel()
    if arr.size == 1:
        return np.full(nc, arr[0])
    if arr.size != nc:
        raise ValueError(f"{name} must be scalar or one value per cell")
    return arr


def _face_trans(g: Grid, t_half: np.ndarray) -> np.ndarray:
    cf = g.cells.faces
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / np.bincount(cf, weights=1.0 / t_half, minlength=g.faces.num)
    out[~np.isfinite(out)] = 0.0
    return out


def _boundary_cells(g: Grid, faces: np.ndarray) -> np.ndarray:
    nb = g.faces.neighbors[np.asarray(faces, dtype=np.int64)]
    return np.where(nb[:, 0] >= 0, nb[:, 0], nb[:, 1])


def _face_pressure(
    g: Grid,
    p: np.ndarray,
    pressure_bc: np.ndarray,
    pressure_val: np.ndarray,
) -> np.ndarray:
    fp = np.zeros(g.faces.num)
    n = g.faces.neighbors
    internal = np.all(n >= 0, axis=1)
    fp[internal] = 0.5 * (p[n[internal, 0]] + p[n[internal, 1]])
    boundary = np.any(n < 0, axis=1)
    bfaces = np.flatnonzero(boundary)
    if bfaces.size:
        cells = _boundary_cells(g, bfaces)
        fp[bfaces] = p[cells]
    fp[pressure_bc] = pressure_val[pressure_bc]
    return fp


__all__ = [
    "LinearCompressibilityModel",
    "LinearCompressibilityReport",
    "lincomp_tpfa",
]
