"""Well-pair diagnostics (MRST ``computeWellPairs.m`` layout)."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.solvers.state import State


@dataclass
class Diagnostics:
    """Forward/backward time-of-flight and well tracer partitions."""

    tof_forward: np.ndarray
    tof_backward: np.ndarray
    tracer_inj: np.ndarray
    tracer_prod: np.ndarray
    inj_wells: List[int]
    prod_wells: List[int]

    @property
    def total_tof(self) -> np.ndarray:
        return self.tof_forward + self.tof_backward

    def injector_region(self) -> np.ndarray:
        return np.argmax(self.tracer_inj, axis=1)

    def producer_region(self) -> np.ndarray:
        return np.argmax(self.tracer_prod, axis=1)


def _well_source(g: Grid, state: State, wells: Sequence[Well]) -> np.ndarray:
    q = np.zeros(g.cells.num)
    for k, w in enumerate(wells):
        np.add.at(q, w.cells, state.wellSol[k].flux)
    return q


def _tof_matrix(
    g: Grid,
    flux: np.ndarray,
    q: np.ndarray,
) -> Tuple[sp.csr_array, np.ndarray]:
    nc = g.cells.num
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    n = g.faces.neighbors[internal]
    f = flux[internal]

    into_2 = np.maximum(f, 0.0)
    into_1 = -np.minimum(f, 0.0)
    qp = np.maximum(q, 0.0)

    rows = np.concatenate([n[:, 1], n[:, 0]])
    cols = np.concatenate([n[:, 0], n[:, 1]])
    vals = np.concatenate([into_2, into_1])

    d = np.bincount(rows, weights=vals, minlength=nc) + qp
    a = sp.csr_array((vals, (rows, cols)), shape=(nc, nc))
    return (-a + sp.diags_array(d, format="csr")).tocsr(), qp


def compute_diagnostics(
    g: Grid,
    rock: Rock,
    state: State,
    wells: Sequence[Well],
    max_tof: Optional[float] = None,
) -> Diagnostics:
    """Compute diagnostics used to derive injector/producer well pairs."""
    pv = pore_volume(g, rock)
    q = _well_source(g, state, wells)

    inj = [k for k in range(len(wells)) if state.wellSol[k].flux.sum() > 0]
    prod = [k for k in range(len(wells)) if state.wellSol[k].flux.sum() < 0]
    assert inj and prod, "diagnostics need an injector and a producer"

    def solve(flux: np.ndarray, qq: np.ndarray, sources: List[int]):
        a, qp = _tof_matrix(g, flux, qq)
        rhs = [pv]
        for k in sources:
            r = np.zeros(g.cells.num)
            r[wells[k].cells] = qp[wells[k].cells]
            rhs.append(r)
        x = spla.spsolve(a.tocsc(), np.column_stack(rhs))
        return x[:, 0], x[:, 1:]

    tof_f, tr_i = solve(state.flux, q, inj)
    tof_b, tr_p = solve(-state.flux, -q, prod)

    if max_tof is None:
        max_tof = 50.0 * pv.sum() / max(np.maximum(q, 0.0).sum(), 1e-30)
    tof_f = np.clip(tof_f, 0.0, max_tof)
    tof_b = np.clip(tof_b, 0.0, max_tof)

    return Diagnostics(
        tof_forward=tof_f,
        tof_backward=tof_b,
        tracer_inj=tr_i,
        tracer_prod=tr_p,
        inj_wells=inj,
        prod_wells=prod,
    )


__all__ = ["Diagnostics", "compute_diagnostics"]
