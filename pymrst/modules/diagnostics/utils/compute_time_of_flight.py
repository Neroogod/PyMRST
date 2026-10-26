"""MRST ``diagnostics/utils/computeTimeOfFlight.m``: time of flight and
tracers from a total flux field.

The steady transport operator `A` (upwind inflow on each cell's diagonal,
outflow coupling off it, plus the positive sources) is solved for the time
of flight, `A tau = pv`, and for each tracer set's concentration. Unless
`allowInf`, cells whose pore volume over inflow exceeds `maxTOF`
(default 50 times total pore volume over total inflow) are cut off and the
results capped there; `processCycles` cuts off closed flow cycles the same
way. `reverse` gives the backward time of flight. Indices are 0-based.

`compute_diagnostics` and `Diagnostics`, below, are PyMRST's earlier
summary of forward and backward time of flight, kept for their callers.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.solvers.state import State


def compute_time_of_flight(state: Any, G: Any, rock: Any, *, bc: Any = None, src: Any = None,
                           wells: Any = None, reverse: bool = False, allowInf: bool = False,
                           maxTOF: Optional[float] = None, tracer: Any = (),
                           solver: Optional[Callable] = None, processCycles: bool = False,
                           computeWellTOFs: bool = False,
                           firstArrival: bool = False) -> Tuple[np.ndarray, sp.csr_array, np.ndarray]:
    """MRST `[T, A, q] = computeTimeOfFlight(state, G, rock, ...)`. `tracer`
    is a list of cell sets (or one set); `T[:, 0]` is the time of flight and
    the following columns the tracer of each set, then - with
    `computeWellTOFs` - each set's own time of flight and, with
    `firstArrival`, its first-arrival time."""
    from pymrst.modules.diagnostics.utils.validate_state_for_diagnostics import (
        validate_state_for_diagnostics,
    )
    from pymrst.params.rock.pore_volume import pore_volume as pore_volume_fn
    _check_input(G, rock, bc, src, wells, maxTOF, allowInf, processCycles)
    if np.ndim(_get(state, "flux")) > 1 and np.shape(_get(state, "flux"))[1] > 1:
        state = validate_state_for_diagnostics(state)
    tr = list(tracer) if _is_set_list(tracer) else [tracer]
    tr = [np.asarray(t, dtype=np.int64).ravel() for t in tr]
    q, qb = _compute_source_term(state, G, wells, src, bc)
    pv = np.asarray(pore_volume_fn(G, rock), dtype=float).copy()
    flux = np.asarray(_get(state, "flux"), dtype=float).ravel()
    if reverse:
        q, qb, flux = -q, -qb, -flux
    neighbors = np.asarray(G.faces.neighbors, dtype=np.int64)
    i = ~np.any(neighbors < 0, axis=1)
    n = neighbors[i].copy()
    nc = G.cells.num
    qp = np.maximum(q + qb, 0)
    out = np.minimum(flux[i], 0)
    inn = np.maximum(flux[i], 0)
    inflow = np.bincount(np.concatenate([n[:, 1], n[:, 0]]), np.concatenate([inn, -out]),
                         minlength=nc)
    d = inflow + qp
    maxIn = d.max() if d.size else 0.0
    if not allowInf:
        if maxTOF is None:
            maxTOF = 50*pv.sum()/qp.sum()
        above_max = pv/np.maximum(d, np.finfo(float).eps*maxIn) > maxTOF
        d[above_max] = maxIn
        pv[above_max] = maxIn*maxTOF
        inn[above_max[n[:, 1]]] = 0
        out[above_max[n[:, 0]]] = 0
    A = (sp.csr_array((inn, (n[:, 1], n[:, 0])), shape=(nc, nc))
         + sp.csr_array((-out, (n[:, 0], n[:, 1])), shape=(nc, nc)))
    if d.size:
        A = -A + sp.diags_array(d, shape=(nc, nc), format="csr")
    if not allowInf and processCycles:
        A, pv = _threshold_connected_components(A, pv, maxIn, maxTOF)
    num_tr = len(tr)
    TrRHS = np.zeros((nc, num_tr))
    for k, cells in enumerate(tr):
        TrRHS[cells, k] = qp[cells]
    solve = solver if solver is not None else _direct_solve
    T = np.column_stack([solve(A, pv)] + [solve(A, TrRHS[:, k]) for k in range(num_tr)])
    if not allowInf:
        T[T > maxTOF] = maxTOF
    if computeWellTOFs:
        C = T[:, 1:]
        pvi = C*pv[:, None]
        pvi[pvi < 0] = 0
        X = np.column_stack([solve(A, pvi[:, k]) for k in range(pvi.shape[1])]) \
            if pvi.shape[1] else np.zeros((nc, 0))
        X[X < 0] = 0
        ix = (pvi*maxTOF > X) & (C > np.sqrt(np.finfo(float).eps))
        X[~ix] = maxTOF
        X[ix] = X[ix]/C[ix]
        T = np.column_stack([T, X])
        if firstArrival:
            T = np.column_stack([T, _first_arrival(n, out, inflow, qp, pv, tr, nc, maxTOF)])
    return T, A, q


def _direct_solve(A: sp.csr_array, b: np.ndarray) -> np.ndarray:
    return np.asarray(spla.spsolve(sp.csc_matrix(A), b)).ravel()


def _first_arrival(n: np.ndarray, out: np.ndarray, inflow: np.ndarray, qp: np.ndarray,
                   pv: np.ndarray, tr: List[np.ndarray], nc: int, maxTOF: float) -> np.ndarray:
    """MRST's first-arrival times: shortest paths from each tracer set through
    the flow graph, an edge costing the pore volume of the cell it enters over
    that cell's inflow."""
    from scipy.sparse.csgraph import dijkstra
    n = n.copy()
    flip = out < 0
    n[flip] = n[flip][:, ::-1]
    nconn = np.array([t.size for t in tr], dtype=np.int64)
    wnum = np.repeat(np.arange(len(tr)), nconn)
    wc = np.concatenate(tr) if tr else np.zeros(0, dtype=np.int64)
    n1 = np.concatenate([n[:, 0], wnum + nc])
    n2 = np.concatenate([n[:, 1], wc])
    fl = np.concatenate([inflow[n[:, 1]], qp[wc]])
    fl = np.maximum(fl, fl.max()*np.finfo(float).eps)
    weight = pv[n2]/fl
    nn = nc + len(tr)
    order = np.lexsort((weight, n2, n1))
    n1, n2, weight = n1[order], n2[order], weight[order]
    keep = np.concatenate([[True], (np.diff(n1) != 0) | (np.diff(n2) != 0)])
    graph = sp.csr_array((weight[keep], (n1[keep], n2[keep])), shape=(nn, nn))
    dist = dijkstra(graph, directed=True, indices=np.arange(nc, nn))
    return np.minimum(np.atleast_2d(dist)[:, :nc].T, maxTOF)


def _compute_source_term(state: Any, G: Any, W: Any, src: Any, bc: Any) -> Tuple[np.ndarray, np.ndarray]:
    nc = G.cells.num
    qi, qs = [], []
    if W:
        qi.append(np.concatenate([np.asarray(w.cells, dtype=np.int64).ravel() for w in W]))
        qs.append(np.concatenate([np.asarray(ws.flux, dtype=float).ravel()
                                  for ws in _get(state, "wellSol")]))
    if src is not None:
        qi.append(np.asarray(src.cell, dtype=np.int64).ravel())
        qs.append(np.asarray(src.rate, dtype=float).ravel())
    q = (np.bincount(np.concatenate(qi), np.concatenate(qs), minlength=nc) if qi else np.zeros(nc))
    if bc is not None:
        neighbors = np.asarray(G.faces.neighbors, dtype=np.int64)
        flux = np.asarray(_get(state, "flux"), dtype=float).ravel()
        types = np.resize(np.array([str(t) for t in np.atleast_1d(bc.type)]), np.size(bc.face))
        faces = np.asarray(bc.face, dtype=np.int64).ravel()
        ff = np.zeros(G.faces.num)
        dirichlet = faces[types == "pressure"]
        if dirichlet.size:
            ff[dirichlet] = flux[dirichlet]*(2.0*(neighbors[dirichlet, 0] < 0) - 1)
        neumann = types == "flux"
        ff[faces[neumann]] = np.asarray(bc.value, dtype=float).ravel()[neumann]
        is_outer = ~np.all(neighbors >= 0, axis=1)
        qb = np.bincount(neighbors[is_outer].max(axis=1), ff[is_outer], minlength=nc)
    else:
        qb = np.zeros(nc)
    return q, qb


def _threshold_connected_components(A: sp.csr_array, pv: np.ndarray, maxIn: float,
                                    maxTOF: float) -> Tuple[sp.csr_array, np.ndarray]:
    """MRST's `thresholdConnectedComponents`: a strongly connected block of
    cells (a flow cycle) whose total pore volume over its net inflow exceeds
    `maxTOF` is cut off, its rows set to `maxIn` on the diagonal."""
    from scipy.sparse.csgraph import connected_components
    nc = pv.size
    _, labels = connected_components(A != 0, directed=True, connection="strong")
    sizes = np.bincount(labels)
    multi = np.flatnonzero(sizes > 1)
    if multi.size == 0:
        return A, pv
    C = sp.csr_array((np.ones(nc), (np.arange(nc), labels)), shape=(nc, sizes.size))[:, multi]
    q_in = (C.T @ A @ C).diagonal()
    above = (C.T @ pv)/np.maximum(q_in, np.finfo(float).eps*maxIn) > maxTOF
    if np.any(above):
        bad = np.flatnonzero(np.isin(labels, multi[above]))
        A = A.tolil()
        A[bad, :] = 0
        A[bad, bad] = maxIn
        A = A.tocsr()
        pv = pv.copy()
        pv[bad] = maxIn*maxTOF
    return A, pv


def _check_input(G: Any, rock: Any, bc: Any, src: Any, wells: Any, maxTOF: Any,
                 allowInf: bool, processCycles: bool) -> None:
    import warnings
    assert not (src is None and bc is None and not wells), (
        "Must have inflow described as boundary conditions, sources, or wells")
    poro = getattr(rock, "poro", None)
    assert poro is not None and np.size(poro) == G.cells.num, (
        "The rock input must have a field poro with porosity for each cell in the grid.")
    assert np.min(poro) > 0, "Rock porosities must be positive numbers."
    if maxTOF is not None and allowInf:
        warnings.warn("Input value for 'maxTOF' ignored since option 'allowInf' has value 'true'.")
    if processCycles and allowInf:
        warnings.warn("Input request to process cycles ignored since option 'allowInf' has "
                      "value 'true'.")


def _is_set_list(tracer: Any) -> bool:
    return isinstance(tracer, (list, tuple)) and (len(tracer) == 0 or np.ndim(tracer[0]) > 0)


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


computeTimeOfFlight = compute_time_of_flight


# ---------------------------------------------------------------------------
# PyMRST's earlier diagnostics, kept for their callers.

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
    """Compute forward and backward TOF with tracer allocations."""
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


__all__ = ["Diagnostics", "computeTimeOfFlight", "compute_diagnostics", "compute_time_of_flight"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock

    N_CELLS, RATE = 5, 2.0
    G = compute_geometry(cart_grid([N_CELLS, 1, 1]))
    ROCK = make_rock(G, 1.0, 0.25)
    PV = 0.25
    INTERNAL = np.all(G.faces.neighbors >= 0, axis=1)
    FLUX = np.where(INTERNAL, RATE, 0.0)
    W = [SimpleNamespace(cells=np.array([0])), SimpleNamespace(cells=np.array([N_CELLS - 1]))]
    STATE = {"flux": FLUX, "wellSol": [SimpleNamespace(flux=np.array([RATE])),
                                       SimpleNamespace(flux=np.array([-RATE]))]}

    def test_uniform_flow_forward_and_backward():
        T, A, q = compute_time_of_flight(STATE, G, ROCK, wells=W, tracer=[W[0].cells])
        assert np.allclose(T[:, 0], (np.arange(N_CELLS) + 1)*PV/RATE)
        assert np.allclose(T[:, 1], 1.0) and q[0] == RATE and A.shape == (N_CELLS, N_CELLS)
        Tb = compute_time_of_flight(STATE, G, ROCK, wells=W, reverse=True)[0]
        assert np.allclose(Tb[:, 0], (N_CELLS - np.arange(N_CELLS))*PV/RATE)
        print("ok test_uniform_flow_forward_and_backward")

    def test_max_tof_caps_cells_without_inflow():
        state = dict(STATE, flux=np.zeros(G.faces.num))
        T = compute_time_of_flight(state, G, ROCK, wells=W, maxTOF=1.0)[0]
        assert np.isclose(T[0, 0], PV/RATE) and np.all(T[1:, 0] == 1.0)
        print("ok test_max_tof_caps_cells_without_inflow")

    def test_well_tofs_and_first_arrival():
        T = compute_time_of_flight(STATE, G, ROCK, wells=W, tracer=[W[0].cells],
                                   computeWellTOFs=True, firstArrival=True)[0]
        assert T.shape == (N_CELLS, 4)
        assert np.allclose(T[:, 2], T[:, 0])
        assert np.allclose(T[:, 3], (np.arange(N_CELLS) + 1)*PV/RATE)
        print("ok test_well_tofs_and_first_arrival")

    def test_phase_fluxes_are_reduced_to_total():
        state = {"flux": np.column_stack([FLUX/2, FLUX/2]),
                 "wellSol": [SimpleNamespace(flux=np.array([[RATE/2, RATE/2]])),
                             SimpleNamespace(flux=np.array([[-RATE/2, -RATE/2]]))]}
        T = compute_time_of_flight(state, G, ROCK, wells=W)[0]
        assert np.allclose(T[:, 0], (np.arange(N_CELLS) + 1)*PV/RATE)
        print("ok test_phase_fluxes_are_reduced_to_total")

    test_uniform_flow_forward_and_backward()
    test_max_tof_caps_cells_without_inflow()
    test_well_tofs_and_first_arrival()
    test_phase_fluxes_are_reduced_to_total()
    print("\nAll tests passed!")
