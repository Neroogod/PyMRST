"""MRST ``diagnostics/utils/computeTOFandTracer.m``: forward and backward
time of flight with injector and producer tracer partitions.

Wells with positive total connection flux are injectors, the rest
producers (a shut well counts by its sign); `tracerWells` limits which
take part. Pressure and flux boundary conditions add inflow and outflow
partitions of the boundary faces, split by `partitionBoundary` and, with
`splitBoundary`, by flow direction. Indices are 0-based.

`compute_diagnostics` and `Diagnostics`, below, are PyMRST's earlier
summary, kept for their callers.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.solvers.state import State


from pymrst.modules.diagnostics.utils.compute_time_of_flight import compute_time_of_flight


def compute_tof_and_tracer(state: Any, G: Any, rock: Any, *, bc: Any = None, src: Any = None,
                           wells: Any = None, tracerWells: Any = None,
                           solver: Optional[Callable] = None, maxTOF: Optional[float] = None,
                           processCycles: bool = False, computeWellTOFs: bool = False,
                           firstArrival: bool = False, splitBoundary: bool = False,
                           partitionBoundary: Any = None) -> SimpleNamespace:
    """MRST `D = computeTOFandTracer(state, G, rock, 'wells', W, ...)`.

    `D.tof[:, 0]`/`D.tof[:, 1]` are the forward/backward times of flight,
    `D.itracer`/`D.ptracer` the injector/producer tracers, `D.ipart`/`D.ppart`
    the dominant one per cell (-1 where none reaches), `D.inj`/`D.prod` the
    well indices; with boundary conditions also `D.in`, `D.out` and their
    tracers and partitions, with `computeWellTOFs` the per-well times of
    flight, with `firstArrival` the first-arrival times."""
    from pymrst.modules.diagnostics.utils.validate_state_for_diagnostics import (
        validate_state_for_diagnostics,
    )
    if firstArrival:
        computeWellTOFs = True
    _check_tracer_input(G, rock, bc, src, wells)
    state = validate_state_for_diagnostics(state)
    wells = list(wells or [])
    tracer_wells = (np.ones(len(wells), dtype=bool) if tracerWells is None
                    else np.asarray(tracerWells, dtype=bool).ravel())
    well_sols = _get(state, "wellSol") or []
    wflux = _total_well_sol_flux(well_sols)
    iwells = wflux > 0
    D = SimpleNamespace()
    D.inj = np.flatnonzero(iwells & tracer_wells)
    injcells = [np.asarray(wells[k].cells) for k in D.inj]
    D.prod = np.flatnonzero(~iwells & tracer_wells)
    prodcells = [np.asarray(wells[k].cells) for k in D.prod]
    has_bc = bc is not None
    if has_bc:
        D.__dict__["in"], D.out, incells, outcells = _setup_boundary_partitions(
            G, state, bc, splitBoundary, partitionBoundary)
    else:
        incells, outcells = [], []
    sum_flux = sum(float(np.abs(np.asarray(ws.flux)).sum()) for ws in well_sols)
    nc = G.cells.num
    if sum_flux == 0.0 and not has_bc:
        D.tof = np.full((nc, 2), np.inf)
        D.itracer = np.full((nc, D.inj.size), np.nan)
        D.ipart = np.full(nc, np.nan)
        D.ptracer = np.full((nc, D.prod.size), np.nan)
        D.ppart = np.full(nc, np.nan)
        return D
    common = dict(wells=wells, solver=solver, maxTOF=maxTOF, processCycles=processCycles,
                  computeWellTOFs=computeWellTOFs, firstArrival=firstArrival, bc=bc)
    t = compute_time_of_flight(state, G, rock, tracer=injcells + incells, **common)[0]
    D.tof = np.zeros((nc, 2))
    D.tof[:, 0] = t[:, 0]
    _split_columns(D, t, D.inj.size, len(incells), has_bc, computeWellTOFs, firstArrival,
                   ("itracer", "intracer", "itof", "intof", "ifa", "infa"))
    D.ipart = _dominant(D.itracer)
    if has_bc:
        D.inpart = _dominant(D.intracer)
    t = compute_time_of_flight(state, G, rock, tracer=prodcells + outcells, reverse=True,
                               **common)[0]
    D.tof[:, 1] = t[:, 0]
    _split_columns(D, t, D.prod.size, len(outcells), has_bc, computeWellTOFs, firstArrival,
                   ("ptracer", "outtracer", "ptof", "outtof", "pfa", "outfa"))
    D.ppart = _dominant(D.ptracer)
    if has_bc:
        D.outpart = _dominant(D.outtracer)
    return D


def _split_columns(D: SimpleNamespace, t: np.ndarray, nw: int, nb: int, has_bc: bool,
                   well_tofs: bool, first_arrival: bool, names: Tuple[str, ...]) -> None:
    setattr(D, names[0], t[:, 1:nw + 1])
    if has_bc:
        setattr(D, names[1], t[:, nw + 1:nw + nb + 1])
    if well_tofs:
        cur = nw + nb + 1
        setattr(D, names[2], t[:, cur:cur + nw])
        if has_bc:
            setattr(D, names[3], t[:, cur + nw:cur + nw + nb])
        if first_arrival:
            cur = cur + nw + nb
            setattr(D, names[4], t[:, cur:cur + nw])
            if has_bc:
                setattr(D, names[5], t[:, cur + nw:])


def _dominant(tracer: np.ndarray) -> np.ndarray:
    """MRST `[val, part] = max(tracer, [], 2); part(val == 0) = 0`, 0-based
    with -1 for no tracer."""
    if tracer.shape[1] == 0:
        return np.full(tracer.shape[0], -1, dtype=np.int64)
    part = np.argmax(tracer, axis=1)
    part[tracer.max(axis=1) == 0] = -1
    return part


def _total_well_sol_flux(well_sols: Sequence[Any]) -> np.ndarray:
    flux = np.zeros(len(well_sols))
    for i, ws in enumerate(well_sols):
        f = float(np.sum(ws.flux))
        status = getattr(ws, "status", None)
        if status is not None and np.size(status):
            f = f*float(np.ravel(status)[0])
        sign = getattr(ws, "sign", None)
        if f == 0 and sign is not None and np.size(sign):
            f = np.sqrt(np.finfo(float).eps)*float(np.ravel(sign)[0])
        flux[i] = f
    return flux


def _setup_boundary_partitions(G: Any, state: Any, bc: Any, split: bool, p: Any):
    f = np.asarray(bc.face, dtype=np.int64).ravel()
    neig = np.asarray(G.faces.neighbors, dtype=np.int64)[f]
    c = neig.max(axis=1)
    fsgn = 2.0*(c == neig[:, 0]) - 1
    p = np.ones(f.size, dtype=np.int64) if p is None else np.asarray(p, dtype=np.int64).ravel()
    assert p.size == f.size, "Boundary partition must correspond to bc.faces"
    if np.any(~(p >= 1)):
        ix = p >= 1
        p, f, c, fsgn = p[ix], f[ix], c[ix], fsgn[ix]
    if p.max() > 1:
        order = np.argsort(p, kind="stable")
        p, f, c, fsgn = p[order], f[order], c[order], fsgn[order]
    bflux = -np.asarray(_get(state, "flux"), dtype=float).ravel()[f]*fsgn
    maxp = int(p.max())
    if split:
        inpart = outpart = np.arange(1, maxp + 1)
        pix = bflux >= 0
        incells = _group(c[pix], p[pix])
        outcells = _group(c[~pix], p[~pix])
    else:
        sgn = np.bincount(p, bflux)[1:]
        cells = _group(c, p)
        inpart, outpart = np.flatnonzero(sgn >= 0) + 1, np.flatnonzero(sgn < 0) + 1
        incells = [cells[k - 1] for k in inpart]
        outcells = [cells[k - 1] for k in outpart]
    return inpart, outpart, incells, outcells


def _group(cells: np.ndarray, part: np.ndarray) -> List[np.ndarray]:
    """MRST `mat2cell(cells, accumarray(part, 1), 1)` on sorted parts."""
    counts = np.bincount(part)[1:] if part.size else np.zeros(0, dtype=np.int64)
    return np.split(cells, np.cumsum(counts)[:-1]) if counts.size else []


def _check_tracer_input(G: Any, rock: Any, bc: Any, src: Any, wells: Any) -> None:
    assert wells or bc is not None, "Wells or boundary structure are required for computeTOFandTracer"
    assert src is None, "Source terms not supported yet"
    poro = getattr(rock, "poro", None)
    assert poro is not None and np.size(poro) == G.cells.num, (
        "The rock input must have a field poro with porosity for each cell in the grid.")
    assert np.min(poro) > 0, "Rock porosities must be positive numbers."


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


computeTOFandTracer = compute_tof_and_tracer


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
    """Compute TOF and injector/producer tracer partitions."""
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


__all__ = ["Diagnostics", "computeTOFandTracer", "compute_diagnostics", "compute_tof_and_tracer"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.incomp.fluid.incompressible.init_single_fluid import init_single_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.solvers.init_state import init_state

    G = compute_geometry(cart_grid([8, 8, 1], [80.0, 80.0, 1.0]))
    ROCK = make_rock(G, 1e-13, 0.2)
    W = add_well(None, G, ROCK, np.array([0]), well_type="rate", val=1e-3, radius=0.1, name="I")
    W = add_well(W, G, ROCK, np.array([7]), well_type="bhp", val=1e7, radius=0.1, name="P1")
    W = add_well(W, G, ROCK, np.array([56]), well_type="bhp", val=1e7, radius=0.1, name="P2")
    with isolated_gravity("off"):
        STATE = incomp_tpfa(init_state(G, W, 1e7), G, compute_trans(G, ROCK),
                            init_single_fluid(1e-3, 1000.0), wells=W)

    def test_injector_and_producers_are_split_by_flux():
        D = compute_tof_and_tracer(STATE, G, ROCK, wells=W)
        assert np.array_equal(D.inj, [0]) and np.array_equal(D.prod, [1, 2])
        assert D.tof.shape == (64, 2) and np.all(D.tof > 0)
        # The far corner sees no flow by symmetry: cut off at maxTOF, no tracer.
        swept = np.arange(64) != 63
        assert np.allclose(D.itracer[swept, 0], 1.0) and np.all(D.ipart[swept] == 0)
        assert D.itracer[63, 0] == 0 and D.ipart[63] == -1 and D.tof[63, 0] == D.tof[:, 0].max()
        assert np.allclose(D.ptracer[swept].sum(axis=1), 1.0)
        assert D.ppart[7] == 0 and D.ppart[56] == 1
        print("ok test_injector_and_producers_are_split_by_flux")

    def test_symmetric_producers_split_the_field_by_the_diagonal():
        D = compute_tof_and_tracer(STATE, G, ROCK, wells=W)
        i, j = np.arange(64) % 8, np.arange(64) // 8
        assert np.all(D.ppart[j < i] == 0) and np.all(D.ppart[j > i] == 1)
        print("ok test_symmetric_producers_split_the_field_by_the_diagonal")

    def test_tracer_wells_restricts_the_partition():
        D = compute_tof_and_tracer(STATE, G, ROCK, wells=W, tracerWells=[True, True, False])
        assert np.array_equal(D.prod, [1]) and D.ptracer.shape == (64, 1)
        print("ok test_tracer_wells_restricts_the_partition")

    def test_no_flow_gives_infinite_tof():
        state = STATE.__class__(**vars(STATE))
        state.wellSol = [type(ws)(flux=0*ws.flux, pressure=ws.pressure) for ws in STATE.wellSol]
        D = compute_tof_and_tracer(state, G, ROCK, wells=W)
        assert np.all(np.isinf(D.tof)) and np.all(np.isnan(D.ipart))
        print("ok test_no_flow_gives_infinite_tof")

    test_injector_and_producers_are_split_by_flux()
    test_symmetric_producers_split_the_field_by_the_diagonal()
    test_tracer_wells_restricts_the_partition()
    test_no_flow_gives_infinite_tof()
    print("\nAll tests passed!")
