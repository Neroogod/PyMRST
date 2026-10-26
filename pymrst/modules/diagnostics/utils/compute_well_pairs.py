"""MRST ``diagnostics/utils/computeWellPairs.m``: injector-producer pair
volumes and well allocations from the tracer partitions of
`computeTOFandTracer`.

The pore volume a pair shares is the volume-weighted overlap of the
injector's and the producer's tracers. Each well's connection fluxes are
split between the wells at the other end by the tracer concentrations in
its perforated cells. Indices are 0-based.

`compute_diagnostics` and `Diagnostics`, below, are PyMRST's earlier
summary, kept for their callers.
"""

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.solvers.state import State


def compute_well_pairs(state: Any, G: Any, rock: Any, W: Sequence[Any], D: Any) -> SimpleNamespace:
    """MRST `WP = computeWellPairs(state, G, rock, W, D)`.

    `WP.pairs` names each injector-producer pair (injectors fastest),
    `WP.vols` is the pore volume both tracers share, `WP.pairIx` the 0-based
    (injector, producer) indices into `D.inj`/`D.prod`. `WP.inj[i]` and
    `WP.prod[j]` give each well's per-perforation allocation to the wells at
    the other end (`alloc`), the remainder (`ralloc`), the perforation depths
    (`z`) and the name; a well with no connection fluxes keeps empty fields.
    """
    from pymrst.params.rock.pore_volume import pore_volume as pore_volume_fn
    well_sols = list(_get(state, "wellSol") or [])
    if well_sols and all(getattr(ws, "flux", None) is None for ws in well_sols) \
            and all(getattr(ws, "cqs", None) is not None for ws in well_sols):
        well_sols = [copy.copy(ws) for ws in well_sols]
        for ws in well_sols:
            ws.flux = np.asarray(ws.cqs, dtype=float).reshape(np.shape(ws.cqs)[0], -1).sum(axis=1)
    inj = np.atleast_1d(np.asarray(D.inj, dtype=np.int64))
    prod = np.atleast_1d(np.asarray(D.prod, dtype=np.int64))
    ni, np_ = inj.size, prod.size
    WP = SimpleNamespace()
    WP.pairs = [f"{W[i].name}, {W[p].name}" for p in prod for i in inj]
    pv = np.asarray(pore_volume_fn(G, rock), dtype=float)
    itracer = np.asarray(D.itracer, dtype=float).reshape(pv.size, -1)
    ptracer = np.asarray(D.ptracer, dtype=float).reshape(pv.size, -1)
    vols = itracer.T @ (pv[:, None]*ptracer)
    WP.vols = vols.ravel(order="F")
    WP.pairIx = np.column_stack([np.tile(np.arange(ni), np_), np.repeat(np.arange(np_), ni)])
    WP.inj = [_allocation(well_sols, W, k, ptracer, np_) for k in inj]
    WP.prod = [_allocation(well_sols, W, k, itracer, ni) for k in prod]
    return WP


def _allocation(well_sols: Sequence[Any], W: Sequence[Any], k: int, tracer: np.ndarray,
                n_other: int) -> SimpleNamespace:
    out = SimpleNamespace(alloc=None, ralloc=None, z=None, name=None)
    flux = np.asarray(well_sols[k].flux, dtype=float)
    if flux.size == 0:
        return out
    q = flux.reshape(flux.shape[0], -1).sum(axis=1)
    c = tracer[np.asarray(W[k].cells, dtype=np.int64)]
    out.alloc = q[:, None]*c
    out.ralloc = q - out.alloc.sum(axis=1)
    out.z = np.asarray(W[k].dZ, dtype=float) + W[k].refDepth
    out.name = W[k].name
    return out


def _get(state: Any, name: str) -> Any:
    return state.get(name) if isinstance(state, dict) else getattr(state, name, None)


computeWellPairs = compute_well_pairs


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


__all__ = ["Diagnostics", "computeWellPairs", "compute_diagnostics", "compute_well_pairs"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.diagnostics.utils.compute_tof_and_tracer import compute_tof_and_tracer
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
    D = compute_tof_and_tracer(STATE, G, ROCK, wells=W)

    def test_pairs_and_shared_volumes():
        WP = compute_well_pairs(STATE, G, ROCK, W, D)
        assert WP.pairs == ["I, P1", "I, P2"]
        assert np.array_equal(WP.pairIx, [[0, 0], [0, 1]])
        # Symmetric producers share the swept volume equally.
        assert np.isclose(WP.vols[0], WP.vols[1], rtol=1e-10)
        swept = np.arange(64) != 63
        assert np.isclose(WP.vols.sum(), (0.2*100.0)*swept.sum(), rtol=1e-6)
        print("ok test_pairs_and_shared_volumes")

    def test_allocations_split_the_well_rates():
        WP = compute_well_pairs(STATE, G, ROCK, W, D)
        inj = WP.inj[0]
        assert inj.name == "I" and inj.alloc.shape == (1, 2)
        assert np.isclose(inj.alloc.sum() + inj.ralloc.sum(), 1e-3)
        assert np.allclose(inj.alloc, 0.5e-3, rtol=1e-8)
        for wp, flux in zip(WP.prod, (STATE.wellSol[1].flux, STATE.wellSol[2].flux)):
            assert np.allclose(wp.alloc[:, 0], flux) and np.allclose(wp.ralloc, 0, atol=1e-18)
        print("ok test_allocations_split_the_well_rates")

    test_pairs_and_shared_volumes()
    test_allocations_split_the_well_rates()
    print("\nAll tests passed!")
