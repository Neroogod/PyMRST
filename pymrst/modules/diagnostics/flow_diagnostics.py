"""Flow diagnostics (MRST diagnostics/).

Cheap, steady-state measures of *where the fluid goes*, computed from a
single pressure solution with no time stepping. Given a flux field:

- **Time-of-flight** tau solves  v . grad(tau) = phi  by upwind finite
  volumes: how long a particle takes to travel from an injector (forward
  TOF) or to reach a producer (backward TOF).
- **Tracer partitions** solve the same operator with indicator sources,
  giving each cell's allocation to each well - the drainage and sweep
  regions.
- **F-Phi diagram**: flow capacity F against storage capacity Phi, with
  cells sorted by total travel time. F = Phi is piston displacement; the
  more F bulges above the diagonal, the more channelled the flow.
- **Lorenz coefficient** Lc = 2 x (area between the F-Phi curve and the
  diagonal): 0 = uniform displacement, 1 = infinitely channelled.
- **Sweep efficiency** Ev against dimensionless time, from the same curve.

These rank well placements without running a full simulation.
"""

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

    tof_forward: np.ndarray       # (nc,) travel time from the injectors
    tof_backward: np.ndarray      # (nc,) travel time to the producers
    tracer_inj: np.ndarray        # (nc, n_inj) injector allocation
    tracer_prod: np.ndarray       # (nc, n_prod) producer allocation
    inj_wells: List[int]          # indices into the well list
    prod_wells: List[int]

    @property
    def total_tof(self) -> np.ndarray:
        """Residence time: forward + backward travel time."""
        return self.tof_forward + self.tof_backward

    def injector_region(self) -> np.ndarray:
        """Which injector (index into inj_wells) sweeps each cell."""
        return np.argmax(self.tracer_inj, axis=1)

    def producer_region(self) -> np.ndarray:
        """Which producer (index into prod_wells) each cell drains to."""
        return np.argmax(self.tracer_prod, axis=1)


def _well_source(g: Grid, state: State, wells: Sequence[Well]) -> np.ndarray:
    """Net volumetric source per cell from the well solution."""
    q = np.zeros(g.cells.num)
    for k, w in enumerate(wells):
        np.add.at(q, w.cells, state.wellSol[k].flux)
    return q


def _tof_matrix(g: Grid, flux: np.ndarray, q: np.ndarray
                ) -> Tuple[sp.csr_array, np.ndarray]:
    """Upwind advection operator A and the positive sources qp."""
    nc = g.cells.num
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    n = g.faces.neighbors[internal]
    f = flux[internal]

    into_2 = np.maximum(f, 0.0)      # flux from n[:,0] into n[:,1]
    into_1 = -np.minimum(f, 0.0)     # flux from n[:,1] into n[:,0]
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
    """Time-of-flight and tracer partitions from a converged flow solution."""
    pv = pore_volume(g, rock)
    q = _well_source(g, state, wells)

    inj = [k for k in range(len(wells)) if state.wellSol[k].flux.sum() > 0]
    prod = [k for k in range(len(wells)) if state.wellSol[k].flux.sum() < 0]
    assert inj and prod, "diagnostics need an injector and a producer"

    def solve(flux, qq, sources: List[int]):
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

    return Diagnostics(tof_forward=tof_f, tof_backward=tof_b,
                       tracer_inj=tr_i, tracer_prod=tr_p,
                       inj_wells=inj, prod_wells=prod)


def compute_f_phi(pv: np.ndarray, tof_total: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Flow capacity F against storage capacity Phi (computeFandPhi)."""
    order = np.argsort(tof_total)
    t = tof_total[order]
    v = pv[order]
    phi = np.cumsum(v)
    phi = np.concatenate([[0.0], phi / phi[-1]])
    flux = v / np.maximum(t, 1e-300)
    ff = np.cumsum(flux)
    f = np.concatenate([[0.0], ff / ff[-1]])
    return f, phi


def lorenz_coefficient(f: np.ndarray, phi: np.ndarray) -> float:
    """Lorenz coefficient: 0 = uniform displacement, 1 = fully channelled."""
    dv = np.diff(phi)
    return float(2.0 * (np.sum(0.5 * (f[:-1] + f[1:]) * dv) - 0.5))


def compute_sweep(f: np.ndarray, phi: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep efficiency Ev against dimensionless time tD (computeSweep)."""
    keep = np.ones(f.size, dtype=bool)
    keep[1:] = f[:-1] <= f[1:] - np.sqrt(np.finfo(float).eps)
    f, phi = f[keep], phi[keep]
    td = np.concatenate([[0.0], np.diff(phi) / np.diff(f)])
    ev = phi + (1.0 - f) * td
    return ev, td


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.incomp import incomp_tpfa_ad as incomp_tpfa, init_single_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.state import init_state
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    N = 32

    def _quarter_five_spot(perm=None, n=N):
        """Injector in one corner, producer in the opposite one."""
        g = compute_geometry(cart_grid([n, n, 1], [500.0, 500.0, 10.0]))
        if perm is None:
            perm = np.full(g.cells.num, 100 * MILLI * DARCY)
        rock = make_rock(g, perm[:, None], 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-3, radius=0.1)
        w = add_well(w, g, rock, np.array([n * n - 1]), well_type="bhp",
                     val=100 * BARSA, radius=0.1)
        st = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)
        return g, rock, st, w

    def test_tof_and_tracer_basics():
        """TOF is positive and grows away from the injector; a single
        injector/producer pair claims every cell."""
        g, rock, st, w = _quarter_five_spot()
        d = compute_diagnostics(g, rock, st, w)
        assert d.inj_wells == [0] and d.prod_wells == [1]
        assert np.all(d.tof_forward >= 0) and np.all(d.tof_backward >= 0)
        assert np.allclose(d.tracer_inj[:, 0], 1.0, atol=1e-8)
        assert np.allclose(d.tracer_prod[:, 0], 1.0, atol=1e-8)
        diag = [i * N + i for i in range(N)]
        assert np.all(np.diff(d.tof_forward[diag]) > 0)
        assert np.all(np.diff(d.tof_backward[diag]) < 0)
        print("ok test_tof_and_tracer_basics")

    def test_mean_residence_time():
        """Mean residence time is of the order PV / injection rate."""
        g, rock, st, w = _quarter_five_spot()
        d = compute_diagnostics(g, rock, st, w)
        pv = pore_volume(g, rock)
        mean_rt = float(pv @ d.total_tof) / pv.sum()
        expect = pv.sum() / 1e-3
        assert 0.5 * expect < mean_rt < 3.0 * expect, (mean_rt, expect)
        print("ok test_mean_residence_time")

    def test_lorenz_detects_channelling():
        """A high-permeability streak raises the Lorenz coefficient."""
        g, rock, st, w = _quarter_five_spot()
        d = compute_diagnostics(g, rock, st, w)
        pv = pore_volume(g, rock)
        f, phi = compute_f_phi(pv, d.total_tof)
        lc_hom = lorenz_coefficient(f, phi)
        assert 0.0 < lc_hom < 1.0
        assert np.isclose(f[0], 0.0) and np.isclose(f[-1], 1.0)
        assert np.isclose(phi[0], 0.0) and np.isclose(phi[-1], 1.0)
        assert np.all(np.diff(f) >= -1e-12)

        perm = np.full(N * N, 50 * MILLI * DARCY)
        perm[[i * N + i for i in range(N)]] = 5000 * MILLI * DARCY
        g2, rock2, st2, w2 = _quarter_five_spot(perm=perm)
        d2 = compute_diagnostics(g2, rock2, st2, w2)
        f2, phi2 = compute_f_phi(pore_volume(g2, rock2), d2.total_tof)
        lc_het = lorenz_coefficient(f2, phi2)
        assert lc_het > lc_hom, (lc_het, lc_hom)
        print(f"ok test_lorenz_detects_channelling (homogeneous "
              f"{lc_hom:.3f} -> channelled {lc_het:.3f})")

    def test_sweep_efficiency():
        """Sweep efficiency starts at zero and stays within [0, 1]."""
        g, rock, st, w = _quarter_five_spot()
        d = compute_diagnostics(g, rock, st, w)
        f, phi = compute_f_phi(pore_volume(g, rock), d.total_tof)
        ev, td = compute_sweep(f, phi)
        assert np.isclose(ev[0], 0.0, atol=1e-9)
        assert 0.0 < ev[-1] <= 1.0 + 1e-9
        assert np.all(np.diff(td) >= -1e-9)
        print(f"ok test_sweep_efficiency (Ev at breakthrough = {ev[1]:.3f})")

    def test_two_injectors_partition():
        """Tracer allocations sum to one, and each injector sweeps its own
        corner."""
        n = 20
        g = compute_geometry(cart_grid([n, n, 1], [500.0, 500.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
        t = compute_trans(g, rock)
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=5e-4, radius=0.1)
        w = add_well(w, g, rock, np.array([n - 1]), well_type="rate",
                     val=5e-4, radius=0.1)
        w = add_well(w, g, rock, np.array([n * n - 1]), well_type="bhp",
                     val=100 * BARSA, radius=0.1)
        st = incomp_tpfa(init_state(g, w, 0.0), g, t, fluid, wells=w)
        d = compute_diagnostics(g, rock, st, w)
        assert d.inj_wells == [0, 1] and d.prod_wells == [2]
        assert np.allclose(d.tracer_inj.sum(axis=1), 1.0, atol=1e-7)
        reg = d.injector_region()
        assert reg[0] == 0 and reg[n - 1] == 1
        print("ok test_two_injectors_partition")

    test_tof_and_tracer_basics()
    test_mean_residence_time()
    test_lorenz_detects_channelling()
    test_sweep_efficiency()
    test_two_injectors_partition()
    print("\nAll tests passed!")
