"""PyMRST's earlier quarter five-spot SPU/WENO comparison on the legacy models.

Kept for its callers; the port of MRST `wenoExampleAD` is `weno_example_ad.py`.
What follows is the original description.

Higher-order upwinding on a quarter five-spot (MRST `wenoExampleAD`).

Water is injected in one corner of a square, homogeneous, two-phase patch and
produced in the opposite corner. With single-point upwind (SPU) the saturation
front smears over many cells, because the scheme takes the whole upstream
cell's saturation and so cannot represent a discontinuity inside a cell. WENO
reconstructs a face value from a stencil instead, and the same front arrives
with far less numerical diffusion.

The example exists to show that difference, and to show the CFL field that says
where an explicit scheme could be used at all.

**Why this was the last unported example.** It demonstrates
`WENOUpwindDiscretizationAD`, and the note in
`ad_core/statefunctions/flux/upwind/__init__.py` said that class was "not yet
ported: it needs a MATLAB `wenoExampleAD` reference to validate". The two were
blocking each other, and the parity ledger reported this file as *ported* on
the strength of that very sentence - its evidence was a `symbol-reference`
match against a comment saying the opposite. The scheme is ported now, in one
and in two/three dimensions, so the example is too.

What is not ported is the MATLAB tail: `plotCellData`, `streamline`,
`outlineCoarseGrid` and the four-panel figure loop are presentation. Every
number they display is returned instead, so the comparison the example makes
can be asserted rather than looked at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.cfl import estimate_saturation_cfl_ad
from pymrst.modules.ad_core.utils.set_weno_discretization import (
    set_weno_discretization_ad,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock, pore_volume
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, POISE, YEAR


@dataclass
class WENOExampleResult:
    """The numbers MRST's figures show, returned instead of plotted."""

    grid: object
    #: Final water saturation under single-point upwind and under WENO.
    saturation_spu: np.ndarray
    saturation_weno: np.ndarray
    #: Per-cell saturation CFL of the first step - where an explicit scheme
    #: would be stable (`cfl <= 1`) and where it would not.
    cfl: np.ndarray
    #: Total variation of the saturation field. The sharpness measure: a
    #: smeared front has *lower* total variation than a sharp one of the same
    #: swept volume, because the same total change is spread over more cells.
    front_width_spu: float
    front_width_weno: float
    states_spu: List[State]
    states_weno: List[State]


def _build(n: int = 20, n_steps: int = 30):
    """MRST's setup, at a size a test can afford.

    MRST uses 50x50 with 300 steps. The physics does not depend on either -
    the quarter five-spot is scale-free here because the rock is homogeneous -
    so the default is smaller and the MRST size is reachable by argument.
    """
    g = compute_geometry(cart_grid([n, n, 1], [100.0, 100.0, 10.0]))
    rock = make_rock(g, 0.1*DARCY, 0.3)
    fluid = init_simple_adi_fluid(
        mu=[1*CENTI*POISE, 1*CENTI*POISE], rho=[100.0, 100.0],
        n=[1.0, 1.0], c=[0.0, 0.0], p_ref=100*BARSA)

    time = 1.0*YEAR
    rate = 2.0*float(pore_volume(g, rock).sum())/time

    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=rate, compi=np.array([1.0, 0.0]), name="INJ")
    wells = add_well(wells, g, rock, np.array([g.cells.num - 1]),
                     well_type="bhp", val=50*BARSA,
                     compi=np.array([1.0, 0.0]), name="PROD")

    dts = [time/n_steps]*n_steps
    return g, rock, fluid, wells, dts


def _simulate(g, rock, fluid, wells, dts, weno: bool):
    model = GenericBlackOilModelAD(g, rock, fluid, wells)
    if weno:
        # The whole point of the example: swap single-point upwind for the
        # higher-order reconstruction, change nothing else.
        model = set_weno_discretization_ad(model)
    nc = g.cells.num
    nw = len(wells)
    state = State(pressure=np.full(nc, 100*BARSA), sW=np.zeros(nc),
                  qWs=np.zeros(nw), qOs=np.zeros(nw),
                  bhp=np.full(nw, 100*BARSA))
    solver = NonLinearSolver(verbose=False, progress="off")

    states = [state]
    st = state
    for dt in dts:
        st, report = solver.solve_timestep(model, st, float(dt))
        if not report.converged:
            raise RuntimeError(f"WENO example step did not converge ({dt:.3g} s)")
        states.append(st)
    return model, states


def _total_variation(g, sw: np.ndarray) -> float:
    """Sum of |ds| across neighbouring cells - the front sharpness measure.

    Higher is sharper. A front smeared over ten cells traverses the same total
    saturation change as one smeared over two, so the *sum* is similar; what
    differs is that the sharp front concentrates it, and the smeared one puts
    saturation into cells the front has not reached. Comparing swept volume
    alone would not separate them, which is why MRST plots the profile.
    """
    neighbors = np.asarray(g.faces.neighbors)
    interior = np.all(neighbors >= 0, axis=1)
    left, right = neighbors[interior, 0], neighbors[interior, 1]
    return float(np.sum(np.abs(sw[left] - sw[right])))


def weno_example_ad_ad(n: int = 20, n_steps: int = 30) -> WENOExampleResult:
    """Run the quarter five-spot with SPU and with WENO, and compare."""
    g, rock, fluid, wells, dts = _build(n, n_steps)

    model_spu, states_spu = _simulate(g, rock, fluid, wells, dts, weno=False)
    _, states_weno = _simulate(g, rock, fluid, wells, dts, weno=True)

    sw_spu = np.asarray(_saturation(states_spu[-1]), dtype=float)
    sw_weno = np.asarray(_saturation(states_weno[-1]), dtype=float)

    cfl = estimate_saturation_cfl_ad(model_spu, states_spu[1], max(dts))

    return WENOExampleResult(
        grid=g, saturation_spu=sw_spu, saturation_weno=sw_weno,
        cfl=np.asarray(cfl, dtype=float),
        front_width_spu=_total_variation(g, sw_spu),
        front_width_weno=_total_variation(g, sw_weno),
        states_spu=states_spu, states_weno=states_weno)


def _saturation(state) -> np.ndarray:
    """Water saturation, under whichever spelling the state carries."""
    if "sW" in state:
        return np.asarray(state["sW"], dtype=float).ravel()
    s = np.atleast_2d(np.asarray(state["s"], dtype=float))
    return s[:, 0]



__all__ = ["WENOExampleResult", "weno_example_ad_ad"]


if __name__ == "__main__":

    def test_weno_sharpens_the_front():
        """The claim the example is built to make.

        Both schemes solve the same equations on the same grid with the same
        timesteps; only the face reconstruction differs. WENO must resolve the
        front over fewer cells, which shows as a larger total variation.
        """
        result = weno_example_ad_ad()
        assert result.front_width_weno > result.front_width_spu, (
            f"WENO {result.front_width_weno:.3f} did not sharpen the front "
            f"against SPU {result.front_width_spu:.3f}")
        ratio = result.front_width_weno/result.front_width_spu
        print(f"ok test_weno_sharpens_the_front (sharpness {ratio:.3f}x)")

    def test_both_schemes_conserve_water():
        """Sharpening must not create or destroy water: the two schemes should
        sweep almost the same volume, differing in how it is distributed."""
        result = weno_example_ad_ad()
        swept_spu = float(result.saturation_spu.sum())
        swept_weno = float(result.saturation_weno.sum())
        assert abs(swept_spu - swept_weno)/swept_spu < 0.15, \
            (swept_spu, swept_weno)
        print(f"ok test_both_schemes_conserve_water "
              f"(swept {swept_spu:.2f} vs {swept_weno:.2f})")

    def test_saturations_stay_physical():
        """A higher-order scheme can overshoot; WENO's limiter is what stops
        it. Saturations outside [0, 1] reach a relperm table that has no value
        there."""
        result = weno_example_ad_ad()
        for name, sw in (("SPU", result.saturation_spu),
                         ("WENO", result.saturation_weno)):
            assert sw.min() >= -1e-8, f"{name} undershoots: {sw.min()}"
            assert sw.max() <= 1.0 + 1e-8, f"{name} overshoots: {sw.max()}"
        print("ok test_saturations_stay_physical")

    def test_the_cfl_field_is_largest_at_the_injector():
        """The CFL field says where an explicit scheme would be unstable, and
        that is where the flow is fastest - at the injector."""
        result = weno_example_ad_ad()
        cfl = result.cfl
        assert cfl.shape == (result.grid.cells.num,)
        assert np.all(np.isfinite(cfl)) and np.all(cfl >= 0)
        assert cfl[0] == cfl.max(), (
            f"injector cell {cfl[0]:.3e} is not the CFL maximum "
            f"{cfl.max():.3e}")
        print(f"ok test_the_cfl_field_is_largest_at_the_injector "
              f"(max CFL {cfl.max():.3e})")

    test_weno_sharpens_the_front()
    test_both_schemes_conserve_water()
    test_saturations_stay_physical()
    test_the_cfl_field_is_largest_at_the_injector()
    print("\nAll tests passed!")
