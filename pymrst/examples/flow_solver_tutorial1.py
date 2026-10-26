"""My First Flow Solver: Gravity Column (MRST ``flowSolverTutorial1.m``).

A single column of 30 cells, filled with one fluid, closed on every side except
the top, where the pressure is fixed at 100 bar. Nothing drives flow, so the
solution is the hydrostatic profile

    p(z) = p_top + rho * g * z

and every flux is zero. That makes this the smallest useful check of the TPFA
pressure solver: the answer is known in closed form, so a discretisation or
gravity-sign error shows up immediately rather than as a plausible-looking
field.

MRST's version ends by plotting face pressures. The port computes the same
quantities, checks them against the analytic profile, and leaves plotting to
the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp import incomp_tpfa_ad as incomp_tpfa
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.params import gravity_vector
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import pside
from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.units import BARSA, CENTI, DARCY, POISE


@dataclass
class GravityColumnResult:
    """Grid, rock, fluid and converged state for the gravity column."""

    G: object
    rock: object
    fluid: object
    state: object
    rho: float

    @property
    def pressure(self) -> np.ndarray:
        return np.asarray(self.state.pressure, dtype=float)

    @property
    def face_pressure(self) -> np.ndarray:
        return np.asarray(self.state.facePressure, dtype=float)

    def analytic_pressure(self, p_top: float = 100.0 * BARSA) -> np.ndarray:
        """`p_top + rho*g*z` at the cell centroids."""
        z = np.asarray(self.G.cells.centroids[:, 2], dtype=float)
        return p_top + self.rho * 9.80665 * z


def run_flow_solver_tutorial1(
    *,
    nz: int = 30,
    height: float = 30.0,
    permeability: float = 0.1 * DARCY,
    porosity: float = 1.0,
    mu: float = 1.0 * CENTI * POISE,
    rho: float = 1014.0,
    p_top: float = 100.0 * BARSA,
) -> GravityColumnResult:
    """Solve MRST's gravity column and return the state.

    Defaults reproduce ``flowSolverTutorial1.m``: a 1x1x30 column one metre
    square and 30 m tall, 0.1 darcy, unit porosity, 1 cP fluid at
    1014 kg/m3, and 100 bar imposed on the top face.
    """
    g = compute_geometry(cart_grid([1, 1, int(nz)], [1.0, 1.0, float(height)]))
    rock = make_rock(g, permeability, porosity)
    fluid = init_single_fluid(mu, rho)
    bc = pside(None, g, "TOP", p_top)
    trans = compute_trans(g, rock)
    state = incomp_tpfa(init_res_sol(g, 0.0), g, trans, fluid, bc=bc,
                        gravity=gravity_vector(g.griddim))
    return GravityColumnResult(G=g, rock=rock, fluid=fluid, state=state, rho=rho)


def plot_flow_solver_tutorial1(result: GravityColumnResult | None = None):
    """Plot face pressures in bar, as MRST's ``plotFaces`` call does.

    Returns ``None`` when no plotting backend is installed, so the tutorial
    still runs headless.
    """
    result = run_flow_solver_tutorial1() if result is None else result
    try:
        from pymrst.plotting import plot_faces
    except ImportError:
        return None
    return plot_faces(result.G, np.arange(result.G.faces.num),
                      result.face_pressure / BARSA)


if __name__ == "__main__":
    def test_matches_the_hydrostatic_profile():
        """The whole point of the case: p = p_top + rho*g*z, exactly.

        MRST's tutorial only plots the field, so an error in the gravity term
        would look like a slightly different colour ramp. Checking against the
        closed form instead is what makes this a test rather than a picture.
        """
        out = run_flow_solver_tutorial1()
        assert out.G.cells.num == 30, out.G.cells.num
        assert np.allclose(out.pressure, out.analytic_pressure(), rtol=1e-12), (
            np.max(np.abs(out.pressure - out.analytic_pressure())))
        print("ok test_matches_the_hydrostatic_profile")

    def test_no_flow_anywhere():
        """Hydrostatic equilibrium means every flux is zero, not merely small."""
        out = run_flow_solver_tutorial1()
        assert np.max(np.abs(np.asarray(out.state.flux))) < 1e-16, (
            np.max(np.abs(np.asarray(out.state.flux))))
        print("ok test_no_flow_anywhere")

    def test_top_boundary_holds_its_pressure():
        out = run_flow_solver_tutorial1()
        z = out.G.faces.centroids[:, 2]
        exterior = np.any(np.asarray(out.G.faces.neighbors) == -1, axis=1)
        top = np.flatnonzero(exterior & (z < 1e-10))
        assert top.size > 0, "no top face found"
        assert np.allclose(out.face_pressure[top], 100.0 * BARSA, rtol=1e-12)
        print("ok test_top_boundary_holds_its_pressure")

    def test_density_enters_the_answer():
        """A heavier fluid must give a steeper gradient, in proportion."""
        light = run_flow_solver_tutorial1(rho=500.0)
        heavy = run_flow_solver_tutorial1(rho=1000.0)
        dl = light.pressure.max() - light.pressure.min()
        dh = heavy.pressure.max() - heavy.pressure.min()
        assert np.isclose(dh / dl, 2.0, rtol=1e-10), (dl, dh)
        print("ok test_density_enters_the_answer")

    def test_column_height_scales_the_drop():
        short = run_flow_solver_tutorial1(height=15.0)
        tall = run_flow_solver_tutorial1(height=30.0)
        ds = short.pressure.max() - short.pressure.min()
        dt = tall.pressure.max() - tall.pressure.min()
        # Cell centroids sit half a cell from each end, so the ratio is taken
        # over the sampled span rather than the full column height.
        assert dt > ds and np.isclose(dt / ds, 2.0, rtol=1e-10), (ds, dt)
        print("ok test_column_height_scales_the_drop")

    test_matches_the_hydrostatic_profile()
    test_no_flow_anywhere()
    test_top_boundary_holds_its_pressure()
    test_density_enters_the_answer()
    test_column_height_scales_the_drop()
    print("\nAll tests passed!")
