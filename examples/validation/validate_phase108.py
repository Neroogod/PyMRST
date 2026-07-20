"""Phase 108: standalone vemmech LinearCompressibilityModel."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.incomp import incomp_tpfa, init_single_fluid
from pymrst.modules.vemmech.models import LinearCompressibilityModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_bc, add_source
from pymrst.solvers import compute_trans, init_res_sol
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _boundary_faces(g, side):
    boundary = np.any(g.faces.neighbors < 0, axis=1)
    c = g.faces.centroids
    if side == "xmin":
        return np.flatnonzero(boundary & np.isclose(c[:, 0], c[:, 0].min()))
    if side == "xmax":
        return np.flatnonzero(boundary & np.isclose(c[:, 0], c[:, 0].max()))
    raise ValueError(side)


def test_zero_compressibility_matches_incomp_tpfa():
    g = compute_geometry(cart_grid([5, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    bc = add_bc(None, _boundary_faces(g, "xmin"), "pressure", 200 * BARSA)
    bc = add_bc(bc, _boundary_faces(g, "xmax"), "pressure", 100 * BARSA)
    st0 = init_res_sol(g, 150 * BARSA)

    ref = incomp_tpfa(
        init_res_sol(g, 150 * BARSA),
        g,
        compute_trans(g, rock),
        fluid,
        bc=bc,
    )
    model = LinearCompressibilityModel(g, rock, fluid)
    out, report = model.step_function(
        st0,
        st0,
        10.0,
        SimpleNamespace(src=None, bc=bc, W=[]),
    )

    assert report.Converged and not report.Failure
    assert np.allclose(out.pressure, ref.pressure, rtol=1e-12, atol=1e-6)
    assert np.allclose(out.flux, ref.flux, rtol=1e-12, atol=1e-12)
    print("  ok zero-compressibility limit matches incompTPFA")


def test_closed_cell_compressible_source():
    g = compute_geometry(cart_grid([1, 1, 1], [10.0, 10.0, 1.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    rock.cr = np.array([2.0e-9])
    fluid = init_single_fluid(1 * CENTI * POISE, 1000.0)
    fluid.cf = 3.0e-9
    dt = 20.0
    p0 = np.array([100 * BARSA])
    q = 0.01
    src = add_source(None, np.array([0]), q)
    st0 = init_res_sol(g, p0)

    model = LinearCompressibilityModel(g, rock, fluid)
    out = model.step(st0, dt, {"src": src})
    pv = g.cells.volumes * rock.poro
    expected = p0 + q * dt / ((rock.cr + fluid.cf) * pv)
    assert np.allclose(out.pressure, expected, rtol=1e-12)
    assert np.allclose(out.flux, 0.0)
    print("  ok closed-cell compressible source analytic update")


def main():
    print("Phase 108: standalone LinearCompressibilityModel")
    test_zero_compressibility_matches_incomp_tpfa()
    test_closed_cell_compressible_source()
    print("Phase 108 validation passed: LinearCompressibilityModel is runnable.")


if __name__ == "__main__":
    main()
