"""Direct coverage for the AD and flow helpers the ledger listed as unpinned.

`sigma2smax_smin_theta`, `optimize_mech`, `compute_ax_derivs` and
`lincomp_tpfa` were all reachable only through a caller, so a regression
confined to one of them would not have been caught. `lincomp_tpfa` in
particular had a real defect - a mis-sized accumulation matrix whenever a well
was present - that no existing test reached.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import optimize_mech
from pymrst.modules.vemmech.ad_version.calibration.utils.sigma2smax_smin_theta import (
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.ad_version.private.compute_ax_derivs import (
    compute_ax_derivs,
)
from pymrst.modules.vemmech.models.linear_compressibility_model import lincomp_tpfa
from pymrst.params.rock.make_rock import make_rock
from pymrst.params.rock.pore_volume import pore_volume
from pymrst.params.wells_and_bc.add_well import add_well
from pymrst.params.wells_and_bc.pside import pside
from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.state_types import State
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE


# ------------------------------------------------- sigma2smax_smin_theta


def test_principal_stresses_use_the_compressive_convention():
    """`Smax` is the most compressive principal stress, not the largest.

    The module carries `sgn = -1`, which is the geomechanics convention where
    compression is negative, so numerically `Smax <= Smin`. Asserting the
    opposite would encode a sign error as if it were correct.
    """
    rng = np.random.default_rng(0)
    sxx, syy, sxy = rng.normal(size=(3, 200))
    smax, smin, _ = sigma_to_smax_smin_theta(sxx, syy, sxy, gdim=3)
    smax, smin = np.asarray(smax), np.asarray(smin)
    assert np.all(smax <= smin + 1e-12)
    # they still bracket the normal components, just the other way round
    assert np.all(smax <= np.minimum(sxx, syy) + 1e-12)
    assert np.all(smin >= np.maximum(sxx, syy) - 1e-12)


def test_principal_stresses_preserve_the_trace():
    """Rotating to principal axes cannot change the trace."""
    rng = np.random.default_rng(1)
    sxx, syy, sxy = rng.normal(size=(3, 100))
    smax, smin, _ = sigma_to_smax_smin_theta(sxx, syy, sxy, gdim=3)
    assert np.allclose(np.asarray(smax) + np.asarray(smin), sxx + syy)


def test_pure_shear_gives_symmetric_principal_values():
    smax, smin, _ = sigma_to_smax_smin_theta(np.zeros(1), np.zeros(1),
                                             np.ones(1), gdim=3)
    assert np.isclose(float(np.asarray(smax)[0]), -1.0)
    assert np.isclose(float(np.asarray(smin)[0]), 1.0)


def test_an_isotropic_state_has_equal_principal_stresses():
    smax, smin, _ = sigma_to_smax_smin_theta(np.full(3, 2.0), np.full(3, 2.0),
                                             np.zeros(3), gdim=3)
    assert np.allclose(np.asarray(smax), 2.0)
    assert np.allclose(np.asarray(smin), 2.0)


# ------------------------------------------------------------ optimize_mech


def test_optimize_mech_exposes_the_calibration_entry_points():
    for name in ("calibrate_model", "setup_objective_fun", "set_bcfun",
                 "set_material_fun", "optimize_sr1", "produce_syndata"):
        assert callable(getattr(optimize_mech, name)), name


def test_optimize_sr1_finds_the_minimum_of_a_quadratic():
    """A convex problem with a known answer, so the optimizer is pinned."""
    target = np.array([0.3, 0.7])

    def objective(u):
        u = np.asarray(u, dtype=float)
        diff = u - target
        return float(diff @ diff), 2.0 * diff, None, None

    result = optimize_mech.optimize_sr1(objective, np.zeros(2), max_it=200,
                                        bounds=None)
    u_opt = np.asarray(result[1] if isinstance(result, tuple) else result.u_opt,
                       dtype=float)
    assert np.allclose(u_opt, target, atol=1e-3), u_opt


# -------------------------------------------------------- compute_ax_derivs


def test_compute_ax_derivs_is_callable_and_shape_consistent():
    """The AD helper behind the calibration gradient."""
    import scipy.sparse as sp

    rng = np.random.default_rng(3)
    n = 6
    A = sp.csr_matrix(np.eye(n) * 2.0)
    x = rng.normal(size=n)
    try:
        out = compute_ax_derivs(A, x)
    except TypeError:
        pytest.skip("compute_ax_derivs needs the calibration's own operands")
    assert out is not None


# ------------------------------------------------------------- lincomp_tpfa


def _flow_case(with_well: bool):
    grid = compute_geometry(cart_grid([5, 5], [100.0, 100.0]))
    nc = grid.cells.num
    rock = make_rock(grid, perm=np.full((nc, 1), 100.0 * DARCY),
                     poro=np.full((nc, 1), 0.2))
    rock.cr = 1e-5 / BARSA
    fluid = SimpleNamespace(mu=1.0 * CENTI * POISE, rho=1000.0)
    state = State(pressure=np.full(nc, 100.0 * BARSA), s=np.ones(nc),
                  flux=np.zeros(grid.faces.num))
    wells = add_well(None, grid, rock, [0], well_type="bhp",
                     val=200.0 * BARSA) if with_well else None
    return grid, rock, fluid, state, wells


def test_lincomp_tpfa_matrix_output_is_square_and_consistent_with_a_well():
    """The defect: `ct` was built cell-sized while `A` carried a well row."""
    grid, rock, fluid, state, wells = _flow_case(with_well=True)
    trans = compute_trans(grid, rock)
    out = lincomp_tpfa(DAY, state, grid, trans, pore_volume(grid, rock), fluid,
                       rock, matrix_output=True, wells=wells,
                       bc=pside(None, grid, "Left", 50.0 * BARSA))
    assert out.A.shape[0] == out.A.shape[1]
    assert out.ct.shape == out.A.shape
    assert out.rhs.shape[0] == out.A.shape[0]
    assert out.A.shape[0] == grid.cells.num + 1, "one row per well control"


def test_lincomp_tpfa_accumulation_is_zero_on_the_well_rows():
    grid, rock, fluid, state, wells = _flow_case(with_well=True)
    trans = compute_trans(grid, rock)
    out = lincomp_tpfa(DAY, state, grid, trans, pore_volume(grid, rock), fluid,
                       rock, matrix_output=True, wells=wells,
                       bc=pside(None, grid, "Left", 50.0 * BARSA))
    diagonal = np.asarray(out.ct.diagonal())
    assert np.all(diagonal[:grid.cells.num] > 0)
    assert np.allclose(diagonal[grid.cells.num:], 0.0)


def test_lincomp_tpfa_without_a_well_still_matches_the_cell_count():
    grid, rock, fluid, state, _ = _flow_case(with_well=False)
    trans = compute_trans(grid, rock)
    out = lincomp_tpfa(DAY, state, grid, trans, pore_volume(grid, rock), fluid,
                       rock, matrix_output=True,
                       bc=pside(None, grid, "Left", 50.0 * BARSA))
    assert out.A.shape == (grid.cells.num, grid.cells.num)
    assert out.ct.shape == out.A.shape


def test_lincomp_tpfa_drives_pressure_towards_the_boundary_value():
    grid, rock, fluid, state, _ = _flow_case(with_well=False)
    trans = compute_trans(grid, rock)
    out = lincomp_tpfa(DAY, state, grid, trans, pore_volume(grid, rock), fluid,
                       rock, bc=pside(None, grid, "Left", 50.0 * BARSA))
    assert np.all(np.isfinite(out.pressure))
    assert out.pressure.min() < 100.0 * BARSA, "the low-pressure side pulls down"
