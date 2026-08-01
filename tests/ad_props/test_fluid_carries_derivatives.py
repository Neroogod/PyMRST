"""Fluid properties must propagate derivatives, not just values.

`ad-props` exists to be differentiated. A property that returns the right
number but drops its Jacobian produces a Newton iteration that converges
slowly, to the wrong place, or not at all - and the values will look correct
throughout, so a value-only test says nothing about it.

The derivatives are checked against finite differences of the same function,
which is independent of how the analytic derivative is computed. Tolerances
are loose because a central difference on a piecewise-linear table is only
accurate to the step size; what is being caught is a missing or badly wrong
Jacobian, not a last-digit discrepancy.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_adi_fluid
from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)
from pymrst.utils.adi import ADI

DATA = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/examples/data")


def variable(values: np.ndarray) -> ADI:
    """An AD variable: values with an identity Jacobian."""
    values = np.asarray(values, dtype=float)
    return ADI(values, [sp.csr_array(sp.eye(values.size, format="csr"))])


def derivative(quantity: ADI) -> np.ndarray:
    """The diagonal of a property's Jacobian, one entry per cell."""
    return np.asarray(quantity.jac[0].todense()).diagonal()


@pytest.fixture(scope="module")
def spe1():
    path = DATA / "SPE1" / "BENCH_SPE1.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    return init_deck_adi_fluid(convert_deck_units(read_eclipse_deck(path)))


@pytest.fixture(scope="module")
def spe3():
    path = DATA / "SPE3" / "BENCH_SPE3.DATA"
    if not path.exists():
        pytest.skip("MRST example data not available")
    return init_deck_adi_fluid(convert_deck_units(read_eclipse_deck(path)))


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["bW", "pvMultR"])
def test_pressure_properties_return_an_adi(spe1, name):
    out = spe1[name](variable(np.array([2.0e7, 3.0e7])))
    assert isinstance(out, ADI), f"{name} dropped its derivatives"
    assert out.jac and out.jac[0].shape == (2, 2)


def test_constant_viscosity_returns_a_plain_array(spe1):
    """SPE1's PVTW has zero viscosibility, so muW is genuinely constant.

    MRST takes the same branch and returns a plain vector rather than an AD
    value. A constant has zero derivative, and AD arithmetic treats a plain
    array as constant, so nothing downstream loses information - but the
    return type does differ from the varying case.
    """
    p = variable(np.array([2.0e7, 3.0e7]))
    out = spe1["muW"](p)
    assert not isinstance(out, ADI)
    assert np.allclose(out, out[0]), "constant across the probe"


def test_a_constant_viscosity_still_composes_under_ad(spe1):
    """The thing that would actually break: mobility must stay differentiable."""
    p = variable(np.array([2.0e7, 3.0e7]))
    mobility = spe1["bW"](p) / spe1["muW"](p)
    assert isinstance(mobility, ADI), "mobility lost its derivatives"
    assert np.all(derivative(mobility) != 0.0)


@pytest.mark.parametrize("name", ["bW", "pvMultR"])
def test_pressure_derivatives_match_finite_differences(spe1, name):
    p = np.array([1.0e7, 2.0e7, 3.5e7])
    analytic = derivative(spe1[name](variable(p)))
    step = 1.0e3
    numeric = (spe1[name](p + step) - spe1[name](p - step)) / (2 * step)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-14), \
        (analytic, numeric)


def test_water_shrinkage_derivative_is_positive(spe1):
    """bW rises with pressure, so its derivative must be positive."""
    analytic = derivative(spe1["bW"](variable(np.array([1.0e7, 3.0e7]))))
    assert np.all(analytic > 0), analytic


@pytest.mark.parametrize("name", ["krW", "krG", "krOW", "krOG"])
def test_relative_permeability_returns_an_adi(spe1, name):
    out = spe1[name](variable(np.array([0.3, 0.6])))
    assert isinstance(out, ADI), f"{name} dropped its derivatives"


@pytest.mark.parametrize("name", ["krW", "krG"])
def test_relative_permeability_derivatives_match_finite_differences(spe1, name):
    """Probed strictly inside table segments.

    A central difference across a table breakpoint averages the slopes either
    side of the kink, so it disagrees with the analytic derivative there by
    construction. That is a property of piecewise-linear interpolation, not a
    defect - `test_the_derivative_at_a_breakpoint_is_one_sided` covers it.
    """
    s = np.array([0.2371, 0.4123, 0.6612])
    analytic = derivative(spe1[name](variable(s)))
    step = 1e-7
    numeric = (spe1[name](s + step) - spe1[name](s - step)) / (2 * step)
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-8), \
        (analytic, numeric)


def test_the_derivative_at_a_breakpoint_is_one_sided(spe1):
    """At a kink the interpolant picks one segment's slope, not the average."""
    breakpoint_ = 0.45
    analytic = derivative(spe1["krG"](variable(np.array([breakpoint_]))))
    step = 1e-6
    left = ((spe1["krG"](np.array([breakpoint_]))
             - spe1["krG"](np.array([breakpoint_ - step]))) / step)
    right = ((spe1["krG"](np.array([breakpoint_ + step]))
              - spe1["krG"](np.array([breakpoint_]))) / step)
    assert np.isclose(analytic, left, rtol=1e-4) or \
        np.isclose(analytic, right, rtol=1e-4), (analytic, left, right)


def test_relative_permeability_is_flat_outside_the_table(spe1):
    """The flat extension must have zero slope, not an extrapolated one."""
    analytic = derivative(spe1["krW"](variable(np.array([-0.5, 1.5]))))
    assert np.allclose(analytic, 0.0), analytic


def test_live_oil_carries_pressure_derivatives(spe1):
    p = np.array([2.0e7, 3.0e7])
    rs = spe1["rsSat"](p)
    out = spe1["bO"](variable(p), rs, np.ones(2, dtype=bool))
    assert isinstance(out, ADI)
    assert np.all(derivative(out) != 0.0), "bO must vary with pressure"


def test_live_oil_carries_dissolved_gas_derivatives(spe1):
    """Rs is a primary variable too, so bO must differentiate in it."""
    p = np.array([2.0e7, 3.0e7])
    rs = spe1["rsSat"](p)
    out = spe1["bO"](p, variable(rs), np.zeros(2, dtype=bool))
    assert isinstance(out, ADI)
    assert np.any(derivative(out) != 0.0), "bO must vary with Rs"


def test_live_oil_derivatives_add_when_both_are_variables(spe1):
    """d(bO)/dp and d(bO)/dRs must both appear, not one replacing the other."""
    p = np.array([2.0e7, 3.0e7])
    rs = spe1["rsSat"](p)
    unsaturated = np.zeros(2, dtype=bool)
    d_p = derivative(spe1["bO"](variable(p), rs, unsaturated))
    d_rs = derivative(spe1["bO"](p, variable(rs), unsaturated))
    both = derivative(spe1["bO"](variable(p), variable(rs), unsaturated))
    assert np.allclose(both, d_p + d_rs, rtol=1e-9), (both, d_p, d_rs)


def test_saturation_envelope_carries_derivatives(spe1):
    out = spe1["rsSat"](variable(np.array([1.0e7, 3.0e7])))
    assert isinstance(out, ADI)
    assert np.all(derivative(out) > 0), "Rs_sat rises with pressure"


def test_dead_oil_carries_pressure_derivatives(spe3):
    p = np.array([2.0e7, 3.0e7])
    out = spe3["bO"](variable(p))
    assert isinstance(out, ADI)
    assert np.all(derivative(out) != 0.0)


def test_wet_gas_carries_derivatives_in_both_arguments(spe3):
    p = np.array([2.0e7, 3.0e7])
    rv = spe3["rvSat"](p)
    unsaturated = np.zeros(2, dtype=bool)
    d_p = derivative(spe3["bG"](variable(p), rv, unsaturated))
    d_rv = derivative(spe3["bG"](p, variable(rv), unsaturated))
    assert np.any(d_p != 0.0), "bG must vary with pressure"
    assert np.any(d_rv != 0.0), "bG must vary with Rv"


def test_a_plain_array_input_returns_a_plain_array(spe1):
    """Differentiating is opt-in: without an ADI in, there is none out."""
    out = spe1["bW"](np.array([2.0e7, 3.0e7]))
    assert not isinstance(out, ADI)
    assert isinstance(out, np.ndarray)


def test_derivatives_are_per_cell_not_coupled(spe1):
    """Each cell's property depends only on its own pressure."""
    jac = spe1["bW"](variable(np.array([1.0e7, 2.0e7, 3.0e7]))).jac[0]
    dense = np.asarray(jac.todense())
    off_diagonal = dense - np.diag(np.diag(dense))
    assert np.allclose(off_diagonal, 0.0), dense
