"""Black-oil capillary pressure and Leverett J-function scaling (ad_core port).

Two pieces of physics that PyMRST's generic `CapillaryPressure` - which just
evaluates the fluid's single `pc` curve - does not have.

**The water sign.** Capillary pressure is the non-wetting minus the wetting
phase pressure. Water is wetting, so its offset from the reference pressure is
*negative*, and MRST stores `pc{water} = -pcow`. Getting that backwards moves
the oil-water contact the wrong way and still produces a perfectly plausible
saturation profile, which is why it is the first thing tested here.

**J-function scaling.** A measured `pc` curve belongs to the core plug it came
from. Applied unscaled across a heterogeneous field it gives every cell the
same entry pressure regardless of permeability - wrong in the direction that
matters, because tight rock holds water far more strongly than good rock. The
J-function restores the dependence as `sigma * phi^a / k^b`, and the classic
Leverett choice `a = b = 0.5` gives `sqrt(phi/k)`.

Scaling is off unless a surface tension is set, so a model that never asks for
it is unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.statefunctions.flowprops import (
    BlackOilCapillaryPressure,
)

N = 3


class _Fluid:
    """A linear pc curve, so every expected value is closed-form."""

    @staticmethod
    def pcOW(sw):
        return 1.0e5*(1.0 - np.asarray(sw, dtype=float))

    @staticmethod
    def pcOG(sg):
        return 2.0e5*np.asarray(sg, dtype=float)


def _model(poro=0.2, perm=1e-13, phases=("W", "O"), fluid=_Fluid):
    return SimpleNamespace(
        fluid=SimpleNamespace(phases=list(phases)),
        raw_fluid=fluid,
        rock=SimpleNamespace(poro=np.full(N, poro),
                             perm=np.full((N, 3), perm)),
        get_phase_names=lambda: list(phases),
        get_prop=lambda state, name: state[name],
    )


# -- the sign -------------------------------------------------------------

def test_the_water_offset_is_negative():
    """Water is the wetting phase, so it sits *below* the reference pressure.

    A flipped sign moves the oil-water contact the wrong way and produces a
    saturation profile that looks entirely reasonable.
    """
    out = BlackOilCapillaryPressure().evaluate_on_domain(
        _model(), {"sW": np.full(N, 0.4)})
    expected = -_Fluid.pcOW(np.full(N, 0.4))
    np.testing.assert_allclose(out[0], expected)
    assert np.all(np.asarray(out[0]) < 0)


def test_the_gas_offset_is_positive():
    """Gas is non-wetting, so its offset carries the opposite sign to water."""
    model = _model(phases=("W", "O", "G"))
    out = BlackOilCapillaryPressure().evaluate_on_domain(
        model, {"sW": np.full(N, 0.3), "sG": np.full(N, 0.2)})
    assert np.all(np.asarray(out[0]) < 0), "water below reference"
    assert np.all(np.asarray(out[2]) > 0), "gas above reference"


def test_the_oil_phase_carries_no_offset():
    """Oil is the reference, so its entry is zero rather than absent - the
    caller indexes by phase and a shorter list would misalign."""
    model = _model(phases=("W", "O", "G"))
    out = BlackOilCapillaryPressure().evaluate_on_domain(
        model, {"sW": np.full(N, 0.3), "sG": np.full(N, 0.2)})
    assert len(out) == 3
    np.testing.assert_allclose(out[1], np.zeros(N))


def test_a_fluid_with_no_curves_gives_no_offsets():
    """Inventing a curve for a model that has none would be worse than
    refusing; MRST returns empties and so does this."""
    class _Dry:
        pass

    model = _model(fluid=_Dry)
    assert not BlackOilCapillaryPressure.pc_present(model)
    assert all(o is None
               for o in BlackOilCapillaryPressure().evaluate_on_domain(
                   model, {"sW": np.full(N, 0.5)}))


# -- the J-function --------------------------------------------------------

def test_the_classic_leverett_ratio_is_sqrt_phi_over_k():
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.5, 0.5, "xyz")
    np.testing.assert_allclose(
        prop.get_j_function_static_ratio(_model(0.25, 1.0e-12)),
        np.sqrt(0.25/1.0e-12))


def test_the_exponents_are_free_rather_than_hard_coded():
    """Published correlations differ, so MRST leaves `a` and `b` open."""
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(1.0, 2.0, "xyz")
    np.testing.assert_allclose(
        prop.get_j_function_static_ratio(_model(0.3, 1.0e-13)),
        0.3**1.0/(1.0e-13)**2.0)


def test_tighter_rock_scales_the_curve_up():
    """The whole point. Low permeability holds water more strongly, so it must
    get a *larger* capillary pressure - if this ran the other way the scaling
    would be actively harmful."""
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.5, 0.5, "xyz")
    tight = prop.get_j_function_static_ratio(_model(0.2, 1.0e-15))
    loose = prop.get_j_function_static_ratio(_model(0.2, 1.0e-12))
    assert np.all(tight > loose)


def test_permeability_is_averaged_before_the_exponent():
    """The mean of a power is not the power of a mean, so the order matters.

    The exponent must not be 1, or the two orders coincide and the test proves
    nothing - which is what an earlier version of it did. With `b = 0.5` and
    permeabilities 1, 4, 9: `sqrt(mean) = 2.16` against `mean(sqrt) = 2.0`.
    """
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.0, 0.5, "xyz")
    model = _model()
    model.rock.perm = np.tile(np.array([[1.0, 4.0, 9.0]]), (N, 1))

    ratio = prop.get_j_function_static_ratio(model)
    average_then_power = 1.0/np.sqrt((1.0 + 4.0 + 9.0)/3.0)
    power_then_average = 1.0/((1.0 + 2.0 + 3.0)/3.0)
    assert not np.isclose(average_then_power, power_then_average), (
        "the exponent must distinguish the two orders")
    np.testing.assert_allclose(ratio, average_then_power)


def test_a_single_direction_reads_only_that_column():
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.0, 1.0, "z")
    model = _model()
    model.rock.perm = np.tile(np.array([[1.0, 2.0, 4.0]]), (N, 1))
    np.testing.assert_allclose(prop.get_j_function_static_ratio(model), 1.0/4.0)


def test_an_unknown_direction_letter_is_refused():
    """A typo would quietly change which permeability the scaling reads."""
    with pytest.raises(ValueError, match="unknown permeability direction"):
        BlackOilCapillaryPressure().set_j_function_constants(0.5, 0.5, "q")


def test_the_pressure_unit_divides_the_ratio():
    """The table may be in different units from the model."""
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.5, 0.5, "xyz", pressure_unit=1.0e5)
    np.testing.assert_allclose(
        prop.get_j_function_static_ratio(_model(0.25, 1.0e-12)),
        np.sqrt(0.25/1.0e-12)/1.0e5)


# -- scaling is opt-in -----------------------------------------------------

def test_scaling_is_off_until_a_surface_tension_is_set():
    """A model that never asks for it must be unaffected.

    The unset tension is asserted to be `None` rather than a default, because a
    default of 1.0 would silently enable scaling for every model.
    """
    prop = BlackOilCapillaryPressure()
    assert prop.get_surface_tension("ow") is None
    assert prop.get_surface_tension("og") is None
    assert not prop.has_j_function_scaler("ow")
    assert not prop.has_j_function_scaler("og")

    unscaled = prop.evaluate_on_domain(_model(), {"sW": np.full(N, 0.4)})
    np.testing.assert_allclose(unscaled[0], -_Fluid.pcOW(np.full(N, 0.4)))


def test_setting_a_surface_tension_enables_scaling_for_that_pair_only():
    prop = BlackOilCapillaryPressure()
    prop.set_surface_tension(0.03, "ow")
    assert prop.has_j_function_scaler("ow")
    assert not prop.has_j_function_scaler("og")
    assert prop.get_surface_tension("ow") == pytest.approx(0.03)


def test_a_scaled_curve_is_sigma_times_ratio_times_the_table():
    prop = BlackOilCapillaryPressure()
    prop.set_j_function_constants(0.5, 0.5, "xyz")
    prop.set_surface_tension(0.03, "ow")

    model = _model(0.25, 1.0e-12)
    sw = np.full(N, 0.4)
    out = prop.evaluate_on_domain(model, {"sW": sw})

    ratio = np.sqrt(0.25/1.0e-12)
    np.testing.assert_allclose(out[0], -(0.03*ratio*_Fluid.pcOW(sw)))


def test_an_unknown_phase_pair_is_refused():
    with pytest.raises(ValueError, match="unsupported phase pair"):
        BlackOilCapillaryPressure().set_surface_tension(0.03, "wg")
