"""Two-phase matrix-fracture transfer (MRST ``KazemiTwoPhaseTransferFunction``).

On F0.4's risk list with no focused test. Transfer is the whole physics of a
dual-porosity model: get its sign or its upstream weighting wrong and the
simulation still runs, it just moves fluid the wrong way between continua.
"""

import numpy as np
import pytest

from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.modules.dual_porosity.transfer_models.kazemi_two_phase_transfer_function import (
    KazemiTwoPhaseTransfer,
)
from pymrst.params.rock import Rock


class _Sigma:
    """A shape factor with a value chosen by the test, not by a geometry."""

    def __init__(self, value=1.0):
        self.value = value

    def sigma(self, rock):
        return np.full(np.size(rock.poro), self.value)


def _fluid():
    return init_simple_adi_fluid(mu=[1.0e-3, 5.0e-3], rho=[1000.0, 800.0])


def _rock(cells=1):
    return Rock(perm=np.full((cells, 1), 1.0e-13), poro=np.full(cells, 0.1))


def _transfer(**kwargs):
    model = KazemiTwoPhaseTransfer(shape_factor=_Sigma(kwargs.pop("sigma", 1.0)),
                                   **kwargs)
    fluid = _fluid()
    defaults = dict(po_fracture=np.array([2.0e7]), sw_fracture=np.array([0.5]),
                    po_matrix=np.array([2.0e7]), sw_matrix=np.array([0.5]))
    return model, fluid, defaults


def test_equal_potentials_transfer_nothing():
    """The property a sign error cannot satisfy by accident.

    With both continua at the same pressure and saturation there is no
    driving force, so both transfers must be exactly zero - not small.
    """
    model, fluid, fields = _transfer()
    t_w, t_o = model.transfer(_rock(), fluid, fluid, **fields)
    np.testing.assert_array_equal(np.asarray(t_w), 0.0)
    np.testing.assert_array_equal(np.asarray(t_o), 0.0)


def test_the_sign_convention_is_positive_from_fracture_into_matrix():
    model, fluid, fields = _transfer()
    into = dict(fields, po_fracture=np.array([2.1e7]))
    t_w, t_o = model.transfer(_rock(), fluid, fluid, **into)
    assert float(np.asarray(t_w)[0]) > 0.0
    assert float(np.asarray(t_o)[0]) > 0.0

    out = dict(fields, po_matrix=np.array([2.1e7]))
    t_w, t_o = model.transfer(_rock(), fluid, fluid, **out)
    assert float(np.asarray(t_w)[0]) < 0.0
    assert float(np.asarray(t_o)[0]) < 0.0


def test_transfer_scales_with_the_shape_factor():
    """`sigma` carries the fracture geometry, and enters linearly."""
    fields = dict(po_fracture=np.array([2.1e7]), sw_fracture=np.array([0.5]),
                  po_matrix=np.array([2.0e7]), sw_matrix=np.array([0.5]))
    fluid = _fluid()
    one = KazemiTwoPhaseTransfer(shape_factor=_Sigma(1.0))
    ten = KazemiTwoPhaseTransfer(shape_factor=_Sigma(10.0))
    a_w, a_o = one.transfer(_rock(), fluid, fluid, **fields)
    b_w, b_o = ten.transfer(_rock(), fluid, fluid, **fields)
    np.testing.assert_allclose(np.asarray(b_w), 10.0*np.asarray(a_w), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(b_o), 10.0*np.asarray(a_o), rtol=1e-12)


def test_relative_permeability_is_taken_from_the_upstream_continuum():
    """Which side supplies `kr` is decided by the potential, per phase.

    The two continua are given different saturations so their relative
    permeabilities differ, and the flow direction is then reversed. Upstream
    weighting means the mobility must change with the direction; taking `kr`
    always from one side would give the same magnitude both ways.
    """
    fluid = _fluid()
    model = KazemiTwoPhaseTransfer(shape_factor=_Sigma(1.0))
    common = dict(sw_fracture=np.array([0.8]), sw_matrix=np.array([0.2]))

    into_matrix, _ = model.transfer(
        _rock(), fluid, fluid, po_fracture=np.array([2.1e7]),
        po_matrix=np.array([2.0e7]), **common,
    )
    out_of_matrix, _ = model.transfer(
        _rock(), fluid, fluid, po_fracture=np.array([2.0e7]),
        po_matrix=np.array([2.1e7]), **common,
    )
    forward, backward = abs(float(np.asarray(into_matrix)[0])), \
        abs(float(np.asarray(out_of_matrix)[0]))
    assert forward > backward, (
        "water flowing out of the dry matrix should be less mobile than water "
        "flowing in from the wet fracture"
    )
    # The wetter side really is the more mobile one, so the comparison above
    # is testing upstream selection rather than an accident of the curves.
    assert float(np.asarray(fluid.kr_w(np.array([0.8])))[0]) > \
        float(np.asarray(fluid.kr_w(np.array([0.2])))[0])


def test_capillary_pressure_shifts_the_water_potential_and_not_the_oil():
    """`pcOW` enters the water phase only, through `pw = po - pc`.

    A capillary pressure applied to both phases, or to the wrong one, would
    still produce plausible transfers - so the oil leg is pinned as unchanged
    while the water leg moves.
    """
    fluid = _fluid()
    fields = dict(po_fracture=np.array([2.0e7]), sw_fracture=np.array([0.5]),
                  po_matrix=np.array([2.0e7]), sw_matrix=np.array([0.5]))
    plain = KazemiTwoPhaseTransfer(shape_factor=_Sigma(1.0))
    base_w, base_o = plain.transfer(_rock(), fluid, fluid, **fields)

    # Capillary pressure in the matrix only: water is drawn in, oil is not.
    with_pc = KazemiTwoPhaseTransfer(
        shape_factor=_Sigma(1.0),
        pc_ow_matrix=lambda sw: np.full(np.size(sw), 1.0e5),
    )
    pc_w, pc_o = with_pc.transfer(_rock(), fluid, fluid, **fields)

    np.testing.assert_array_equal(np.asarray(base_w), 0.0)
    assert float(np.asarray(pc_w)[0]) > 0.0, (
        "a positive matrix capillary pressure should imbibe water"
    )
    np.testing.assert_allclose(np.asarray(pc_o), np.asarray(base_o), atol=1e-15)


def test_it_works_cellwise_on_more_than_one_cell():
    fluid = _fluid()
    model = KazemiTwoPhaseTransfer(shape_factor=_Sigma(1.0))
    t_w, t_o = model.transfer(
        _rock(3), fluid, fluid,
        po_fracture=np.array([2.1e7, 2.0e7, 1.9e7]),
        sw_fracture=np.full(3, 0.5),
        po_matrix=np.full(3, 2.0e7), sw_matrix=np.full(3, 0.5),
    )
    signs = np.sign(np.asarray(t_o))
    np.testing.assert_array_equal(signs, [1.0, 0.0, -1.0])


def test_the_transfer_equals_its_closed_form():
    """Magnitudes, not just signs and ratios.

    Every other test here compares one transfer against another, so anything
    that scales both phases - or swaps a constant between them - survives.
    Exchanging the water and oil viscosities did exactly that: it left every
    sign, every ratio against sigma, and every upstream comparison intact.

    With sigma = 1, b = 1 and kr = 1/2 at sw = 0.5, the transfers are
    `dp/(2*mu)` per phase, and the two viscosities differ fivefold - so a swap
    exchanges the two numbers and is visible here.
    """
    fluid = _fluid()
    model = KazemiTwoPhaseTransfer(shape_factor=_Sigma(1.0))
    dp = 1.0e6
    t_w, t_o = model.transfer(
        _rock(), fluid, fluid,
        po_fracture=np.array([2.0e7 + dp]), sw_fracture=np.array([0.5]),
        po_matrix=np.array([2.0e7]), sw_matrix=np.array([0.5]),
    )
    # Guard the constants the closed form assumes, so a change in the fluid
    # turns into a clear failure here rather than a wrong expected value.
    np.testing.assert_allclose(np.asarray(fluid.kr_w(np.array([0.5]))), 0.5)
    np.testing.assert_allclose(np.asarray(fluid.kr_o(np.array([0.5]))), 0.5)
    np.testing.assert_allclose(np.asarray(fluid.b_w(np.array([2.0e7]))), 1.0)

    np.testing.assert_allclose(np.asarray(t_w), 0.5*dp/1.0e-3, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(t_o), 0.5*dp/5.0e-3, rtol=1e-12)
