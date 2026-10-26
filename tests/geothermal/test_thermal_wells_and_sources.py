"""Wells and sources in the geothermal kernel.

Both were refused - "thermal-water source terms are not implemented", "thermal
-water well equations are not implemented" - while the same file already
contained the pattern they need. The boundary block computes a rate and
upwinds the enthalpy by flow direction; a well is that with a well index in
place of the face transmissibility, and a source is that without the mobility
term because the rate is prescribed rather than driven.

The load-bearing property is the **enthalpy upwinding**, and it is the reason a
thermal model exists at all:

* an *injecting* perforation carries the **well's** enthalpy;
* a *producing* one carries the **cell's**.

Use the cell's for an injector and cold injected water arrives at reservoir
temperature - the injection heats nothing, no thermal front forms, and the
mass balance is perfectly correct throughout.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.geothermal.geothermal_model import (
    GeothermalModel, ThermalFluid,
)
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI


class _Fluid(ThermalFluid):
    """Water whose enthalpy rises with temperature - enough to see the front."""

    lambda_f = 0.6

    def rho(self, p, t):
        return 1000.0 + 0.0*np.asarray(p, dtype=float)

    def mu(self, p, t):
        return 1.0e-3 + 0.0*np.asarray(p, dtype=float)

    def u(self, p, t):
        return 4180.0*t

    def h(self, p, t):
        return 4180.0*t


def _model(n=10, **kwargs):
    g = compute_geometry(cart_grid([n, 1, 1], [100.0, 10.0, 10.0]))
    rock = make_rock(g, 100*MILLI*DARCY, 0.2)
    return GeothermalModel(g, rock, _Fluid(), **kwargs)


def _well(cells, wi=1e-11, val=300*BARSA, control="bhp"):
    return SimpleNamespace(cells=np.atleast_1d(np.asarray(cells)),
                           WI=np.full(np.size(cells), wi), val=val,
                           type=control)


# -- the injection temperature must be stated -------------------------------

def test_a_well_without_an_injection_temperature_is_refused():
    """Defaulting it would have injected water arrive at the reservoir's own
    temperature, which is precisely what a thermal model is for."""
    with pytest.raises(ValueError, match="injection temperature"):
        _model(wells=[_well(0)])


def test_a_source_without_an_injection_temperature_is_refused():
    with pytest.raises(ValueError, match="injection temperature"):
        _model(src=SimpleNamespace(cell=np.array([0]), rate=np.array([1e-4])))


# -- wells -------------------------------------------------------------------

def test_an_injector_drives_mass_into_the_reservoir():
    model = _model(wells=[_well(0)], well_T=np.array([300.0]))
    state = model.initial_state(200*BARSA, 350.0)
    mass, energy = model.assemble(state, state.pressure, state.T, float(DAY))

    # The residual is `accumulation + divergence - source`, so an injecting
    # cell's residual is pushed negative by its source.
    assert float(np.asarray(mass)[0]) < 0.0
    assert np.all(np.isfinite(np.asarray(energy)))


def test_a_cold_injector_removes_heat_a_hot_one_adds_it():
    """The anchor. Same rate, same cell, only the injection temperature
    differs - so any difference in the energy residual is the enthalpy the
    well carries, which is exactly what the upwinding decides."""
    reservoir_t = 350.0
    cold = _model(wells=[_well(0)], well_T=np.array([300.0]))
    hot = _model(wells=[_well(0)], well_T=np.array([420.0]))

    out = []
    for model in (cold, hot):
        state = model.initial_state(200*BARSA, reservoir_t)
        _, energy = model.assemble(state, state.pressure, state.T, float(DAY))
        out.append(float(np.asarray(energy)[0]))

    assert out[0] != pytest.approx(out[1]), (
        "the injection temperature did not reach the energy equation")
    # Hotter injection removes more from the residual, i.e. adds more heat.
    assert out[1] < out[0]


def test_a_producer_carries_the_cells_enthalpy_not_the_wells():
    """A producer draws what the cell holds. Its declared injection
    temperature must not affect anything - if it does, the upwinding is
    reversed and every producer exports the well's enthalpy instead."""
    residuals = []
    for well_t in (300.0, 420.0):
        model = _model(wells=[_well(0, val=100*BARSA)],
                       well_T=np.array([well_t]))
        state = model.initial_state(200*BARSA, 350.0)
        _, energy = model.assemble(state, state.pressure, state.T, float(DAY))
        residuals.append(float(np.asarray(energy)[0]))

    assert residuals[0] == pytest.approx(residuals[1]), (
        "a producer's energy depends on the well temperature; the enthalpy "
        "upwinding is reversed")


def test_a_rate_controlled_well_is_refused_clearly():
    """A rate control needs its own unknown and a closure equation. Inventing
    a bottom-hole pressure for it would put the wrong drawdown on every
    perforation."""
    model = _model(wells=[_well(0, control="rate")], well_T=np.array([300.0]))
    state = model.initial_state(200*BARSA, 350.0)
    with pytest.raises(NotImplementedError, match="bottom-hole pressure"):
        model.assemble(state, state.pressure, state.T, float(DAY))


def test_wells_are_inert_when_none_are_attached():
    """The path everything else in the suite takes must be untouched."""
    plain = _model()
    state = plain.initial_state(200*BARSA, 350.0)
    mass, energy = plain.assemble(state, state.pressure, state.T, float(DAY))
    np.testing.assert_allclose(np.asarray(mass), 0.0, atol=1e-9)


# -- sources -----------------------------------------------------------------

def test_a_source_injects_mass_and_its_own_heat():
    src = SimpleNamespace(cell=np.array([3]), rate=np.array([1e-4]))
    model = _model(src=src, src_T=np.array([420.0]))
    state = model.initial_state(200*BARSA, 350.0)
    mass, energy = model.assemble(state, state.pressure, state.T, float(DAY))

    assert float(np.asarray(mass)[3]) < 0.0
    assert float(np.asarray(energy)[3]) < 0.0


def test_a_sink_takes_the_cells_enthalpy():
    """Negative rate: the cell's own enthalpy leaves, so the source
    temperature is irrelevant - the mirror of the producer case."""
    residuals = []
    for src_t in (300.0, 420.0):
        src = SimpleNamespace(cell=np.array([3]), rate=np.array([-1e-4]))
        model = _model(src=src, src_T=np.array([src_t]))
        state = model.initial_state(200*BARSA, 350.0)
        _, energy = model.assemble(state, state.pressure, state.T, float(DAY))
        residuals.append(float(np.asarray(energy)[3]))
    assert residuals[0] == pytest.approx(residuals[1])


def test_the_equations_refuse_forces_the_model_does_not_carry():
    """The kernel reads wells and sources from the *model*, where their
    injection temperature is validated. Passing them only as driving forces
    would silently drop them."""
    from pymrst.modules.ad_blackoil.utils.equations_water_thermal import (
        equations_water_thermal,
    )
    model = _model()
    state = model.initial_state(200*BARSA, 350.0)
    with pytest.raises(ValueError, match="carries no"):
        equations_water_thermal(state, state, model, float(DAY),
                                {"W": [_well(0)]}, res_only=True)
