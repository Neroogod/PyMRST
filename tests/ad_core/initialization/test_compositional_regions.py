from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.initialization import (
    get_initialization_regions_compositional,
)
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.models.eos.equation_of_state_model import (
    EquationOfStateModel,
)


def _model():
    mix = Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.array([190.564, 425.125, 617.7]),
        Pcrit=np.array([4599200.0, 3796000.0, 2103000.0]),
        Vcrit=np.array([9.86e-5, 2.55e-4, 6.098e-4]),
        acentric=np.array([0.0114, 0.201, 0.4884]),
        molarMass=np.array([0.016043, 0.058124, 0.142285]),
    )
    return SimpleNamespace(
        G=SimpleNamespace(cells=SimpleNamespace(num=4)),
        fluid=SimpleNamespace(phases=("oil", "gas"), swcon=0.0),
        EOSModel=EquationOfStateModel(None, mix, "PR"),
        n_phases=2,
        getPhaseNames=lambda: ("O", "G"),
        getNumberOfPhases=lambda: 2,
    )


def test_compositional_region_density_closures_match_eos_calls():
    model = _model()
    x = np.array([0.10, 0.25, 0.65])
    y = np.array([0.80, 0.15, 0.05])
    region = get_initialization_regions_compositional(
        model, [1100.0], cells=[1, 3], T=350.0, x=x, y=y,
        datum_depth=1000.0, datum_pressure=8.0e6,
    )
    p = np.array([7.5e6, 8.0e6])
    z = np.array([975.0, 1025.0])
    rho_l = region.rho[0](p, z)
    rho_v = region.rho[1](p, z)
    assert np.all(np.isfinite(rho_l)) and np.all(rho_l > 0)
    assert np.all(np.isfinite(rho_v)) and np.all(rho_v > 0)
    assert np.all(rho_l > rho_v)
    assert region.reference_index == 0
    assert np.array_equal(region.cells, [1, 3])


def test_composition_and_temperature_may_vary_with_depth():
    model = _model()
    x = lambda p, z: np.column_stack([
        0.1 + 0 * np.atleast_1d(z),
        0.2 + 0 * np.atleast_1d(z),
        0.7 + 0 * np.atleast_1d(z),
    ])
    y = lambda p, z: np.column_stack([
        0.8 + 0 * np.atleast_1d(z),
        0.15 + 0 * np.atleast_1d(z),
        0.05 + 0 * np.atleast_1d(z),
    ])
    region = get_initialization_regions_compositional(
        model, [1100.0], T=lambda p, z: 340.0 + 0.02 * np.asarray(z),
        x=x, y=y,
    )
    rho = region.rho[0](np.array([8.0e6, 8.0e6]), np.array([900.0, 1100.0]))
    assert rho.shape == (2,)
    assert rho[0] != rho[1]
