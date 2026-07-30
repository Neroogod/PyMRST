import numpy as np

from examples.validation.validate_ad_blackoil_polymer_utils import main as polymer_validation
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.water_thermal_model import WaterThermalModel
from pymrst.modules.ad_blackoil.utils import equations_water_thermal, get_well_polymer
from pymrst.params.rock import make_rock
from pymrst.utils.adi import value


def test_polymer_utilities_match_mrst_reference():
    assert polymer_validation() == 0


def test_empty_well_polymer_mapping():
    result = get_well_polymer([])
    assert all(array.size == 0 for array in result)


def test_equations_water_thermal_delegates_with_ad_jacobians():
    grid = compute_geometry(cart_grid([3,1,1],[3.,1.,1.]))
    model = WaterThermalModel(grid, make_rock(grid, 1.0, 0.2))
    state0 = model.initial_state(2e7, 330.)
    state = state0.copy()
    state.pressure += np.array([1e4,0.,-1e4])
    equations, returned = equations_water_thermal(state0, state, model, 10.)
    assert returned is state
    assert len(equations) == 2
    assert len(equations[0].jac) == 2
    assert equations[0].jac[0].shape == (3,3)
    assert np.all(np.isfinite(value(equations[1])))
