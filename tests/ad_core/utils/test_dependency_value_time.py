from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    add_property_dependence,
    assign_value,
    get_simulation_time,
)
from pymrst.utils.adi import ADI


def test_assign_value_preserves_sparse_jacobian():
    jac = sp.csr_array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    original = ADI(np.array([10.0, 20.0, 30.0]), [jac])
    assigned = assign_value(original, [-2.0, -3.0], [0, 2])
    assert np.array_equal(assigned.val, [-2.0, 20.0, -3.0])
    assert np.array_equal(original.val, [10.0, 20.0, 30.0])
    assert (assigned.jac[0] != jac).nnz == 0
    assert np.array_equal(assign_value([1, 2, 3], 9, 1), [1, 9, 3])


def test_add_property_dependence_internal_and_external():
    prop = SimpleNamespace(dependencies=[], externals=[])
    add_property_dependence(prop, ["Pressure", "Saturation"])
    add_property_dependence(prop, "Mobility", "FlowPropertyFunctions")
    supplied = SimpleNamespace(name="temperature", grouping="state")
    add_property_dependence(prop, supplied)
    assert prop.dependencies == ["Pressure", "Saturation"]
    assert prop.externals[0].name == "Mobility"
    assert prop.externals[0].grouping == "FlowPropertyFunctions"
    assert prop.externals[1] is supplied


def test_get_simulation_time_for_control_steps_and_ministeps():
    control_reports = [
        SimpleNamespace(SimulationTime=2.0),
        SimpleNamespace(SimulationTime=3.0),
    ]
    assert np.array_equal(
        get_simulation_time([object(), object()], control_reports), [2.0, 5.0]
    )
    detailed = SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=False, Timestep=4.0),
            SimpleNamespace(Converged=True, Timestep=1.5),
        ]),
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=True, Timestep=2.5),
        ]),
    ])
    assert np.array_equal(
        get_simulation_time([object()] * 3, detailed), [1.5, 4.0]
    )
