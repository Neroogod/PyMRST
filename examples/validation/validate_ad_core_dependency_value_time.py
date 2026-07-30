"""Validate ``assignValue``, ``addPropertyDependence``, and
``getSimulationTime`` against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    add_property_dependence,
    assign_value,
    get_simulation_time,
)
from pymrst.utils.adi import ADI


def _strings(value):
    return [str(item) for item in np.atleast_1d(value).ravel()]


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_dependency_value_time.mat",
        squeeze_me=True,
    )
    ad = ADI(np.array([10.0, 20.0, 30.0]), [sp.eye(3, format="csr")])
    assigned = assign_value(ad, [-2.0, -3.0], [0, 2])
    prop = SimpleNamespace(dependencies=[], externals=[])
    add_property_dependence(prop, ["Pressure", "Saturation"])
    add_property_dependence(prop, "Mobility", "FlowPropertyFunctions")
    add_property_dependence(
        prop, SimpleNamespace(name="temperature", grouping="state")
    )
    control_report = SimpleNamespace(SimulationTime=2.0)
    detailed = SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=False, Timestep=4.0),
            SimpleNamespace(Converged=True, Timestep=1.5),
        ]),
        SimpleNamespace(StepReports=[
            SimpleNamespace(Converged=True, Timestep=2.5),
        ]),
    ])
    numerical = {
        "double assignment": (assign_value([1, 2, 3], 9, 1), ref["x_double"]),
        "AD values": (assigned.val, ref["x_ad_value"]),
        "AD Jacobian": (assigned.jac[0].toarray(), ref["x_ad_jacobian"]),
        "control times": (
            get_simulation_time([None], control_report),
            ref["control_times"],
        ),
        "ministep times": (
            get_simulation_time([None] * 3, detailed),
            ref["ministep_times"],
        ),
    }
    failed = 0
    for name, (actual, expected) in numerical.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    string_checks = {
        "dependencies": (
            prop.dependencies,
            _strings(ref["dependencies"]),
        ),
        "external names": (
            [item.name for item in prop.externals],
            _strings(ref["external_names"]),
        ),
        "external groupings": (
            [item.grouping for item in prop.externals],
            _strings(ref["external_groupings"]),
        ),
    }
    for name, (actual, expected) in string_checks.items():
        ok = actual == expected
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {actual!r}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
