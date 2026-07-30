"""Tests for MRST ``checkWellConvergence`` and ``printConvergenceReport``."""

from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils.convergence import (
    print_convergence_report,
    well_control_residual,
)


def test_well_convergence_scales_rate_and_control_equations():
    wells = [
        SimpleNamespace(type="rate", val=2e-3),
        SimpleNamespace(type="bhp", val=150e5),
    ]
    well_value, control_value = well_control_residual(
        [np.array([1e-6, 0.0]), np.array([0.0, 2e-6])],
        np.array([1e-4, 3e3]),
        wells,
    )
    assert np.isclose(well_value, 2e-6 / 2e-3)
    assert np.isclose(control_value, max(1e-4 / 2e-3, 3e3 / 1e5))
    assert well_control_residual([], np.array([]), []) == (0.0, 0.0)


def test_convergence_report_flags_pass_and_failure(capsys):
    print_convergence_report(
        ["well", "control"], [1e-4, 2e-2], [1e-3, 1e-3], iteration=4
    )
    output = capsys.readouterr().out
    assert "iteration 4" in output
    assert "ok" in output and "well" in output
    assert "FAIL" in output and "control" in output
