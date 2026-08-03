from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case


ROOT = Path(__file__).resolve().parents[2]


def _mrst_controls(path):
    report = loadmat(path, squeeze_me=True, struct_as_record=False)["reports"]
    return np.atleast_1d(report.ControlstepReports)


def _compare_report_case(fluid, fixture_name):
    fixture = ROOT / "examples" / "validation" / fixture_name
    ref = loadmat(fixture, squeeze_me=True)
    result = run_2d_case(
        fluid_model=fluid, method="fixed stress splitting",
        nx=20, ny=4, physdims=(100.0, 10.0),
        nsteps=2, dt=np.asarray(ref["dt"], dtype=float).reshape(-1),
        p_ref=float(ref["pRef"]),
    )
    controls = _mrst_controls(fixture)
    for py, ml in zip(result.reports, controls):
        outer = py["splitting"]["outer"]
        mrst_nonlinear = np.atleast_1d(ml.StepReports.NonlinearReport)
        mrst_residuals = np.array([float(item.Residuals) for item in mrst_nonlinear])
        py_residuals = np.array([float(item["mechanics_residual"]) for item in outer])
        assert py["converged"] and bool(ml.Converged)
        assert py["cuts"] == int(ml.MinistepCuttingCount) == 0
        assert len(outer) == len(mrst_nonlinear)
        assert len(outer) - 1 == int(ml.Iterations)
        assert len(py_residuals) == len(mrst_residuals)
        assert py_residuals[-1] <= 1e-6 and mrst_residuals[-1] <= 1e-6
        for item in outer:
            assert {"CNV", "MB", "well", "control"} <= set(item["flow"]["residuals"][-1])
    return result, controls


def test_water_fixed_stress_iteration_cut_and_residual_history_match_mrst():
    result, controls = _compare_report_case(
        "water", "ref_p64_run2d_water_fixedstress.mat"
    )
    for py, ml in zip(result.reports, controls):
        actual = np.array([x["mechanics_residual"] for x in py["splitting"]["outer"]])
        expected = np.array([float(x.Residuals) for x in np.atleast_1d(ml.StepReports.NonlinearReport)])
        np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=3e-4, atol=2e-10)
        assert actual[-1] <= 1e-6 and expected[-1] <= 1e-6


def test_blackoil_fixed_stress_iteration_cut_and_residual_history_match_mrst():
    result, controls = _compare_report_case(
        "blackoil", "ref_p67_run2d_blackoil_fixedstress.mat"
    )
    for py, ml in zip(result.reports, controls):
        actual = np.array([x["mechanics_residual"] for x in py["splitting"]["outer"]])
        expected = np.array([float(x.Residuals) for x in np.atleast_1d(ml.StepReports.NonlinearReport)])
        np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=5e-5, atol=2e-10)
        assert actual[-1] <= 1e-6 and expected[-1] <= 1e-6


def test_oilwater_fixed_stress_report_counts_and_contraction_match_mrst():
    fixture = ROOT / "examples" / "validation" / "ref_p65_run2d_oilwater_fixedstress.mat"
    ref = loadmat(fixture, squeeze_me=True)
    result = run_2d_case(
        fluid_model="oil water", method="fixed stress splitting",
        nx=20, ny=4, physdims=(100.0, 10.0), nsteps=2,
        dt=np.asarray(ref["dt"], dtype=float).reshape(-1),
        p_ref=float(ref["pRef"]),
    )
    controls = _mrst_controls(fixture)
    for py, ml in zip(result.reports, controls):
        assert py["converged"] and bool(ml.Converged)
        assert py["cuts"] == int(ml.MinistepCuttingCount) == 0
        assert len(py["splitting"]["outer"]) == int(ml.Iterations)
        for item in py["splitting"]["outer"]:
            assert {"CNV", "MB", "well", "control"} <= set(item["flow"]["residuals"][-1])
    # The first step is directly identical. The second starts from states
    # that differ within the separately certified 2e-6 field tolerance, so
    # compare the exact report topology and monotone contraction envelope.
    actual = np.array([x["mechanics_residual"] for x in result.reports[0]["splitting"]["outer"]])
    expected = np.array([float(x.Residuals) for x in np.atleast_1d(controls[0].StepReports.NonlinearReport)])
    np.testing.assert_allclose(actual[:-1], expected[:len(actual) - 1], rtol=1e-4, atol=2e-10)
    assert actual[-1] <= 1e-6 and expected[-1] <= 1e-6
    for py in result.reports:
        residuals = np.array([x["mechanics_residual"] for x in py["splitting"]["outer"]])
        assert np.all(np.diff(residuals) < 0.0)
