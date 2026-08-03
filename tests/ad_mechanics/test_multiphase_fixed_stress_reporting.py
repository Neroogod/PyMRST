import pytest

from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
from pymrst.utils.units import DAY


@pytest.mark.parametrize("fluid", ["oil water", "blackoil"])
def test_multiphase_fixed_stress_workflow_retains_nested_reports(fluid):
    result = run_2d_case(
        fluid_model=fluid,
        method="fixed stress splitting",
        # The translated MRST rate is deliberately aggressive; ten columns
        # give its saturation front enough cells for the default Newton cap.
        nx=10,
        ny=2,
        nsteps=1,
        dt=DAY,
    )
    report = result.reports[0]
    assert report["converged"]
    split = report["splitting"]
    assert split["converged"]
    assert split["iterations"] == report["iterations"]
    assert len(split["outer"]) == split["iterations"]
    for outer in split["outer"]:
        assert outer["mechanics_residual"] is not None
        flow = outer["flow"]
        assert flow["converged"]
        assert len(flow["residuals"]) >= 1
        assert {"CNV", "MB", "well", "control"} <= set(flow["residuals"][-1])
