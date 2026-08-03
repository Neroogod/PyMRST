import numpy as np

from pymrst.modules.ad_mechanics.examples.run_adjoint_example import (
    AdjointExampleResult,
    mrst_adjoint_schedule,
    plot_adjoint_example,
    run_adjoint_example,
)
from pymrst.utils.units import DAY


def test_no_argument_adjoint_workflow_builds_runs_and_checks_gradient():
    result = run_adjoint_example(dts=[10.0 * DAY])
    assert isinstance(result, AdjointExampleResult)
    assert len(result.states) == 2
    assert result.controls.shape == (1, 3)
    assert result.adjoint.gradient.shape == (1, 3)
    assert result.finite_difference_gradient.shape == (1, 3)
    assert np.isfinite(result.objective)
    assert result.objective != 0.0
    assert result.gradient_relative_error < 1e-5
    # z points down in the reservoir convention: uplift is negative.
    assert result.adjoint.gradient[0, 0] < 0.0
    assert set(result.exponent_results) == {1.0, 10.0}
    for exponent, sensitivity in result.exponent_results.items():
        assert sensitivity.value > 0.0
        assert result.updated_controls[exponent][0, 0] < result.controls[0, 0]
        assert result.updated_controls[exponent][0, 1] < result.controls[0, 1]
        assert result.updated_objectives[exponent] < sensitivity.value
    figure, axes = plot_adjoint_example(result)
    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "Final pressure and wells"
    assert axes[0, 1].get_title() == "Injection schedules"
    assert axes[1, 0].get_title() == "Surface uplift"
    assert axes[1, 1].get_title() == "Normalized uplift sensitivity"
    figure.canvas.draw()
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_mrst_adjoint_schedule_has_exact_decline_and_plateau():
    dts, controls = mrst_adjoint_schedule()
    assert dts.shape == (31,)
    assert controls.shape == (31, 3)
    assert dts[0] == DAY
    assert np.all(dts[1:] == 10.0 * DAY)
    assert np.allclose(controls[:, 0], controls[:, 1])
    assert np.all(np.diff(controls[:, 0]) <= 0.0)
    assert np.allclose(controls[15:, 0], 0.1 / DAY)
    assert np.all(controls[:, 2] == 100.0e5)


def test_complete_31_step_mrst_control_schedule_converges():
    result = run_adjoint_example(
        use_mrst_schedule=True,
        check_finite_difference=False,
        exponents=(),
        perform_schedule_updates=False,
    )
    assert len(result.dts) == 31
    assert len(result.states) == 32
    assert result.adjoint.gradient.shape == (31, 3)
    assert np.isfinite(result.objective)


def test_exact_mrst_31_by_30_geometry_and_schedule_converge():
    result = run_adjoint_example(
        use_mrst_geometry=True,
        use_mrst_schedule=True,
        check_finite_difference=False,
        exponents=(),
        perform_schedule_updates=False,
    )
    assert tuple(result.model.G.cartDims) == (31, 30)
    assert result.model.G.cells.num == 930
    assert result.model.G.griddim == 2
    assert len(result.states) == 32
    assert result.adjoint.gradient.shape == (31, 3)
    assert np.isfinite(result.objective)
