"""End-to-end test for the runnable MRST ``mrst_demo`` counterpart."""

import pytest

from pymrst.modules.ad_core.examples import mrst_demo


def test_demo_runs_complete_ad_displacement():
    result = mrst_demo(timesteps=5)

    assert result.cell_count == 50
    assert result.state_count == 6
    assert result.inlet_saturation > 0.5
    assert result.outlet_saturation > 1e-3
    assert result.swept_cells > 25
    assert result.monotonicity_error < 1e-9


def test_demo_rejects_empty_schedule():
    with pytest.raises(ValueError):
        mrst_demo(timesteps=0)
