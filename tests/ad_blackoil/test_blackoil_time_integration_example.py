import numpy as np
import pytest

from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
    run_blackoil_time_integration_example,
)


def test_blackoilTimeIntegrationExample_runs_all_spe1_modes():
    variants = run_blackoil_time_integration_example("spe1", nsteps=2)
    assert list(variants) == [
        "Fully-Implicit", "Explicit", "Adaptive-Implicit (AIM)",
    ]
    for variant in variants.values():
        assert len(variant.result.states) == 2
        assert variant.result.report.converged
        assert variant.diagnostics.ministeps.sum() == pytest.approx(
            variant.diagnostics.report_steps.sum()
        )
        assert np.all(variant.diagnostics.iterations > 0)

    fim = variants["Fully-Implicit"].diagnostics
    aim = variants["Adaptive-Implicit (AIM)"].diagnostics
    assert np.all(fim.implicit_fraction == 1.0)
    assert aim.implicit_fraction[0] == 1.0
    assert 0.0 < aim.implicit_fraction[1] < 1.0


def test_blackoilTimeIntegrationExample_guards_spe9_dependency():
    with pytest.raises(NotImplementedError, match="SPE9"):
        run_blackoil_time_integration_example("spe9", nsteps=1)
