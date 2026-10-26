import numpy as np
import pytest

from pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example import (
    run_blackoil_time_integration_example_ad,
)


def test_blackoilTimeIntegrationExample_runs_all_spe1_modes():
    variants = run_blackoil_time_integration_example_ad("spe1", nsteps=2)
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


def test_blackoilTimeIntegrationExample_runs_spe9():
    """SPE9 used to be refused for a "pending deck, VFP and schedule
    importer". All three existed - `setup_spe9` builds them and
    `simulate_spe9_ad` runs on them - so the refusal had outlived its reason.

    Two steps only: the point is that all three discretizations reach the same
    benchmark, not that they agree on a converged answer.
    """
    variants = run_blackoil_time_integration_example_ad("spe9", nsteps=2)
    assert set(variants) == {"Fully-Implicit", "Explicit",
                             "Adaptive-Implicit (AIM)"}
    for name, variant in variants.items():
        d = variant.diagnostics
        assert d.ministeps.size > 0, name
        assert d.iterations.sum() > 0, name

    # The comparison the example exists to make: AIM solves fewer cells
    # implicitly than the fully-implicit scheme and more than the explicit one.
    fractions = {n: v.diagnostics.implicit_fraction.mean()
                 for n, v in variants.items()}
    assert fractions["Explicit"] < fractions["Adaptive-Implicit (AIM)"] \
        < fractions["Fully-Implicit"], fractions


def test_blackoilTimeIntegrationExample_rejects_an_unknown_case():
    with pytest.raises(ValueError, match="unknown"):
        run_blackoil_time_integration_example_ad("spe42", nsteps=1)
