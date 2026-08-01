from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.compositional.examples.compositional_validation_simple import (
    setup_compositional_validation_simple_problem,
    external_comparison_metrics,
    validation_comparison_data,
    validation_plot_sections,
)
from pymrst.modules.compositional.examples.utils import (
    load_simplecomp_references,
    simplecomp_data_directory,
)


def test_missing_simplecomp_dataset_has_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="fetch_simplecomp_dataset"):
        simplecomp_data_directory(tmp_path)


def test_official_simplecomp_references_are_aligned_when_installed():
    try:
        root = simplecomp_data_directory()
    except FileNotFoundError:
        pytest.skip("official MRST simplecomp dataset is not installed")
    references = load_simplecomp_references(data_root=root)
    assert references["report_steps"] == 499
    assert len(references["E300"]) == len(references["AD-GPRS"]) == 499
    for name in ("E300", "AD-GPRS"):
        state = references[name][0]
        assert state.pressure.shape == (1000,)
        assert state.components.shape == (1000, 3)
        assert np.all(np.isfinite(state.pressure))
        assert np.allclose(state.components.sum(axis=1), 1.0, atol=2.0e-5)
    assert references["E300"][0].saturation.shape == (1000, 2)
    assert references["AD-GPRS"][0].saturation is None


def test_validation_example_exposes_all_installed_reference_profiles():
    try:
        root = simplecomp_data_directory()
    except FileNotFoundError:
        pytest.skip("official MRST simplecomp dataset is not installed")
    case = setup_compositional_validation_simple_problem(
        nstep=1, load_external_references=True, reference_root=root
    )
    run_output = {
        "result": type("Result", (), {"states": [case["setup"].state]})(),
        "setup": case["setup"],
        "external_references": case["external_references"],
    }
    comparison = validation_comparison_data(run_output)
    sections = validation_plot_sections(run_output)
    assert comparison["available"] == ("MRST", "E300", "AD-GPRS")
    assert sections["pressure_saturation_comparison"]["E300"] is not None
    assert sections["pressure_saturation_comparison"]["E300"]["components"].shape == (1000, 3)


def test_full_schedule_uses_the_official_deck_dimensions_controls_and_steps():
    case = setup_compositional_validation_simple_problem(
        full_schedule=True, load_external_references=False
    )
    setup = case["setup"]
    assert setup.grid.cells.num == 1000
    assert setup.mixture.names == ("n-Decane", "CarbonDioxide", "Methane")
    assert setup.state.pressure.shape == (1000,)
    assert setup.state.z.shape == (1000, 3)
    assert [well.type for well in setup.wells] == ["bhp", "bhp"]
    assert np.allclose([well.val for well in setup.wells], [100.0e5, 50.0e5])
    steps = case["schedule"].step.val
    assert steps.shape == (500,)
    assert np.allclose(steps[:5], np.array([0.01, 0.02, 0.04, 0.04, 0.08]) * 86400.0)
    assert np.allclose(steps[-473:], 15.0 * 86400.0)


def test_external_metrics_are_exact_for_identical_histories_and_refuse_cell_mismatch():
    state = SimpleNamespace(
        pressure=np.array([5.0e6, 6.0e6]),
        z=np.array([[0.6, 0.1, 0.3], [0.5, 0.2, 0.3]]),
        sT=np.array([0.2, 0.4]),
    )
    reference = SimpleNamespace(
        pressure=state.pressure.copy(),
        components=state.z.copy(),
        saturation=np.column_stack([1.0 - state.sT, state.sT]),
    )
    output = {
        "result": SimpleNamespace(states=[state]),
        "external_references": {"E300": [reference], "AD-GPRS": [reference]},
    }
    metrics = external_comparison_metrics(output)
    assert metrics["E300"]["available"] is True
    assert metrics["E300"]["pressure_rmse_bar"] == 0.0
    assert metrics["E300"]["composition_mae"] == 0.0
    assert metrics["E300"]["gas_saturation_mae"] == 0.0

    short = SimpleNamespace(pressure=np.array([5.0e6]), z=np.array([[0.6, 0.1, 0.3]]), sT=np.array([0.2]))
    output["result"] = SimpleNamespace(states=[short])
    refused = external_comparison_metrics(output)
    assert refused["E300"]["available"] is False
    assert "cell-count mismatch" in refused["E300"]["reason"]
