from __future__ import annotations

import numpy as np
from scipy.io import savemat

from pymrst.modules.compositional.ML_models.compare_with_pymrst import (
    ConsoleProgress,
    FlashValues,
    export_mrst_input,
    load_mrst_values,
    pair_metrics,
    phase_codes,
    physical_metrics,
    reference_values,
    run_pymrst,
    state_digest,
)
from pymrst.modules.compositional.ML_models.victory_flash import load_reference_rows


def test_phase_aware_metrics_ignore_absent_phase_composition():
    phase = np.array([0, 1, 2])
    base = FlashValues(
        "a", np.array([0.0, 1.0, 0.5]),
        np.array([[0, 0, 0], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]),
        np.array([[0.7, 0.2, 0.1], [0, 0, 0], [0.8, 0.15, 0.05]]),
        None, None, phase,
    )
    changed = FlashValues(
        "b", base.L.copy(), base.x.copy(), base.y.copy(), None, None, phase,
    )
    # Change only absent-phase values: liquid composition on gas row and vapor
    # composition on liquid row. Every phase-aware metric must stay zero.
    changed.x[0] = [10, 20, 30]
    changed.y[1] = [10, 20, 30]
    rows = pair_metrics(changed, base, phase)
    assert all(row["mae"] == 0.0 for row in rows)


def test_mrst_fixture_roundtrip_and_digest_guard(tmp_path):
    data = load_reference_rows(rows=3, seed=5)
    input_path = tmp_path / "inputs.mat"
    digest = export_mrst_input(input_path, data)
    assert digest == state_digest(data)
    result_path = tmp_path / "mrst.mat"
    savemat(result_path, {
        "L": data["L"], "x": data["x"], "y": data["y"],
        "Z_L": data["Z_L"], "Z_V": data["Z_V"],
        "input_digest": digest, "flash_elapsed_seconds": 0.25,
    })
    result = load_mrst_values(result_path, digest, 3)
    assert result.name == "mrst"
    assert np.allclose(result.outputs, reference_values(data).outputs)


def test_reference_physical_metrics_close_material_balance():
    data = load_reference_rows(rows=20, seed=9)
    metrics = physical_metrics(reference_values(data), data)
    assert metrics["nonfinite_count"] == 0
    assert metrics["x_closure_rmse"] < 1e-12
    assert metrics["y_closure_rmse"] < 1e-12
    assert metrics["material_balance_rmse"] < 1e-4


def test_phase_code_convention():
    assert np.array_equal(phase_codes([0.0, 1.0, 0.25]), [0, 1, 2])


def test_batched_pymrst_flash_preserves_results():
    data = load_reference_rows(rows=5, seed=12)
    one_batch, _ = run_pymrst(data, batch_size=20)
    small_batches, _ = run_pymrst(data, batch_size=2)
    assert np.allclose(one_batch.outputs, small_batches.outputs, rtol=1e-9, atol=1e-10)
    assert np.allclose(one_batch.Z_L, small_batches.Z_L, rtol=1e-9, atol=1e-10)
    assert np.allclose(one_batch.Z_V, small_batches.Z_V, rtol=1e-9, atol=1e-10)


def test_console_progress_reaches_100_percent(capsys):
    progress = ConsoleProgress(True, width=8)
    progress.stage("Testing")
    progress.update(5, 10, "half")
    progress.update(10, 10, "done")
    output = capsys.readouterr().out
    assert "50.00%" in output
    assert "100.00%" in output
    assert "done" in output
