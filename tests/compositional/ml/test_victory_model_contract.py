from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pymrst.modules.compositional.ML_models.victory_flash import (
    MODEL_ROOT,
    OUTPUT_COLUMNS,
    VictoryFlashEnsemble,
    load_reference_rows,
    phase_flags_from_liquid_fraction,
    victory_mixture,
)


def test_beta22_root_checkpoints_match_web_app_copies():
    for fold in range(1, 8):
        name = f"model_runtest_beta2.2_fold{fold}.pt"
        root = MODEL_ROOT / name
        web = MODEL_ROOT / "Web_app" / "models" / "beta2.2" / name
        assert root.is_file() and web.is_file()
        assert hashlib.sha256(root.read_bytes()).digest() == hashlib.sha256(web.read_bytes()).digest()


def test_ensemble_contract_and_fixed_scaling_are_batch_invariant():
    data = load_reference_rows(rows=4, seed=19)
    model = VictoryFlashEnsemble()
    assert len(model.models) == 7
    assert OUTPUT_COLUMNS == ("x1", "x3", "x14", "y1", "y3", "y14", "L")
    one = model.predict(data["p"][:1], data["z"][:1], liquid_fraction_for_flags=data["L"][:1])
    batch = model.predict(data["p"], data["z"], liquid_fraction_for_flags=data["L"])
    assert np.allclose(one.mean[0], batch.mean[0], rtol=0.0, atol=2e-7)
    assert np.allclose(one.std[0], batch.std[0], rtol=0.0, atol=2e-7)


def test_phase_flags_and_domain_rejections_are_explicit():
    flags = phase_flags_from_liquid_fraction(np.array([0.0, 1.0, 0.4]))
    assert np.array_equal(flags, [[1, 0], [0, 0], [0, 1]])
    model = VictoryFlashEnsemble()
    with pytest.raises(ValueError, match="353"):
        model.predict([1e6], [[0.5, 0.3, 0.2]], liquid_fraction_for_flags=[0.5], temperature=350.0)
    with pytest.raises(ValueError, match="outside"):
        model.predict([1e9], [[0.5, 0.3, 0.2]], liquid_fraction_for_flags=[0.5])


def test_victory_mixture_matches_recovered_training_constants():
    mix = victory_mixture()
    assert mix.names == ("Methane", "n-Propane", "n-Tetradecane")
    assert np.allclose(mix.Tcrit, [190.58, 369.82, 723.0])
    assert np.allclose(mix.Pcrit, [4.604e6, 4.249e6, 1.620e6])
    assert np.allclose(mix.acentric, [0.0108, 0.1520, 0.8])
    assert np.allclose(mix.bic, mix.bic.T)
    assert np.allclose(mix.bic[np.triu_indices(3, 1)], [0.00683, 0.01230, 0.00082])
