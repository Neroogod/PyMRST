import numpy as np
import pytest

from pymrst.params.rock.permeability_converter import permeability_converter


def _box():
    coord = np.array([
        0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
    ], dtype=float)
    return {
        "cartDims": np.array([1, 1, 1]), "COORD": coord,
        "ZCORN": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
        "PERMX": np.array([100.0]), "PERMY": np.array([50.0]),
        "PERMZ": np.array([10.0]), "PERMXY": np.array([7.0]),
    }


def test_corrected_off_diagonal_uses_supplied_cross_term():
    out = permeability_converter(_box(), "cell", off_diagonal_mode="corrected")
    np.testing.assert_allclose(out["PERMX"], 100.0)
    np.testing.assert_allclose(out["PERMY"], 50.0)
    np.testing.assert_allclose(out["PERMZ"], 10.0)
    np.testing.assert_allclose(out["PERMXY"], 7.0)
    np.testing.assert_allclose(out["PERMYZ"], 0.0)
    np.testing.assert_allclose(out["PERMZX"], 0.0)


def test_mrst_mode_intentionally_differs_from_corrected_mode():
    with pytest.raises(NotImplementedError, match="MRST 2024a fails"):
        permeability_converter(_box(), "cell")
    corrected = permeability_converter(_box(), "cell", off_diagonal_mode="corrected")
    np.testing.assert_allclose(corrected["PERMXY"], 7.0)


def test_invalid_mode_and_keyword_lengths_are_rejected():
    with pytest.raises(ValueError, match="off_diagonal_mode"):
        permeability_converter(_box(), "cell", off_diagonal_mode="unknown")
    deck = _box()
    deck["PERMXY"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="PERMXY"):
        permeability_converter(deck, "cell", off_diagonal_mode="corrected")
