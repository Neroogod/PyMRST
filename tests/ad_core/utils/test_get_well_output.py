"""Tests for MRST ``getWellOutput``."""

import numpy as np
import pytest

from pymrst.modules.ad_core.utils.get_well_output import get_well_output_ad


def _solutions():
    def step(bhp_i, bhp_p, q_i, q_p):
        return [
            {"name": "I", "bhp": bhp_i, "q": np.array([q_i, 0.0])},
            {"name": "P", "bhp": bhp_p, "q": np.array([0.0, q_p])},
        ]
    return [
        step(210.0, 150.0, 2e-3, -1e-3),
        step(212.0, 149.0, 2.1e-3, -1.1e-3),
        step(214.0, 148.0, 2.2e-3, -1.2e-3),
    ]


def test_selects_fields_wells_and_one_based_vector_index():
    data, names = get_well_output_ad(
        _solutions(), ["bhp", "q"], ["P", "I"], index=1
    )
    assert names == ["P", "I"]
    assert data.shape == (3, 2, 2)
    assert np.array_equal(data[:, 0, 0], [150.0, 149.0, 148.0])
    assert np.allclose(data[:, 1, 1], [2e-3, 2.1e-3, 2.2e-3])

    oil_rate, names = get_well_output_ad(
        _solutions(), "q", ["P"], index=2
    )
    assert names == ["P"]
    assert np.allclose(oil_rate[:, 0, 0], [-1e-3, -1.1e-3, -1.2e-3])


def test_empty_output_and_invalid_selection():
    data, names = get_well_output_ad([])
    assert data.shape == (0, 0, 0) and names == []
    with pytest.raises(ValueError, match="has no value"):
        get_well_output_ad(_solutions(), "bhp", index=2)
    with pytest.raises(ValueError):
        get_well_output_ad(_solutions(), "bhp", ["missing"])
