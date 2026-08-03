import math
from itertools import combinations

import numpy as np
import pytest

from pymrst.modules.ad_mechanics.utils.poro_params import (
    _relations,
    poro_params,
    solve_poro_params,
)


def test_legacy_four_argument_interface_is_preserved():
    out = poro_params(12.0e9, 0.25, 0.8, 5.0e9)
    assert out["K"] == pytest.approx(8.0e9)
    assert out["G"] == pytest.approx(4.8e9)
    assert out["K_u"] == pytest.approx(11.2e9)


def test_mrst_style_derives_drained_and_undrained_constants():
    out, history, residuals, relations = solve_poro_params(
        0.2, E=12.0e9, nu=0.25, alpha=0.8, M=5.0e9
    )
    assert out["K"] == pytest.approx(8.0e9, rel=2e-7)
    assert out["G"] == pytest.approx(4.8e9, rel=2e-7)
    assert out["K_u"] == pytest.approx(11.2e9, rel=2e-7)
    assert out["B"] == pytest.approx(0.8 * 5.0e9 / 11.2e9, rel=2e-7)
    assert history
    assert len(residuals) == len(relations)


def test_alternative_elastic_basis_round_trip():
    a = poro_params(0.25, K=9.0e9, G=4.0e9, alpha=0.75, M=6.0e9)
    b = poro_params(0.25, E=a["E"], nu=a["nu"], alpha=0.75, M=6.0e9)
    for name in ("K", "G", "E", "nu", "K_u", "B", "nu_u"):
        assert b[name] == pytest.approx(a[name], rel=2e-6)


def test_uniform_grains_derives_equal_unjacketed_moduli():
    out = poro_params(0.2, True, K=8.0e9, G=4.0e9, alpha=0.8, M=5.0e9)
    assert math.isfinite(out["K_s"])
    assert out["K_phi"] == pytest.approx(out["K_s"], rel=2e-6)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="phi"):
        poro_params(1.0, E=1.0, nu=0.2)
    with pytest.raises(KeyError, match="Unknown"):
        poro_params(0.2, nonsense=1.0)


def test_relation_graph_has_full_mrst_executable_relation_count():
    assert len(_relations(0.2, False)) == 112
    assert len(_relations(0.2, True)) == 113


def test_every_drained_elastic_pair_recovers_reference_constants():
    reference = poro_params(
        0.22, E=12.0e9, nu=0.25, alpha=0.8, M=5.0e9
    )
    names = ("K", "E", "G", "lambda", "nu", "K_v")
    for left, right in combinations(names, 2):
        if {left, right} in ({"E", "K_v"}, {"E", "lambda"}):
            # These pairs have multiple algebraic branches. MRST's small-seed
            # fzero ordering selects an auxetic branch, certified separately
            # by the exhaustive MATLAB basis fixture.
            continue
        out = poro_params(
            0.22,
            alpha=0.8,
            M=5.0e9,
            **{left: reference[left], right: reference[right]},
        )
        for name in names:
            assert out[name] == pytest.approx(reference[name], rel=3e-6), (
                left, right, name
            )


def test_every_undrained_elastic_pair_recovers_reference_constants():
    reference = poro_params(
        0.22, E=12.0e9, nu=0.25, alpha=0.8, M=5.0e9
    )
    names = ("K_u", "E_u", "G", "lambda_u", "nu_u", "K_vu")
    for left, right in combinations(names, 2):
        if {left, right} in ({"E_u", "K_vu"}, {"E_u", "lambda_u"}):
            continue
        out = poro_params(0.22, **{left: reference[left], right: reference[right]})
        for name in names:
            assert out[name] == pytest.approx(reference[name], rel=3e-6), (
                left, right, name
            )
