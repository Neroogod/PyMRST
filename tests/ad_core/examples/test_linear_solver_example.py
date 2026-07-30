"""Tests for the runnable MRST ``linearSolverExample`` counterpart."""

import pytest

from pymrst.modules.ad_core.examples import linear_solver_example


def test_native_solvers_recover_the_same_sparse_solution():
    results = linear_solver_example(size=40, seed=4)

    assert [result.solver for result in results] == [
        "BackslashSolverAD",
        "GMRES_ILUSolverAD",
        "CPRSolverAD",
    ]
    for result in results:
        assert result.relative_residual < 1e-9
        assert result.relative_error < 1e-9
        assert result.elapsed > 0.0
    assert results[0].iterations == 1
    assert results[1].iterations >= 1


def test_solver_example_rejects_degenerate_system():
    with pytest.raises(ValueError):
        linear_solver_example(size=1)
