"""MRST ``ad-core/utils/getConvergenceValuesWells.m``: the infinity norm of
each well equation (types `perf`, `well`, `seg`, `node`, `alpha`), with the
model's `toleranceWellRate` for perforation and well equations and
`toleranceWellMS` for multisegment ones, named `'<equation> (<type>)'`.
Returns `(values, names, tolerances, evaluated)`."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def get_convergence_values_wells(model: Any, problem: Any) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """MRST `[values, names, tolerances, evaluated] = getConvergenceValuesWells(model, problem)`."""
    from pymrst.utils.adi import value
    isperf = _mask(problem, "perf")
    iswell = _mask(problem, "well")
    isseg = _mask(problem, "seg")
    isnode = _mask(problem, "node")
    isalpha = _mask(problem, "alpha")
    evaluated = isperf | iswell | isseg | isnode | isalpha
    tmp = np.flatnonzero(evaluated)
    values = np.array([np.linalg.norm(np.ravel(value(problem.equations[i])), np.inf) for i in tmp])
    tolerances = np.zeros(tmp.size)
    tolerances[isperf[tmp]] = _tol(model, "toleranceWellRate")
    tolerances[iswell[tmp]] = _tol(model, "toleranceWellRate")
    tolerances[isseg[tmp]] = _tol(model, "toleranceWellMS")
    tolerances[isnode[tmp]] = _tol(model, "toleranceWellMS")
    tolerances[isalpha[tmp]] = _tol(model, "toleranceWellMS")
    names = [f"{_names(problem)[i]} ({problem.types[i]})" for i in tmp]
    return values, names, tolerances, evaluated


def _mask(problem: Any, type_name: str) -> np.ndarray:
    return np.array([str(t).lower() == type_name for t in problem.types], dtype=bool)


def _names(problem: Any) -> List[str]:
    return list(getattr(problem, "equationNames", None) or problem.equation_names)


def _tol(model: Any, name: str) -> float:
    if hasattr(model, name):
        return getattr(model, name)
    import re
    return getattr(model, re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower())


getConvergenceValuesWells = get_convergence_values_wells
__all__ = ["getConvergenceValuesWells", "get_convergence_values_wells"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.ad_core.solvers.linearized_problem import LinearizedProblem

    def test_well_equations_and_their_tolerances():
        problem = LinearizedProblem([np.ones(3), np.array([-4.0, 2.0]), np.array([0.5]), np.array([7.0])],
                                    ["cell", "perf", "well", "seg"], ["water", "waterWells",
                                                                      "closureWells", "segs"],
                                    ["p", "qw", "bhp", "ps"], {})
        model = SimpleNamespace(toleranceWellRate=1e-3, toleranceWellMS=1e-6)
        values, names, tolerances, evaluated = get_convergence_values_wells(model, problem)
        assert np.array_equal(evaluated, [False, True, True, True])
        assert np.allclose(values, [4.0, 0.5, 7.0]) and np.allclose(tolerances, [1e-3, 1e-3, 1e-6])
        assert names == ["waterWells (perf)", "closureWells (well)", "segs (seg)"]
        print("ok test_well_equations_and_their_tolerances")

    test_well_equations_and_their_tolerances()
    print("\nAll tests passed!")
