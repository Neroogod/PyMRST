"""MRST ``ad-core/utils/CNV_MBConvergence.m``: the CNV and material-balance
convergence measures of a black-oil problem.

For each phase, with `B` the inverse shrinkage factors and `R` the phase's
equation: `CNV = mean(B) dt max(|R|/pv)` and `MB = dt |mean(B) sum(R)|/sum(pv)`,
compared with the model's `toleranceCNV` and `toleranceMB`. Returns
`(converged, values, evaluated, names)`, CNV first, then MB.

MRST 2024a's function reads the shrinkage factors with
`FlowPropertyFunctions.getProperty`, which today's state-function groupings
no longer have: for a current model it stops with "Unrecognized method,
property, or field 'getProperty'". The port calls the same method and stops
the same way (AttributeError); given a grouping that has it, it computes
MRST's measures.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def cnv_mb_convergence(model: Any, problem: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """MRST `[converged, values, evaluated, names] = CNV_MBConvergence(model, problem)`."""
    from pymrst.utils.adi import value
    state = problem.state
    pv = np.asarray(model.operators.pv, dtype=float)
    pvsum = pv.sum()
    tol_mb = model.toleranceMB
    tol_cnv = model.toleranceCNV
    evaluated = np.zeros(problem.numeq(), dtype=bool)
    b = model.FlowPropertyFunctions.getProperty(model, state, "ShrinkageFactors")
    dt = problem.dt
    ph, phase_names = model.get_phase_names()
    CNV = np.zeros(len(ph))
    MB = np.zeros(len(ph))
    for i in range(len(b)):
        sub = _index_of_equation_name(problem, phase_names[i])
        B = 1/np.asarray(value(b[i]), dtype=float)
        R = np.asarray(value(problem.equations[sub]), dtype=float)
        B_avg = np.mean(B)
        CNV[i] = B_avg*dt*np.max(np.abs(R)/pv)
        MB[i] = dt*abs(B_avg*np.sum(R))/pvsum
        evaluated[sub] = True
    converged = np.concatenate([CNV <= tol_cnv, MB <= tol_mb])
    values = np.concatenate([CNV, MB])
    names = [f"CNV_{x}" for x in ph] + [f"MB_{x}" for x in ph]
    return converged, values, evaluated, names


def _index_of_equation_name(problem: Any, name: str) -> int:
    index = problem.index_of_equation_name(name)
    if isinstance(index, np.ndarray):
        return int(np.flatnonzero(index)[0])
    return int(index)


CNV_MBConvergence = cnv_mb_convergence
__all__ = ["CNV_MBConvergence", "cnv_mb_convergence"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.ad_core.solvers.linearized_problem import LinearizedProblem

    def test_cnv_and_mb_per_phase():
        model = SimpleNamespace(
            operators=SimpleNamespace(pv=np.array([1.0, 3.0])), toleranceMB=1e-7, toleranceCNV=1e-3,
            FlowPropertyFunctions=SimpleNamespace(
                getProperty=lambda m, s, n: [np.array([1.0, 1.0]), np.array([0.5, 0.5])]),
            get_phase_names=lambda: (["W", "O"], ["water", "oil"]))
        problem = LinearizedProblem([np.array([1e-4, -2e-4]), np.array([0.0, 3e-4])], ["cell", "cell"],
                                    ["water", "oil"], ["p", "sw"], {}, 2.0)
        converged, values, evaluated, names = cnv_mb_convergence(model, problem)
        assert names == ["CNV_W", "CNV_O", "MB_W", "MB_O"] and evaluated.all()
        assert np.allclose(values, [2*1e-4, 2*2*1e-4, 2*1e-4/4, 2*2*3e-4/4])
        assert np.array_equal(converged, [True, True, False, False])
        print("ok test_cnv_and_mb_per_phase")

    test_cnv_and_mb_per_phase()
    print("\nAll tests passed!")
