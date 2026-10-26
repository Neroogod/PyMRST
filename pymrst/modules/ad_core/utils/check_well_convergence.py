"""Well-equation convergence of a linearized problem (MRST
`checkWellConvergence` / `getConvergenceValuesWells`).

Ported as its own module because it is its own function. PyMRST already had
`convergence.well_control_residual`, and an earlier pass in this port recorded
that as the counterpart - which was wrong in a way worth naming, because it is
the kind of mistake a parity ledger cannot see:

* `well_control_residual(well_residuals, control_residual, wells)` takes
  *arrays* and returns two *scalars* - a scaled rate residual and a scaled
  control residual, for a model that already knows which is which;
* `check_well_convergence(model, problem)` takes a **linearized problem** and
  works out which of its equations are well equations *from their declared
  types*, then returns a per-equation verdict.

They compute different things for different callers. Recording one as the port
of the other made a real gap invisible.

**What the types mean.** MRST tags every equation in a linearized problem, and
five of those tags are wells: `perf` (a perforation rate), `well` (a well
closure), and `seg`/`node`/`alpha` (multisegment wellbore unknowns). Reservoir
equations - tagged `cell` - are deliberately excluded: they converge on the CNV
and material-balance criteria instead, which are scaled quite differently, and
mixing the two makes whichever is stricter decide every step.

**Why the tolerances differ.** Perforation and closure equations use
`tolerance_well_rate`; the multisegment unknowns use `tolerance_well_ms`. A
segment-flow residual is not comparable to a surface rate, so one tolerance for
both would be either far too tight for one or far too loose for the other.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from pymrst.utils.adi import value

#: Equation types that count as well equations, and which tolerance each takes.
#: `cell` is absent on purpose - see the module docstring.
_RATE_TYPES = ("perf", "well")
_MULTISEGMENT_TYPES = ("seg", "node", "alpha")


def check_well_convergence(model: Any, problem: Any
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                      List[str], np.ndarray]:
    """`(convergence, values, evaluated, names, tolerances)`, as MRST returns.

    `convergence[i]` is `values[i] < tolerances[i]` for the well equations of
    `problem`; `evaluated` is the mask selecting them out of every equation.
    """
    values, names, tolerances, evaluated = get_convergence_values_wells(
        model, problem)
    return values < tolerances, values, evaluated, names, tolerances


checkWellConvergence = check_well_convergence


def get_convergence_values_wells(model: Any, problem: Any
                                 ) -> Tuple[np.ndarray, List[str],
                                            np.ndarray, np.ndarray]:
    """`(values, names, tolerances, evaluated)` for a problem's well equations.

    The residual norm is the **infinity** norm, not the 2-norm: a well is
    converged when its worst equation is small, and a 2-norm would let one bad
    perforation hide behind many good ones in a long completion.
    """
    types = [str(t).lower() for t in getattr(problem, "types", [])]
    is_rate = np.array([t in _RATE_TYPES for t in types], dtype=bool)
    is_ms = np.array([t in _MULTISEGMENT_TYPES for t in types], dtype=bool)
    evaluated = is_rate | is_ms

    selected = np.flatnonzero(evaluated)
    equations = list(getattr(problem, "equations", []))
    values = np.array(
        [float(np.max(np.abs(np.asarray(value(equations[i]),
                                        dtype=float).ravel()), initial=0.0))
         for i in selected])

    tolerances = np.zeros(selected.size)
    rate_tol = float(getattr(model, "tolerance_well_rate",
                             getattr(model, "toleranceWellRate", 1e-6)))
    ms_tol = float(getattr(model, "tolerance_well_ms",
                           getattr(model, "toleranceWellMS", 1e-6)))
    tolerances[is_rate[selected]] = rate_tol
    tolerances[is_ms[selected]] = ms_tol

    equation_names = list(getattr(problem, "equation_names",
                                  getattr(problem, "equationNames", [])))
    names = [f"{_at(equation_names, i, f'eq{i}')} ({types[i]})"
             for i in selected]
    return values, names, tolerances, evaluated


getConvergenceValuesWells = get_convergence_values_wells


def _at(seq: Sequence[Any], index: int, default: Any) -> Any:
    return seq[index] if index < len(seq) else default


__all__ = ["check_well_convergence", "checkWellConvergence",
           "get_convergence_values_wells", "getConvergenceValuesWells"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _problem(types, residuals):
        return SimpleNamespace(
            types=list(types), equations=[np.asarray(r, float) for r in residuals],
            equation_names=[f"e{i}" for i in range(len(types))])

    MODEL = SimpleNamespace(tolerance_well_rate=1e-6, tolerance_well_ms=1e-3)

    def test_only_well_equations_are_judged():
        """Reservoir equations converge on CNV and material balance, which are
        scaled differently; including them lets one criterion decide both."""
        problem = _problem(["cell", "perf", "well", "cell"],
                           [[1e3], [1e-9], [1e-9], [1e3]])
        conv, values, evaluated, names, tol = check_well_convergence(
            MODEL, problem)
        assert list(evaluated) == [False, True, True, False]
        assert values.size == 2 and conv.all()
        print(f"ok test_only_well_equations_are_judged (names {names})")

    def test_multisegment_equations_take_their_own_tolerance():
        """A segment-flow residual is not comparable to a surface rate."""
        problem = _problem(["perf", "seg"], [[1e-9], [1e-4]])
        conv, values, _, _, tol = check_well_convergence(MODEL, problem)
        assert tol[0] == 1e-6 and tol[1] == 1e-3
        assert conv.all(), "the seg equation should pass its own tolerance"
        print("ok test_multisegment_equations_take_their_own_tolerance")

    def test_the_norm_is_the_worst_entry_not_the_average():
        """One bad perforation must not hide behind many good ones."""
        problem = _problem(["perf"], [[1e-12, 1e-12, 1e-12, 5.0]])
        conv, values, _, _, _ = check_well_convergence(MODEL, problem)
        assert values[0] == 5.0, values
        assert not conv[0]
        print("ok test_the_norm_is_the_worst_entry_not_the_average")

    def test_a_problem_with_no_well_equations_converges_vacuously():
        conv, values, evaluated, names, tol = check_well_convergence(
            MODEL, _problem(["cell", "cell"], [[1.0], [1.0]]))
        assert values.size == 0 and conv.size == 0 and not evaluated.any()
        print("ok test_a_problem_with_no_well_equations_converges_vacuously")

    test_only_well_equations_are_judged()
    test_multisegment_equations_take_their_own_tolerance()
    test_the_norm_is_the_worst_entry_not_the_average()
    test_a_problem_with_no_well_equations_converges_vacuously()
    print("\nAll tests passed!")
