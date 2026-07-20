"""Phase 309: MRST FacilityModel convergence helper parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class TinyProblem:
    def __init__(self):
        self.equations = [
            np.array([1.0e-7, -2.0e-7]),
            np.array([3.0e-7]),
            np.array([2.0e-6]),
            np.array([4.0e-7]),
            np.array([9.0e-7]),
            np.array([99.0]),
        ]
        self.equationNames = ["perf_eq", "well_eq", "seg_eq", "node_eq", "alpha_eq", "cell_eq"]
        self.types = ["perf", "well", "seg", "node", "alpha", "cell"]

    def indexOfType(self, name):
        return np.asarray([typ == name for typ in self.types], dtype=bool)


def validate_facility_convergence_helpers() -> int:
    facility = FacilityModel([])
    problem = TinyProblem()
    values, names, tolerances, evaluated = facility.getConvergenceValuesWells(problem)
    convergence, c_values, c_names, c_evaluated = facility.checkFacilityConvergence(problem)
    fv, ftol, fnames, feval = facility.getFacilityConvergenceValues(problem)
    gv, gtol, gnames = facility.getConvergenceValues(problem)

    dict_problem = {
        "equations": problem.equations,
        "equation_names": problem.equationNames,
        "types": problem.types,
    }
    d_values, d_names, d_tolerances, d_evaluated = facility.get_convergence_values_wells(dict_problem)

    failures = 0
    failures += _check("facility convergence evaluates MRST facility equation types only",
                       np.array_equal(evaluated, [True, True, True, True, True, False]))
    failures += _check("residual values are infinity norms in equation order",
                       np.allclose(values, [2.0e-7, 3.0e-7, 2.0e-6, 4.0e-7, 9.0e-7]))
    failures += _check("names match MRST '<equation> (<type>)' format",
                       names == ["perf_eq (perf)", "well_eq (well)", "seg_eq (seg)",
                                 "node_eq (node)", "alpha_eq (alpha)"])
    failures += _check("rate and multisegment tolerances follow FacilityModel defaults",
                       np.allclose(tolerances[:2], facility.toleranceWellRate)
                       and np.allclose(tolerances[2:], facility.toleranceWellMS))
    failures += _check("checkFacilityConvergence returns flags, values, names and mask",
                       np.array_equal(convergence, values < tolerances)
                       and np.allclose(c_values, values)
                       and c_names == names
                       and np.array_equal(c_evaluated, evaluated))
    failures += _check("getFacilityConvergenceValues and getConvergenceValues aliases are MRST-shaped",
                       np.allclose(fv, values)
                       and np.allclose(ftol, tolerances)
                       and fnames == names
                       and np.array_equal(feval, evaluated)
                       and np.allclose(gv, values)
                       and np.allclose(gtol, tolerances)
                       and gnames == names)
    failures += _check("dictionary problems without indexOfType use type strings",
                       np.allclose(d_values, values)
                       and d_names == names
                       and np.allclose(d_tolerances, tolerances)
                       and np.array_equal(d_evaluated, evaluated))
    return failures


def main() -> int:
    print("\nPhase 309: MRST FacilityModel convergence helper parity")
    failures = validate_facility_convergence_helpers()
    if failures:
        print(f"\nPhase 309 failed ({failures} checks).")
        return 1
    print("\nFacilityModel convergence helper methods are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
