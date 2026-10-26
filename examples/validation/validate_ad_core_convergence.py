"""Validate MRST's `CNV_MBConvergence`, `getConvergenceValuesWells` and
`printConvergenceReport`: the ports in `ad_core/utils/`.

The reference is MRST 2024a (`dump_ad_core_convergence_reference.m`) on the
first Newton system of the two-well oil-water model of
`validate_ad_core_linearized_problem.py`:
- `CNV_MBConvergence` stops in MRST 2024a (its groupings have no
  `getProperty`) and the port stops at the same call;
- `getConvergenceValuesWells` on the facility model: values, names,
  tolerances and the evaluated mask;
- `printConvergenceReport`: the text of three rows (a first iteration, a
  converged one, and one with Inf and NaN closing the block), character for
  character, from MRST's own names and values.
"""

import contextlib
import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def main() -> int:
    import sys
    sys.path.insert(0, str(HERE))
    from validate_ad_core_linearized_problem import build_problem

    from pymrst.modules.ad_core.utils.cnv_mb_convergence import cnv_mb_convergence
    from pymrst.modules.ad_core.utils.get_convergence_values_wells import (
        get_convergence_values_wells,
    )
    from pymrst.modules.ad_core.utils.print_convergence_report import print_convergence_report
    from pymrst.params.gravity.gravity import isolated_gravity

    ref = loadmat(HERE/"ref_ad_core_convergence.mat", squeeze_me=True, struct_as_record=False)
    with isolated_gravity():
        problem, model = build_problem(return_model=True)
    ok = report("getProperty" in str(ref["cnv_error"]),
                f"MRST's CNV_MBConvergence stops: {ref['cnv_error']!s}")
    try:
        cnv_mb_convergence(model, problem)
        stopped = ""
    except AttributeError as err:
        stopped = str(err)
    ok &= report("getProperty" in stopped, f"the port stops at the same call: {stopped}")

    wells = ref["wells"]
    values, names, tolerances, evaluated = get_convergence_values_wells(model.FacilityModel, problem)
    ok &= report(names == [str(n) for n in np.atleast_1d(wells.names)],
                 f"getConvergenceValuesWells: names {names}")
    err = np.abs(values - wells.values).max()/np.abs(wells.values).max()
    ok &= report(err < RTOL, f"getConvergenceValuesWells: values rel. diff {err:.1e}")
    ok &= report(np.array_equal(tolerances, np.atleast_1d(wells.tolerances))
                 and np.array_equal(evaluated, np.atleast_1d(wells.evaluated).astype(bool)),
                 "getConvergenceValuesWells: tolerances and evaluated mask identical")

    allnames = [str(n) for n in np.atleast_1d(ref["names"])] + [str(n) for n in np.atleast_1d(wells.names)]
    v = np.concatenate([np.atleast_1d(ref["values"]), np.atleast_1d(wells.values)])
    conv = np.concatenate([np.atleast_1d(ref["converged"]).astype(bool),
                           np.atleast_1d(wells.values) <= np.atleast_1d(wells.tolerances)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_convergence_report(allnames, v, conv, 1)
        print_convergence_report(allnames, np.concatenate([np.atleast_1d(ref["values"])/10,
                                                           np.atleast_1d(wells.values)/100]),
                                 np.ones(len(allnames), dtype=bool), 2)
        print_convergence_report(allnames, np.concatenate([[np.inf, np.nan],
                                                           np.atleast_1d(ref["values"])[2:],
                                                           np.atleast_1d(wells.values)]),
                                 np.zeros(len(allnames), dtype=bool), 3, True)
    same = buf.getvalue() == str(ref["txt"])
    ok &= report(same, "printConvergenceReport: the text, character for character")
    if not same:
        print(buf.getvalue())
        print(str(ref["txt"]))
    print("\nThe convergence utilities match MRST" if ok
          else "\nThe convergence utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
