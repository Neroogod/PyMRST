"""Phase 252: MRST-style separator constructor/property parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators import BasicSeparator, EOSSeparator  # noqa: E402
from pymrst.utils.units import ATM, BARSA  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    sentinel = object()
    basic = BasicSeparator()
    eos_empty = EOSSeparator()
    eos_mrst = EOSSeparator(pressure=2.0 * ATM, T=310.0, EOSModel=sentinel)
    eos_py = EOSSeparator(pressure=ATM, T=300.0, eos_model=sentinel)
    clone = eos_mrst.clone(pressure=5.0 * BARSA, T=330.0)

    conflict_raised = False
    try:
        EOSSeparator(EOSModel=object(), eos_model=object())
    except ValueError:
        conflict_raised = True

    print("\nPhase 252: MRST-style separator constructor/property parity")
    failures = 0
    failures += _check("BasicSeparator can be constructed empty like MRST",
                       np.isnan(basic.pressure) and np.isnan(basic.T))
    failures += _check("EOSSeparator can be constructed empty like MRST",
                       np.isnan(eos_empty.pressure)
                       and np.isnan(eos_empty.T)
                       and eos_empty.EOSModel is None
                       and eos_empty.eos_model is None)
    failures += _check("EOSSeparator accepts MRST EOSModel property",
                       eos_mrst.EOSModel is sentinel
                       and eos_mrst.eos_model is sentinel
                       and np.isclose(eos_mrst.pressure, 2.0 * ATM)
                       and np.isclose(eos_mrst.T, 310.0))
    failures += _check("EOSSeparator accepts Python eos_model alias",
                       eos_py.EOSModel is sentinel and eos_py.eos_model is sentinel)
    failures += _check("EOSSeparator clone preserves EOSModel and overrides conditions",
                       clone.EOSModel is sentinel
                       and np.isclose(clone.pressure, 5.0 * BARSA)
                       and np.isclose(clone.T, 330.0))
    failures += _check("conflicting EOSModel aliases are rejected", conflict_raised)

    eos_py.eos_model = None
    failures += _check("eos_model setter updates MRST EOSModel property",
                       eos_py.EOSModel is None)

    if failures:
        print(f"\nPhase 252 failed ({failures} checks).")
        return 1
    print("\nSeparators now support MRST-style empty construction and EOSModel storage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
