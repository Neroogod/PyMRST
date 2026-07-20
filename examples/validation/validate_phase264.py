"""Phase 264: getDefaultFlashNonLinearSolver utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver  # noqa: E402
from pymrst.modules.compositional.utils import (  # noqa: E402
    getDefaultFlashNonLinearSolver,
    get_default_flash_nonlinear_solver,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    default = getDefaultFlashNonLinearSolver()
    kwargs = get_default_flash_nonlinear_solver(maxIterations=77, useRelaxation=False)
    pairs = getDefaultFlashNonLinearSolver("maxTimestepCuts", 3, "continueOnFailure", False)
    opts = getDefaultFlashNonLinearSolver({"errorOnFailure": True, "verbose": 2})

    print("\nPhase 264: getDefaultFlashNonLinearSolver utility parity")
    failures = 0
    failures += _check("factory returns the real AD-core NonLinearSolver",
                       isinstance(default, NonLinearSolver))
    failures += _check("MRST flash defaults are applied",
                       default.verbose == -1
                       and default.max_iterations == 500
                       and default.maxIterations == 500
                       and default.max_timestep_cuts == 0
                       and default.maxTimestepCuts == 0
                       and default.use_relaxation is True
                       and default.useRelaxation is True
                       and default.continueOnFailure is True
                       and default.errorOnFailure is False)
    failures += _check("camelCase keyword options update native attributes",
                       kwargs.max_iterations == 77
                       and kwargs.maxIterations == 77
                       and kwargs.use_relaxation is False
                       and kwargs.useRelaxation is False)
    failures += _check("name/value pair options follow MRST merge_options style",
                       pairs.max_timestep_cuts == 3
                       and pairs.maxTimestepCuts == 3
                       and pairs.continueOnFailure is False)
    failures += _check("dict options are accepted by the Pythonic alias surface",
                       opts.errorOnFailure is True and opts.verbose == 2)

    if failures:
        print(f"\nPhase 264 failed ({failures} checks).")
        return 1
    print("\ngetDefaultFlashNonLinearSolver now configures PyMRST's real nonlinear solver with MRST flash defaults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
