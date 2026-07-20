"""Phase 277: MRST-style setupSimpleCompositionalExample API."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.utils import (  # noqa: E402
    SimpleCompositionalSetup,
    setupSimpleCompositionalExample,
    setup_simple_compositional_example,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    py_setup = setup_simple_compositional_example(use_natural=False, nx=5, nstep=2)
    state_o, model_o, schedule_o, ref_o = setupSimpleCompositionalExample(False, nx=5, nstep=2)
    state_n, model_n, schedule_n, ref_n = setupSimpleCompositionalExample(True, nx=5, nstep=2)

    print("\nPhase 277: MRST-style setupSimpleCompositionalExample API")
    failures = 0
    failures += _check("Pythonic setup still returns the rich setup object",
                       isinstance(py_setup, SimpleCompositionalSetup))
    failures += _check("MATLAB-style alias returns state/model/schedule/ref tuple",
                       state_o.z.shape == (5, 3)
                       and isinstance(model_o, OverallCompositionModel)
                       and schedule_o.step.val.size == 2
                       and isinstance(ref_o, dict))
    failures += _check("useNatural selects the natural-variable model",
                       isinstance(model_n, NaturalVariablesCompositionalModel)
                       and state_n.z.shape == (5, 3)
                       and schedule_n.step.val.size == 2
                       and isinstance(ref_n, dict))
    failures += _check("MRST injection composition and pure-gas compi are preserved",
                       all(np.allclose(w.components, [0.0, 1.0, 0.0]) for w in schedule_o.control[0].wells)
                       and all(np.allclose(w.compi, [1.0, 0.0]) for w in schedule_o.control[0].wells))
    failures += _check("reference slot mirrors MRST optional comparison output",
                       ref_o.get("mrst_file") == "comparison.mat" and ref_o.get("available") is False)

    if failures:
        print(f"\nPhase 277 failed ({failures} checks).")
        return 1
    print("\nsetupSimpleCompositionalExample now supports MRST's state/model/schedule/ref return shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
