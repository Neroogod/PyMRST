"""Phase 266: FastAD and initVariablesFastAD utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.utils import (  # noqa: E402
    FastAD,
    fast_ad_exp,
    fast_ad_log,
    fast_ad_value,
    fast_max,
    fast_sum,
    initVariablesFastAD,
    init_variables_fast_ad,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    a0 = np.array([1.5, 2.0, 2.5])
    b0 = np.array([0.25, 0.50, 0.75])
    a, b = initVariablesFastAD(a0, b0)
    a2, b2 = init_variables_fast_ad(a0, b0)

    expr = fast_ad_exp(fast_ad_log(a * b + 2.0)) + a**2 / b
    expr_val = np.exp(np.log(a0 * b0 + 2.0)) + a0**2 / b0
    d_da = b0 + 2.0 * a0 / b0
    d_db = a0 - a0**2 / b0**2

    summed = fast_sum(expr)
    maxed = fast_max(expr)
    pair_max = fast_max(a, b)

    print("\nPhase 266: FastAD and initVariablesFastAD utility parity")
    failures = 0
    failures += _check("initVariablesFastAD creates dense identity derivative blocks",
                       isinstance(a, FastAD)
                       and isinstance(b, FastAD)
                       and np.allclose(a.jac, np.column_stack([np.ones(3), np.zeros(3)]))
                       and np.allclose(b.jac, np.column_stack([np.zeros(3), np.ones(3)])))
    failures += _check("snake_case initializer matches camelCase",
                       np.allclose(a2.val, a.val)
                       and np.allclose(b2.jac, b.jac))
    failures += _check("arithmetic, log, exp, power, and division propagate values",
                       np.allclose(fast_ad_value(expr), expr_val))
    failures += _check("arithmetic, log, exp, power, and division propagate Jacobians",
                       np.allclose(expr.jac[:, 0], d_da)
                       and np.allclose(expr.jac[:, 1], d_db))
    failures += _check("sum reduces values and sums Jacobian rows",
                       np.allclose(summed.val, [expr_val.sum()])
                       and np.allclose(summed.jac, np.array([[d_da.sum(), d_db.sum()]])))
    failures += _check("max selects the derivative row of the maximum value",
                       np.allclose(maxed.val, [expr_val.max()])
                       and np.allclose(maxed.jac, expr.jac[[int(np.argmax(expr_val))], :]))
    failures += _check("binary max chooses derivatives from the selected operand",
                       np.allclose(pair_max.val, np.maximum(a0, b0))
                       and np.allclose(pair_max.jac, a.jac))
    failures += _check("comparison operators act on numeric values",
                       np.array_equal(a > b, a0 > b0)
                       and np.array_equal(a <= 2.0, a0 <= 2.0))

    if failures:
        print(f"\nPhase 266 failed ({failures} checks).")
        return 1
    print("\nFastAD now provides MRST-style dense derivative propagation for quick EOS assembly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
