"""Phase 262: MRST cubic utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.utils import (  # noqa: E402
    cubicPositive,
    cubic_positive,
    mrstCubic,
    mrst_cubic,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _sort_complex_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=complex).copy()
    out = np.empty_like(arr)
    for i, row in enumerate(arr):
        finite = row[np.isfinite(row.real) & np.isfinite(row.imag)]
        order = np.lexsort((finite.imag, finite.real))
        out[i, :] = np.nan + 0j
        out[i, :finite.size] = finite[order]
    return out


def _roots_rows(coeffs: np.ndarray) -> np.ndarray:
    out = np.full((coeffs.shape[0], 3), np.nan + 0j, dtype=complex)
    for i, row in enumerate(coeffs):
        roots = np.roots(row)
        out[i, :roots.size] = roots
    return out


def _finite_sorted_real(row: np.ndarray) -> np.ndarray:
    vals = np.asarray(row, float)
    return np.sort(vals[np.isfinite(vals)])


def main() -> int:
    general = np.array([
        [1.0, -6.0, 11.0, -6.0],      # 1, 2, 3
        [2.0, -4.0, -22.0, 24.0],     # -3, 1, 4 for scaled cubic
        [1.0, 0.0, 1.0, 1.0],         # one real, two complex
    ])
    roots_ref = _sort_complex_rows(_roots_rows(general))
    roots_mrst = _sort_complex_rows(mrstCubic(general))
    roots_alias = _sort_complex_rows(mrst_cubic(general[:, 0], general[:, 1], general[:, 2], general[:, 3]))

    monic_three = np.array([[-6.0, 11.0, -6.0]])
    monic_one = np.array([[0.0, 1.0, 1.0]])
    pos_three = cubicPositive(monic_three)
    pos_one = cubicPositive(monic_one)
    pos_alias = cubic_positive(monic_three[:, 0], monic_three[:, 1], monic_three[:, 2])

    print("\nPhase 262: MRST cubic utility parity")
    failures = 0
    failures += _check("mrstCubic matrix input matches numpy.roots",
                       np.allclose(roots_mrst, roots_ref, atol=1e-10, rtol=1e-10))
    failures += _check("mrst_cubic vector input matches camelCase",
                       np.allclose(roots_alias, roots_mrst, atol=1e-10, rtol=1e-10))
    failures += _check("cubicPositive returns all three real roots when present",
                       np.allclose(_finite_sorted_real(pos_three[0]), np.array([1.0, 2.0, 3.0]), atol=1e-10))
    failures += _check("cubic_positive vector alias matches camelCase",
                       np.allclose(_finite_sorted_real(pos_alias[0]), _finite_sorted_real(pos_three[0]), atol=1e-10))
    failures += _check("cubicPositive stores single real root in first column",
                       np.isfinite(pos_one[0, 0]) and np.isnan(pos_one[0, 1:]).all()
                       and np.allclose(pos_one[0, 0], np.roots([1.0, 0.0, 1.0, 1.0])[np.isreal(np.roots([1.0, 0.0, 1.0, 1.0]))].real[0]))

    if failures:
        print(f"\nPhase 262 failed ({failures} checks).")
        return 1
    print("\nMRST cubic utility functions now have native PyMRST counterparts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
