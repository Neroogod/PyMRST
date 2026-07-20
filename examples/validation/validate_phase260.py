"""Phase 260: formatMassString utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.utils import formatMassString, format_mass_string  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 260: formatMassString utility parity")
    failures = 0
    failures += _check("zero mass matches MRST wording",
                       formatMassString(0.0) == "0 Kilogram")
    failures += _check("positive mixed units use MRST unit order",
                       formatMassString(1.0e9 + 2.0e3 + 3.0) == "1 Megatonne, 2 Tonnes, 3 Kilograms")
    failures += _check("negative sign is preserved",
                       formatMassString(-1001.0) == "-1 Tonne, 1 Kilogram")
    failures += _check("gram-scale values are formatted as grams",
                       format_mass_string(0.002) == "2 Grams")
    failures += _check("singular names are used for unit amounts",
                       formatMassString(1001.002) == "1 Tonne, 1 Kilogram, 1 Gram")
    failures += _check("limit uses MRST decimal-at-depth behavior",
                       formatMassString(1234.0, 1) == "1 Tonne, 234.00 Kilograms")

    if failures:
        print(f"\nPhase 260 failed ({failures} checks).")
        return 1
    print("\nformatMassString now mirrors MRST mass display formatting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
