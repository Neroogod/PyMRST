"""Check `convert_input_units` against MRST (scripts/matlab/ref_convert_units.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_convert_units_reference.m')"

One representative value per keyword is enough: the conversion is a scalar
multiply, so if the first entry agrees the array does. What is actually being
checked is the *factor* chosen for each keyword under each unit system, and
that `GRIDUNIT` overrides the system for lengths without touching permeability.

The last check is that MRST **errors** on an unrecognised keyword. That is the
behaviour worth preserving - silently passing a permeability through
unconverted is a factor-of-1e15 mistake that announces itself nowhere - so the
port keeps it, and offers `known_only=True` for callers who would rather be
told than stopped.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.inout.convert_input_units import convert_input_units   # noqa: E402


def deck():
    return {"cartDims": np.array([2, 2, 2]), "COORD": np.ones(54),
            "ZCORN": np.ones(64), "PERMX": np.full(8, 100.0),
            "PORO": np.full(8, 0.2)}


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_convert_units.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []

    for name in ("metric", "field", "lab"):
        ref = reference[name]
        got, _ = convert_input_units(deck(), name)
        for key in ("COORD", "ZCORN", "PERMX", "PORO"):
            mine = float(np.asarray(got[key]).ravel()[0])
            theirs = float(np.asarray(getattr(ref, key)).ravel()[0])
            ok = np.isclose(mine, theirs, rtol=1e-12)
            print(f"  {'ok ' if ok else 'FAIL'} {name}.{key}: "
                  f"{mine:.6e} vs {theirs:.6e}")
            if not ok:
                failures.append(f"{name}.{key}")

    d = deck()
    d["GRIDUNIT"] = ["METRES"]
    got, _ = convert_input_units(d, "field")
    ref = reference["gridunit"]
    for key in ("COORD", "PERMX"):
        mine = float(np.asarray(got[key]).ravel()[0])
        theirs = float(np.asarray(getattr(ref, key)).ravel()[0])
        ok = np.isclose(mine, theirs, rtol=1e-12)
        print(f"  {'ok ' if ok else 'FAIL'} gridunit.{key}: "
              f"{mine:.6e} vs {theirs:.6e}")
        if not ok:
            failures.append(f"gridunit.{key}")

    ran = int(reference["unknown_ran"])
    print(f"  {'ok ' if not ran else 'FAIL'} unknown keyword: MRST "
          f"{'errors, as expected' if not ran else 'accepted it'}")
    if ran:
        failures.append("unknown: expected MRST to error")

    d = deck()
    d["WIBBLE"] = np.ones(8)
    try:
        convert_input_units(d, "metric")
        print("  FAIL unknown keyword: the port accepted it")
        failures.append("unknown: port accepted")
    except ValueError:
        print("  ok  unknown keyword: the port errors too")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("convert_input_units matches MRST's convertInputUnits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
