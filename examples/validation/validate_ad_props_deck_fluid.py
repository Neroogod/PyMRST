"""Check the whole deck-to-fluid pipeline against MRST.

`validate_ad_props_assign.py` feeds each assign function a fixture built by
hand. This runs the real thing: a deck file through `read_eclipse_deck`,
`convert_deck_units` and `init_deck_adi_fluid`, compared against MRST doing
the same with its own reader.

That is the check that cannot be faked by consistent-but-wrong code on either
side, and it is the only one that would catch a defect in one stage cancelling
a defect in another.

Run `dump_ad_props_deck_fluid_reference.m` in MATLAB first.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_adi_fluid
from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "ref_ad_props_deck_fluid.mat"
DATA = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/examples/data")
TOLERANCE = 1e-10


def field(entry, name):
    if name not in (entry.dtype.names or ()):
        return None
    return entry[name][0, 0]


def check(failures, label, got, want) -> None:
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    if got.shape != want.shape:
        failures.append(f"{label}: shape {got.shape} vs MRST {want.shape}")
        return
    if not np.allclose(got, want, rtol=TOLERANCE, atol=0, equal_nan=True):
        error = np.nanmax(np.abs(got - want))
        scale = max(np.nanmax(np.abs(want)), 1e-30)
        failures.append(f"{label}: max abs diff {error:.3e} "
                        f"(relative {error / scale:.3e})")
        return
    print(f"  {label:22} {got.size:3d} value(s)  ok")


def compare(name: str, entry) -> list:
    path = DATA / name / f"BENCH_{name}.DATA"
    deck = convert_deck_units(read_eclipse_deck(path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fluid = init_deck_adi_fluid(deck)
    failures = [f"{name}: assignment warned: {w.message}" for w in caught]

    p = field(entry, "p").ravel()
    s = field(entry, "s").ravel()
    live_oil = bool(field(entry, "live_oil").ravel()[0])
    wet_gas = bool(field(entry, "wet_gas").ravel()[0])

    print(f"\n{name}: {'live' if live_oil else 'dead'} oil, "
          f"{'wet' if wet_gas else 'dry'} gas")

    # The set of fields MRST produced must be the set we produce.
    mrst_fields = {str(f[0]) for f in field(entry, "fields")[0]}
    ours = set(fluid)
    missing = sorted(mrst_fields - ours)
    extra = sorted(ours - mrst_fields)
    if missing:
        failures.append(f"{name}: fields MRST has and we do not: {missing}")
    if extra:
        failures.append(f"{name}: fields we have and MRST does not: {extra}")
    if not missing and not extra:
        print(f"  {'field set':22} {len(mrst_fields):3d} names    ok")

    check(failures, f"{name} bW", fluid["bW"](p), field(entry, "bW"))
    check(failures, f"{name} muW", fluid["muW"](p), field(entry, "muW"))
    check(failures, f"{name} pvMultR", fluid["pvMultR"](p),
          field(entry, "pvMultR"))
    for phase in ("rhoWS", "rhoOS", "rhoGS"):
        check(failures, f"{name} {phase}", fluid[phase], field(entry, phase))

    if live_oil:
        rs = fluid["rsSat"](p)
        check(failures, f"{name} rsSat", rs, field(entry, "rsSat"))
        check(failures, f"{name} pb", fluid["pb"](rs), field(entry, "pb"))
        saturated = np.ones(p.size, dtype=bool)
        check(failures, f"{name} bO sat", fluid["bO"](p, rs, saturated),
              field(entry, "bO"))
        check(failures, f"{name} muO sat", fluid["muO"](p, rs, saturated),
              field(entry, "muO"))
        unsaturated = np.zeros(p.size, dtype=bool)
        check(failures, f"{name} bO usat",
              fluid["bO"](p, rs / 2, unsaturated), field(entry, "bO_usat"))
        check(failures, f"{name} muO usat",
              fluid["muO"](p, rs / 2, unsaturated), field(entry, "muO_usat"))
    else:
        check(failures, f"{name} bO", fluid["bO"](p), field(entry, "bO"))
        check(failures, f"{name} muO", fluid["muO"](p), field(entry, "muO"))

    if wet_gas:
        rv = fluid["rvSat"](p)
        check(failures, f"{name} rvSat", rv, field(entry, "rvSat"))
        saturated = np.ones(p.size, dtype=bool)
        check(failures, f"{name} bG sat", fluid["bG"](p, rv, saturated),
              field(entry, "bG"))
        check(failures, f"{name} muG sat", fluid["muG"](p, rv, saturated),
              field(entry, "muG"))
        unsaturated = np.zeros(p.size, dtype=bool)
        check(failures, f"{name} bG usat",
              fluid["bG"](p, rv / 2, unsaturated), field(entry, "bG_usat"))
        check(failures, f"{name} muG usat",
              fluid["muG"](p, rv / 2, unsaturated), field(entry, "muG_usat"))
    else:
        check(failures, f"{name} bG", fluid["bG"](p), field(entry, "bG"))
        check(failures, f"{name} muG", fluid["muG"](p), field(entry, "muG"))

    for curve in ("krW", "krG", "krOW", "krOG"):
        check(failures, f"{name} {curve}", fluid[curve](s),
              field(entry, curve))
    for curve in ("pcOW", "pcOG"):
        want = field(entry, curve)
        if want is None:
            if curve in fluid:
                failures.append(f"{name}: {curve} present but MRST has none")
            continue
        if curve not in fluid:
            failures.append(f"{name}: MRST has {curve} and we do not")
            continue
        check(failures, f"{name} {curve}", fluid[curve](s), want)

    for part in ("w", "ow", "g", "og"):
        want = field(entry, f"krPts_{part}")
        if want is None:
            continue
        check(failures, f"{name} krPts {part}", fluid["krPts"][part], want)
    return failures


def main() -> int:
    if not REFERENCE.exists():
        print(f"missing {REFERENCE.name}; run "
              f"dump_ad_props_deck_fluid_reference.m in MATLAB first")
        return 2
    reference = loadmat(REFERENCE, struct_as_record=True)
    failures = []
    for name in ("SPE1", "SPE3", "SPE9"):
        if name not in reference:
            print(f"{name}: absent from the reference file")
            continue
        failures += compare(name, reference[name])

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("The whole deck-to-fluid pipeline matches MRST on SPE1, SPE3, SPE9.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
