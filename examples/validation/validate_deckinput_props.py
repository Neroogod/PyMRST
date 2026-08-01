"""Check the Python deck reader against MRST's own readEclipseDeck.

`tests/deckformat/test_deckinput_matches_bespoke_readers.py` compares the new
reader with PyMRST's hand-written SPE readers. That catches a regression but
cannot catch an error the two share, because both are PyMRST code. This
compares against `ref_deckinput_props.mat`, dumped by MRST itself from
`convertDeckUnits(readEclipseDeck(...))`, which is the actual reference.

Run `dump_deckinput_props_reference.m` in MATLAB first.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "ref_deckinput_props.mat"
DATA = Path("/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/examples/data")

TOLERANCE = 1e-10


def _field(entry, name):
    """One field of a MATLAB struct loaded by scipy, or None."""
    if name not in entry.dtype.names:
        return None
    value = entry[name][0, 0]
    return value


def compare(name: str, entry) -> list:
    """Compare one deck, returning a list of failure descriptions."""
    path = DATA / name / f"BENCH_{name}.DATA"
    props = convert_deck_units(read_eclipse_deck(path))["PROPS"]
    failures = []

    keywords = [str(k[0]) for k in _field(entry, "keywords")[0]]
    print(f"\n{name}: MRST reports {len(keywords)} PROPS keywords")

    for keyword in keywords:
        mine = props.get(keyword)

        # Miscible PVT is stored as three separate arrays in the reference.
        key = _field(entry, f"{keyword}_key")
        if key is not None:
            if mine is None:
                failures.append(f"{name} {keyword}: not read")
                continue
            table = mine[0]
            data = _field(entry, f"{keyword}_data")
            pos = _field(entry, f"{keyword}_pos").ravel().astype(int) - 1
            checks = [
                ("key", table.key, key.ravel()),
                ("pos", table.pos, pos),
                ("data", table.data, data),
            ]
            for what, got, want in checks:
                if got.shape != np.asarray(want).shape:
                    failures.append(
                        f"{name} {keyword}.{what}: shape {got.shape} "
                        f"vs MRST {np.asarray(want).shape}")
                elif not np.allclose(got, want, rtol=TOLERANCE, atol=0,
                                     equal_nan=True):
                    error = np.max(np.abs(got - want))
                    failures.append(
                        f"{name} {keyword}.{what}: max abs diff {error:.3e}")
            print(f"  {keyword:10} key {table.key.size}, "
                  f"data {table.data.shape}  ok")
            continue

        reference = _field(entry, keyword)
        if reference is None:
            continue                       # not a numeric keyword
        if mine is None:
            failures.append(f"{name} {keyword}: not read by the Python reader")
            continue

        got = np.atleast_2d(mine[0] if isinstance(mine, list) else mine)
        want = np.atleast_2d(np.asarray(reference, dtype=float))
        if got.shape != want.shape:
            failures.append(
                f"{name} {keyword}: shape {got.shape} vs MRST {want.shape}")
            continue
        # NaN is a legitimate value here: MRST's record templates default
        # unstated fields to NaN, so ROCK's unused columns are NaN in both.
        if not np.allclose(got, want, rtol=TOLERANCE, atol=0, equal_nan=True):
            error = np.nanmax(np.abs(got - want))
            relative = error / max(np.max(np.abs(want)), 1e-30)
            failures.append(f"{name} {keyword}: max abs diff {error:.3e} "
                            f"(relative {relative:.3e})")
            continue
        print(f"  {keyword:10} {want.shape}  ok")

    # The region partitions, where the deck has them.
    for field in (entry.dtype.names or ()):
        if not field.startswith("REGIONS_"):
            continue
        keyword = field[len("REGIONS_"):]
        want = _field(entry, field).ravel().astype(int)
        deck = read_eclipse_deck(path)
        got = deck["REGIONS"].get(keyword)
        if got is None:
            failures.append(f"{name} REGIONS.{keyword}: not read")
        elif not np.array_equal(np.asarray(got).ravel(), want):
            failures.append(f"{name} REGIONS.{keyword}: differs")
        else:
            print(f"  REGIONS.{keyword:6} {want.size} cells  ok")
    return failures


def main() -> int:
    if not REFERENCE.exists():
        print(f"missing {REFERENCE.name}; "
              f"run dump_deckinput_props_reference.m in MATLAB first")
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
    print("All PROPS values match MRST's readEclipseDeck + convertDeckUnits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
