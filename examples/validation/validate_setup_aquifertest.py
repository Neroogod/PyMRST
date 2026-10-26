"""Validate MRST's setupAquifertest.

In MRST 2024a the function stops at `AquiferBlackOilModel`, which no longer
exists. The reference (`dump_setup_aquifertest_reference.m`) records that
error and, for both published decks, what the function computes before it:
the grid's cell count, the compressed rock, and `processAquifer`'s
connection table, initial aquifer pressures and volumes, and
compressibilities. The port must stop with the same error, and its steps
before it must give the same values, to `RTOL`. MATLAB's 1-based aquifer
and cell numbers in the connection table are compared as such.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_blackoil.examples.utils.setup_aquifertest import (
    CASES, aquifertest_data_directory, setup_aquifertest,
)
from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from pymrst.modules.deckformat.grid import init_eclipse_grid
from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
from pymrst.modules.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
from pymrst.modules.deckformat.params.rock.process_aquifer import process_aquifer

HERE = Path(__file__).resolve().parent
RTOL = 1e-14


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def check_case(case: int, ref) -> bool:
    fn = aquifertest_data_directory()/f"{CASES[case - 1]}.DATA"
    try:
        setup_aquifertest(fn)
        message = ""
    except NameError as undefined:
        message = str(undefined)
    ok = report(str(ref.err) in message, f"case {case}: stops with MRST's error '{ref.err}'")
    deck = convert_deck_units(read_eclipse_deck(fn))
    G = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), G.cells.indexMap)
    output = process_aquifer(deck, G)
    ok &= report(G.cells.num == int(ref.ncells), f"case {case}: {G.cells.num} cells")
    for name, actual, expected in (("perm", rock.perm, ref.perm), ("poro", rock.poro, ref.poro)):
        err = _rel(np.reshape(actual, np.shape(expected)), expected)
        ok &= report(err < RTOL, f"case {case}: rock {name} rel. diff {err:.1e}")
    table = np.atleast_2d(np.asarray(output.aquifers, dtype=float)).copy()
    index = output.aquind
    table[:, index.aquid] += 1
    table[:, index.conn] += 1
    err = _rel(table, ref.aquifers)
    ok &= report(err < RTOL, f"case {case}: processAquifer connection table rel. diff {err:.1e}")
    for name, actual, expected in (("pressures", output.initval["pressures"], ref.pressures),
                                   ("volumes", output.initval["volumes"], ref.volumes),
                                   ("C", output.aquiferprops.C, ref.C)):
        err = _rel(np.ravel(actual), np.ravel(expected))
        ok &= report(err < RTOL, f"case {case}: aquifer {name} rel. diff {err:.1e}")
    return ok


def main() -> int:
    ref = loadmat(HERE/"ref_setup_aquifertest.mat", squeeze_me=True, struct_as_record=False)
    ok = check_case(1, ref["case1"]) & check_case(2, ref["case2"])
    print("\nsetupAquifertest matches MRST" if ok else "\nsetupAquifertest does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
