"""Check `read_equil` against MRST (scripts/matlab/ref_equil_record.mat).

Regenerate from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_equil_record_reference.m')"

`readEQUIL.m` sits in a MATLAB `private/` directory, so it is not callable from
outside its parent. The dump copies it and `readDefaultedKW.m` to a temporary
directory on the path - noted here because a reader trying to reproduce this
will otherwise get "Undefined function".

Both unit systems are compared for every case. The conversion is the part worth
checking: EQUIL alternates depth and pressure across its first six items, so an
off-by-one in the column selection converts a pressure with a length factor and
still produces plausible-looking numbers.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.utils.inout.eclipse.deckinput_simple.private.read_equil import (  # noqa: E402
    read_equil,
)

CASES = {
    "full": "2000 300 2200 0 1800 0 1 1 0 1 0 /",
    "defaulted": "2000 300 9* /",
    "identity": "1234.5 1 9* /",
}


def main() -> int:
    reference = loadmat(
        Path(__file__).parent / "matlab" / "ref_equil_record.mat",
        squeeze_me=True, struct_as_record=False)
    failures = []

    for name, text in CASES.items():
        for flag, suffix in ((True, "true"), (False, "false")):
            mine = read_equil(text.replace("/", " / ").split(), metric=flag)
            theirs = np.atleast_2d(np.asarray(reference[f"{name}_{suffix}"],
                                              dtype=float))
            ok = mine.shape == theirs.shape and np.allclose(mine, theirs,
                                                            rtol=1e-12)
            system = "metric" if flag else "field"
            detail = ""
            if not ok and mine.shape == theirs.shape:
                bad = np.flatnonzero(~np.isclose(mine[0], theirs[0]))
                detail = f"  items {(bad + 1).tolist()} differ"
            print(f"  {'ok ' if ok else 'FAIL'} {name} ({system}){detail}")
            if not ok:
                failures.append(f"{name}/{system}")

    mine = read_equil("2000 300 9* / 2500 350 9* /".replace("/", " / ").split(),
                      ntequil=2)
    theirs = np.asarray(reference["two"], dtype=float)
    ok = mine.shape == theirs.shape and np.allclose(mine, theirs, rtol=1e-12)
    print(f"  {'ok ' if ok else 'FAIL'} two regions {mine.shape}")
    if not ok:
        failures.append("two regions")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("read_equil matches MRST's readEQUIL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
