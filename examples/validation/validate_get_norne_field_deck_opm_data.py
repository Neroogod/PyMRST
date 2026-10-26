"""Validate MRST's getNorneFieldDeckOPMData.

The reference is `dump_get_norne_field_deck_opm_data_reference.m` (MRST
2024a), a fingerprint of what the function returns:
- the deck: grid dimensions, PERMX/PORO/ACTNUM sums, the schedule's 247
  controls and step lengths;
- `output.opm` (flow_legacy's summary and restart, unsplit grid): report
  steps, well names, the first five steps' bhp and oil rates, the times, the
  number of restart states and the first one's pressure and saturations;
- `output.eclipse` (ECL.2014.2's summary): its 884 summary steps and first
  bhps, no states (the restart file is not in the dataset, which MRST
  reports and carries on), and the grid split into its two disconnected
  parts.
Values to `RTOL`, counts and names exactly.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.utils.get_norne_field_deck_opm_data import (
    get_norne_field_deck_opm_data,
)

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def _grids(G):
    return list(G) if isinstance(G, (list, tuple)) else [G]


def main() -> int:
    ref = loadmat(HERE/"ref_get_norne_field_deck_opm_data.mat", squeeze_me=True)
    deck, output = get_norne_field_deck_opm_data()
    grid, schedule = deck["GRID"], deck["SCHEDULE"]
    ok = report(np.array_equal(np.ravel(grid["cartDims"]), ref["cartDims"]),
                f"deck cartDims {np.ravel(grid['cartDims']).tolist()}")
    for key, rkey in (("PERMX", "permx_sum"), ("PORO", "poro_sum"), ("ACTNUM", "actnum_sum")):
        err = _rel(np.sum(grid[key]), ref[rkey])
        ok &= report(err < RTOL, f"deck sum({key}) rel. diff {err:.1e}")
    ok &= report(len(schedule.control) == int(ref["nctrl"]),
                 f"deck schedule: {len(schedule.control)} controls")
    step = schedule.step
    err = _rel(step["val"] if isinstance(step, dict) else step.val, ref["step_val"])
    ok &= report(err < RTOL, f"deck schedule step lengths rel. diff {err:.1e}")

    opm = output.opm
    ok &= report(len(opm.wellSols) == int(ref["opm_nsteps"]),
                 f"opm: {len(opm.wellSols)} summary steps")
    names = [w.name for w in opm.wellSols[0]]
    ok &= report(names == [str(n) for n in ref["opm_names"]], f"opm: {len(names)} well names")
    for key in ("bhp", "qOs"):
        actual = [[getattr(w, key) for w in row] for row in opm.wellSols[:5]]
        err = _rel(actual, ref[f"opm_{key}"])
        ok &= report(err < RTOL, f"opm: first five steps' {key} rel. diff {err:.1e}")
    err = _rel(opm.time, ref["opm_time"])
    ok &= report(err < RTOL, f"opm: times rel. diff {err:.1e}")
    ok &= report(len(opm.states) == int(ref["opm_nstates"]), f"opm: {len(opm.states)} restart states")
    err = max(_rel(opm.states[0]["pressure"], ref["opm_p1"]), _rel(opm.states[0]["s"], ref["opm_s1"]))
    ok &= report(err < RTOL, f"opm: first restart state rel. diff {err:.1e}")
    ok &= report([g.cells.num for g in _grids(opm.G)] == [int(ref["opm_ncells"])],
                 f"opm: unsplit grid, {int(ref['opm_ncells'])} cells")
    ok &= report(Path(opm.location).parent.name == str(ref["opm_location"]),
                 f"opm: read from {Path(opm.location).parent.name}")

    ecl = output.eclipse
    ok &= report(len(ecl.wellSols) == int(ref["ecl_nsteps"]),
                 f"eclipse: {len(ecl.wellSols)} summary steps")
    actual = [[w.bhp for w in row] for row in ecl.wellSols[:5]]
    err = _rel(actual, ref["ecl_bhp"])
    ok &= report(err < RTOL, f"eclipse: first five steps' bhp rel. diff {err:.1e}")
    nstates = 0 if ecl.states is None else len(ecl.states)
    ok &= report(nstates == int(ref["ecl_nstates"]), f"eclipse: {nstates} restart states")
    cells = [g.cells.num for g in _grids(ecl.G)]
    ok &= report(cells == np.atleast_1d(ref["ecl_ncells"]).astype(int).tolist(),
                 f"eclipse: grid split into {cells} cells")
    print("\ngetNorneFieldDeckOPMData matches MRST" if ok
          else "\ngetNorneFieldDeckOPMData does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
