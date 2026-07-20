"""Phase 14: Pollock streamline tracing against MRST's pollock.m.

Traces from MRST's own converged flux, so what is compared is the tracer
alone. Checks the forward and the reverse field, the point coordinates,
and the time-of-flight accumulated along each line.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.gridprocessing import cart_grid, compute_geometry   # noqa: E402
from pymrst.modules.streamlines import pollock                  # noqa: E402
from pymrst.utils.units import DAY                              # noqa: E402

REF = Path(__file__).with_name("ref_p14_streamlines.mat")
TOL = 1e-8


def unpack(coord, tof, cell, lengths):
    """Split MATLAB's concatenated streamlines back into a list."""
    out, at = [], 0
    for n in lengths.astype(int):
        out.append((coord[at:at + n], tof[at:at + n], cell[at:at + n]))
        at += n
    return out


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase14_reference.m in MATLAB)")
        return 0

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    n = int(ref["n"])
    g = compute_geometry(cart_grid([n, n], [100.0, 100.0]))

    flux = np.asarray(ref["flux"], float).ravel()
    pv = np.asarray(ref["pv"], float).ravel()
    startpos = np.atleast_2d(ref["startpos"]).astype(float)
    prod = int(ref["prod"]) - 1                       # MATLAB is 1-based

    # cell indices to 0-based; local coordinates are unchanged
    pos = startpos.copy()
    pos[:, 0] -= 1.0

    outflow = np.zeros(g.cells.num, dtype=bool)
    outflow[prod] = True

    results = []

    # ---- forward -----------------------------------------------------------
    got = pollock(g, flux, positions=pos, pvol=pv, isoutflow=outflow,
                  substeps=5)
    want = unpack(np.atleast_2d(ref["coord"]), np.asarray(ref["tof"], float),
                  np.asarray(ref["cell"], float),
                  np.asarray(ref["len"], float))
    results.append(("forward", got, want))

    # ---- reverse -----------------------------------------------------------
    gotr = pollock(g, flux, positions=pos, pvol=pv, reverse=True, substeps=5)
    wantr = unpack(np.atleast_2d(ref["coordr"]), np.asarray(ref["tofr"], float),
                   np.asarray(ref["cellr"], float),
                   np.asarray(ref["lenr"], float))
    results.append(("reverse", gotr, wantr))

    print(f"Pollock streamlines, {len(got)} lines from MRST's flux "
          f"({g.cells.num} cells)\n")

    ok = True
    for name, mine, theirs in results:
        if len(mine) != len(theirs):
            print(f"  FAIL {name}: {len(mine)} lines vs {len(theirs)}")
            ok = False
            continue

        bad_len = [k for k, (a, b) in enumerate(zip(mine, theirs))
                   if a.coords.shape[0] != b[0].shape[0]]
        if bad_len:
            print(f"  FAIL {name} point counts differ on lines {bad_len[:5]}")
            for k in bad_len[:3]:
                print(f"        line {k}: {mine[k].coords.shape[0]} points "
                      f"vs {theirs[k][0].shape[0]}")
            ok = False
            continue

        dx = max(np.abs(a.coords - b[0]).max() for a, b in zip(mine, theirs))

        # A back-traced line ends in the injector, which it cannot leave, so
        # the last step has an infinite exit time in both codes. Require the
        # infinities to fall in the same places and compare the finite part.
        same_inf = all(np.array_equal(np.isfinite(a.tof), np.isfinite(b[1]))
                       for a, b in zip(mine, theirs))
        dt, tmax = 0.0, 0.0
        for a, b in zip(mine, theirs):
            fin = np.isfinite(a.tof)
            ta, tb = np.cumsum(a.tof[fin]), np.cumsum(b[1][np.isfinite(b[1])])
            dt = max(dt, np.abs(ta - tb).max())
            tmax = max(tmax, tb[-1])
        rel_t = dt / tmax

        for label, val, tol in (
            (f"{name} coordinates (m)", dx, 1e-7),
            (f"{name} time-of-flight (rel)", rel_t, TOL),
        ):
            status = "ok  " if val <= tol else "FAIL"
            ok &= val <= tol
            print(f"  {status} {label:34s} max diff = {val:.2e}  (tol {tol:.0e})")
        status = "ok  " if same_inf else "FAIL"
        ok &= same_inf
        print(f"  {status} {name + ' terminating steps':34s} "
              f"{'infinities coincide' if same_inf else 'MISPLACED'}")

    # ---- cell labelling ----------------------------------------------------
    # The one place the two implementations disagree on purpose. pollock.m
    # advances pos(:,1) to the next cell on line 179 and then stores *that*
    # into C on line 187, so every sub-step point is tagged with the cell the
    # streamline is about to enter rather than the one its segment lies in.
    # We tag the segment's own cell. Two checks below: (a) MRST's labels are
    # exactly ours shifted by one cell, and (b) ours are the geometrically
    # correct ones - the point a label carries really does lie in that cell.
    h = 100.0 / n
    shift_ok, ours_in, theirs_in, total = True, 0, 0, 0
    for a, b in zip(got, want):
        mrst_cells = b[2].astype(int) - 1
        predicted = np.concatenate(
            [a.cells[6:], np.full(5, a.cells[-1])])[:len(mrst_cells) - 1]
        shift_ok &= (mrst_cells[0] == a.cells[0]
                     and np.array_equal(mrst_cells[1:], predicted))

        def contains(labels):
            c = g.cells.centroids[labels]
            return int((np.abs(a.coords - c) <= h / 2 + 1e-9).all(axis=1).sum())

        ours_in += contains(a.cells)
        theirs_in += contains(mrst_cells)
        total += len(mrst_cells)

    status = "ok  " if shift_ok else "FAIL"
    ok &= shift_ok
    print(f"\n  {status} cell labels: MRST's are ours shifted one cell "
          f"({'confirmed on all lines' if shift_ok else 'NOT a clean shift'})")
    print(f"       points lying inside the cell they are labelled with: "
          f"ours {ours_in}/{total}, MRST's {theirs_in}/{total}")
    if ours_in == total and theirs_in < total:
        print("       -> pollock.m's C output is off by one cell; ours is "
              "the one that matches its own documentation")

    print("\nPollock streamlines match MATLAB!" if ok
          else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
