"""Validate the coarse-grid coarsening layer against MRST.

Reference: ``dump_coarsening_reference.m`` (MATLAB R2021b, MRST 2024a).

Covers `coarsenGeometry`, `fineToCoarseSign`, `coarsenFlux`, `invertPartition`
and `refineNearWell` on four cases:

* **A** a uniform Cartesian grid in 3x2x1 blocks;
* **B** a graded tensor grid, where the volume weighting in a block centroid
  actually changes the answer;
* **C** an L-shaped block, where a coarse face folds and its area is smaller
  than the summed area of its fine faces;
* **D** `refineNearWell`, logarithmic and linear rings and a masked far field.

Geometry is compared to a relative tolerance, since both sides accumulate the
same sums in a different order. Everything else is exact.

`refineNearWell` returns block *labels*, and MRST's are 1-based with 0 marking
the masked far field while this port's are 0-based with -1. The grouping is
compared rather than the labels, and the mask is compared separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.refine_near_well import refine_near_well
from pymrst.modules.coarsegrid.utils.coarsen_flux import coarsen_flux
from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign
from pymrst.modules.coarsegrid.utils.invert_partition import invert_partition

REFERENCE = Path(__file__).with_name("ref_coarsening.mat")
RTOL = 1e-12


def load_reference():
    if not REFERENCE.exists():
        raise SystemExit(
            f"{REFERENCE.name} is missing. Regenerate it with\n"
            f"  matlab -batch \"startup; run('"
            f"{REFERENCE.with_name('dump_coarsening_reference.m')}')\"")
    return sio.loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)


def compare(name, field, mine, theirs, exact=False) -> list[str]:
    mine = np.asarray(mine)
    theirs = np.asarray(theirs).reshape(mine.shape)
    if exact:
        ok = np.array_equal(mine, theirs)
    else:
        ok = np.allclose(mine, theirs, rtol=RTOL, atol=1e-10)
    if ok:
        return []
    worst = float(np.max(np.abs(mine - theirs))) if not exact else float("nan")
    return [f"{name}: {field} differs (worst {worst:.3e})"]


def check_geometry(name, cg, ref) -> list[str]:
    out: list[str] = []
    out += compare(name, "cells.num", [cg.n_blocks], [int(ref.cells_num)], exact=True)
    out += compare(name, "cells.volumes", cg.cell_volumes, ref.cells_volumes)
    out += compare(name, "cells.centroids", cg.cell_centroids, ref.cells_centroids)
    out += compare(name, "faces.num", [cg.num_faces], [int(ref.faces_num)], exact=True)
    out += compare(name, "faces.areas", cg.face_areas, ref.faces_areas)
    out += compare(name, "faces.centroids", cg.face_centroids, ref.faces_centroids)
    out += compare(name, "faces.normals", cg.face_normals, ref.faces_normals)
    out += compare(name, "faces.neighbors", cg.neighbors + 1, ref.faces_neighbors,
                   exact=True)
    return out


def same_grouping(mine, theirs) -> bool:
    mine, theirs = np.asarray(mine).ravel(), np.asarray(theirs).ravel()
    if mine.size != theirs.size:
        return False
    joint = np.unique(np.column_stack([mine, theirs]), axis=0).shape[0]
    return joint == np.unique(mine).size == np.unique(theirs).size


def main() -> int:
    d = load_reference()
    failures: list[str] = []

    # ---------------------------------------------------------------- case A
    ref = d["A"]
    g = compute_geometry(cart_grid([6, 4, 2], [6.0, 4.0, 2.0]))
    p = partition_ui(g, [3, 2, 1])
    assert np.array_equal(p + 1, np.asarray(ref.p)), "partition A differs"
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    failures += check_geometry("A", cg, ref)
    failures += compare("A", "fineToCoarseSign", fine_to_coarse_sign(cg), ref.sgn,
                        exact=True)
    failures += compare("A", "coarsenFlux", coarsen_flux(cg, np.asarray(ref.flux)),
                        ref.cflux)
    pos, b2c, locno = invert_partition(p)
    failures += compare("A", "invertPartition pos", pos + 1, ref.b2c_pos, exact=True)
    failures += compare("A", "invertPartition b2c", b2c + 1, ref.b2c, exact=True)
    failures += compare("A", "invertPartition locno", locno + 1, ref.locno,
                        exact=True)
    print(f"A  uniform Cartesian:  {cg.n_blocks} blocks, {cg.num_faces} faces, "
          f"volume {cg.cell_volumes.sum():.6f}")

    # ---------------------------------------------------------------- case B
    ref = d["B"]
    g = compute_geometry(tensor_grid(np.array([0.0, 1.0, 3.0, 7.0, 15.0]),
                                     np.array([0.0, 1.0, 3.0, 7.0]),
                                     np.array([0.0, 1.0, 4.0])))
    p = partition_ui(g, [2, 2, 1])
    assert np.array_equal(p + 1, np.asarray(ref.p)), "partition B differs"
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    failures += check_geometry("B", cg, ref)
    failures += compare("B", "fineToCoarseSign", fine_to_coarse_sign(cg), ref.sgn,
                        exact=True)
    # The case only means something if the weighting changes the answer.
    plain = np.array([np.asarray(g.cells.centroids)[p == b].mean(axis=0)
                      for b in range(cg.n_blocks)])
    if np.allclose(plain, cg.cell_centroids):
        failures.append("B: the graded case does not distinguish the weightings")
    print(f"B  graded tensor grid: {cg.n_blocks} blocks, {cg.num_faces} faces")

    # ---------------------------------------------------------------- case C
    ref = d["C"]
    g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    p = np.array([0, 1, 1, 1], dtype=np.int64)
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    failures += check_geometry("C", cg, ref)
    counts = np.diff(cg.conn_pos)
    face_no = np.repeat(np.arange(cg.num_faces), counts)
    summed = np.bincount(face_no, weights=np.asarray(g.faces.areas)[cg.fconn],
                         minlength=cg.num_faces)
    folded = int(np.count_nonzero(~np.isclose(cg.face_areas, summed)))
    if folded == 0:
        failures.append("C: no coarse face folds, so the area rule is untested")
    print(f"C  L-shaped block:     {cg.n_blocks} blocks, {cg.num_faces} faces, "
          f"{folded} folded")

    # ---------------------------------------------------------------- case D
    ref = d["D"]
    g = compute_geometry(cart_grid([20, 20, 1], [20.0, 20.0, 1.0]))
    pts = np.asarray(g.cells.centroids, dtype=float)
    well = np.array([10.0, 10.0, 0.5])
    for label, kwargs, key, ring_key in (
            ("log rings", dict(log_bins=True), "p", "binRadius"),
            ("linear rings", dict(log_bins=False), "p_linear", "binRadius_linear")):
        pw, _, radius, dist = refine_near_well(
            pts, well, angle_bins=4, radius_bins=3, **kwargs)
        if not same_grouping(pw, np.asarray(getattr(ref, key))):
            failures.append(f"D: refineNearWell {label} grouping differs")
        if not np.array_equal(radius + 1, np.asarray(getattr(ref, ring_key))):
            failures.append(f"D: refineNearWell {label} ring index differs")
    failures += compare("D", "distance", dist, ref.dist)

    masked, _, _, _ = refine_near_well(pts, well, max_radius=5.0, angle_bins=4,
                                       radius_bins=3)
    theirs = np.asarray(ref.p_masked, dtype=np.int64)
    if not np.array_equal(masked < 0, theirs == 0):
        failures.append("D: the masked far field differs")
    inside = masked >= 0
    if not same_grouping(masked[inside], theirs[inside]):
        failures.append("D: refineNearWell masked grouping differs")
    print(f"D  refineNearWell:     {int(np.max(masked)) + 1} blocks, "
          f"{int(np.count_nonzero(~inside))} of {masked.size} masked")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("All coarsening checks match MRST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
