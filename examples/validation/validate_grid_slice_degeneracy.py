"""Validate sliceGrid degeneracies against an MRST reference dump.

Covers the two cases where the cut plane coincides with a face: a pinched cell
whose collapsed face lies in the plane (MRST still splits it), and a cut
snapped onto an existing column of faces (MRST splits nothing but reports the
coinciding face as the slice). Cell and face renumbering differ from MATLAB, so
the checks use counts, total and sorted volumes, and status histograms.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

REF = Path(__file__).with_name("ref_grid_slice_degeneracy.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _status_counts(status, size: int = 4) -> np.ndarray:
    return np.bincount(np.asarray(status, dtype=np.int64), minlength=size)[:size]


def _pinched_cell(ref) -> object:
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    grid.nodes.coords[:] = ref["pinch_coords"]
    with warnings.catch_warnings():                      # the collapsed face
        warnings.simplefilter("ignore")
        return compute_geometry(grid)


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_grid_slice_degeneracy_reference.m first")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    print("sliceGrid, plane containing a pinched cell's collapsed face:")
    grid = _pinched_cell(ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut, gix, cross = slice_grid(
            grid, np.array([0.5, 0.5, 0.5]), normal=np.array([1.0, 0.0, 0.0]),
            return_mapping=True, return_slice=True)
        cut = compute_geometry(cut)
    failed += _check(f"cell count ({cut.cells.num})",
                     cut.cells.num == int(ref["pinch_cells_num"]))
    failed += _check(f"total volume ({cut.cells.volumes.sum():.6f})",
                     np.isclose(cut.cells.volumes.sum(),
                                float(ref["pinch_total_volume"])))
    failed += _check("volume is conserved by the cut",
                     np.isclose(cut.cells.volumes.sum(),
                                float(ref["pinch_volume_in"])))
    sorted_volumes = np.sort(np.asarray(cut.cells.volumes))
    error = np.max(np.abs(sorted_volumes - np.atleast_1d(ref["pinch_volumes_sorted"])))
    failed += _check("sorted per-cell volumes match", error < 1e-9,
                     f"max|dV| = {error:.3e}")
    failed += _check("no zero-volume cell", bool(np.all(cut.cells.volumes > 1e-12)))
    # MRST keeps a zero-area sliver where the collapsed face is split; we drop
    # it, so face counts differ by exactly that. Every face with area compares.
    areas = np.sort(np.asarray(cut.faces.areas))
    real, mrst_real = areas[areas > 1e-12], np.atleast_1d(ref["pinch_face_areas_nz"])
    failed += _check(f"non-degenerate face count ({real.size})",
                     real.size == mrst_real.size)
    error = np.max(np.abs(real - mrst_real)) if real.size == mrst_real.size else np.inf
    failed += _check("non-degenerate face areas match", error < 1e-9,
                     f"max|dA| = {error:.3e}")
    failed += _check(
        f"no more zero-area faces than MRST ({int((areas <= 1e-12).sum())} "
        f"vs {int(ref['pinch_degenerate_faces'])})",
        int((areas <= 1e-12).sum()) <= int(ref["pinch_degenerate_faces"]),
    )
    failed += _check(
        "slice face count",
        _status_counts(gix.new.faces)[3]
        == _status_counts(ref["pinch_face_status"])[3],
    )
    failed += _check(
        "cell status histogram",
        np.array_equal(_status_counts(gix.new.cells),
                       _status_counts(ref["pinch_cell_status"])),
    )
    failed += _check(
        "slice grid cell count",
        (0 if cross is None else cross.cells.num)
        == int(ref["pinch_slice_num_cells"]),
    )

    print("sliceGrid, cut snapped onto an existing column of faces:")
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    cut, gix, cross = slice_grid(
        grid, np.array([[1.0, -1.0, 0.0], [1.0, 2.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]), return_mapping=True,
        return_slice=True)
    cut = compute_geometry(cut)
    failed += _check(f"cell count ({cut.cells.num})",
                     cut.cells.num == int(ref["snap_cells_num"]))
    failed += _check("nothing is split, volume unchanged",
                     np.isclose(cut.cells.volumes.sum(),
                                float(ref["snap_total_volume"])))
    failed += _check(
        "face status histogram",
        np.array_equal(_status_counts(gix.new.faces),
                       _status_counts(ref["snap_face_status"])),
        f"{_status_counts(gix.new.faces)[1:]} vs "
        f"{_status_counts(ref['snap_face_status'])[1:]}",
    )
    failed += _check(
        "cell status histogram",
        np.array_equal(_status_counts(gix.new.cells),
                       _status_counts(ref["snap_cell_status"])),
    )
    failed += _check(
        "slice grid cell count",
        (0 if cross is None else cross.cells.num)
        == int(ref["snap_slice_num_cells"]),
    )
    expected = np.asarray(ref["snap_slice_coords"], dtype=float).reshape(-1, 3)
    actual = np.asarray(cross.nodes.coords, dtype=float) if cross is not None else None
    failed += _check(
        "slice grid node coordinates",
        actual is not None and actual.shape == expected.shape
        and np.allclose(_sorted_rows(actual), _sorted_rows(expected)),
    )
    return int(failed > 0)


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(tuple(points.T[::-1]))]


if __name__ == "__main__":
    raise SystemExit(main())
