"""Visualizing the Johansen Data Set.

Port of MRST ``examples/datasets/showJohansen.m``.

Johansen is a candidate CO2 storage aquifer offshore Norway. The example uses
the NPD5 sector, a faulted corner-point model, and builds two grids from it for
different purposes:

1. an **all-active** grid, made by overwriting ``ACTNUM`` with ones before
   processing, so the inactive region survives and can be contrasted with the
   active one, exactly as ``showNorne`` does;
2. the **real** grid, processed with the true ``ACTNUM``, which is the one that
   carries geometry and gets the porosity field.

Porosity arrives separately, as a plain text file over the full logical box,
and is sampled onto the grid through ``G.cells.indexMap``. That step is the one
worth checking: the file has one value per logical cell and the grid holds only
the active ones, so the mapping is the whole content of the operation.

Plotting is optional, as elsewhere in this tree. The port keeps the two-grid
structure rather than simplifying it, since the contrast is the point of the
first half of MRST's script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.modules.deckformat.deckinput.read_grid import read_grid

# NPD5's published logical dimensions.
NPD5_CART_DIMS = (100, 100, 11)


@dataclass
class JohansenSector:
    """Both grids MRST builds, plus the sampled porosity."""

    grdecl: dict
    G_full: Any        # every logical cell, active or not
    G: Any             # the real grid, active cells only
    actnum: np.ndarray
    poro: np.ndarray | None

    @property
    def active_mask_full(self) -> np.ndarray:
        """Activity aligned to the all-active grid."""
        return np.asarray(self.actnum, dtype=bool)[
            np.asarray(self.G_full.cells.indexMap, dtype=np.int64)]


def find_johansen_sector(sector: str = "NPD5", path=None):
    """Locate a Johansen sector's ``.grdecl``, or return ``None``.

    Never downloads, so this cannot prompt or block an automated run.
    """
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("johansen", download=False)
        if root is not None:
            candidate = Path(root) / f"{sector}.grdecl"
            if candidate.exists():
                return candidate
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = (parent.parent / "mrst-2024a-Geomech/examples/data/Johansen"
                     / f"{sector}.grdecl")
        if candidate.exists():
            return candidate
    return None


def load_johansen(sector: str = "NPD5", path=None) -> JohansenSector | None:
    """Read a sector and build both grids, sampling porosity onto the real one."""
    grdecl_file = find_johansen_sector(sector, path)
    if grdecl_file is None:
        return None

    grdecl = read_grid(grdecl_file)
    actnum = np.asarray(grdecl["ACTNUM"], dtype=np.int64).ravel().copy()
    n_logical = int(np.prod(grdecl["cartDims"]))

    # First grid: everything active, so the inactive region stays visible.
    full = dict(grdecl)
    full["ACTNUM"] = np.ones(n_logical, dtype=np.int64)
    g_full = process_grdecl(full, check_grid=False)

    # Second grid: the real one.
    g = compute_geometry(process_grdecl(dict(grdecl)))

    poro = _load_porosity(grdecl_file, g, n_logical)
    return JohansenSector(grdecl=grdecl, G_full=g_full, G=g, actnum=actnum,
                          poro=poro)


def _load_porosity(grdecl_file: Path, g, n_logical: int):
    """Sample ``<sector>_Porosity.txt`` onto the active grid.

    The file holds one value per logical cell; the grid holds only the active
    ones, so `indexMap` selects. Returns ``None`` when the file is absent,
    since the grid is still usable without it.
    """
    poro_file = grdecl_file.with_name(grdecl_file.stem + "_Porosity.txt")
    if not poro_file.exists():
        return None
    # The file opens with a `% Porosity` header, MATLAB's comment marker rather
    # than the `#` NumPy assumes.
    values = np.loadtxt(poro_file, comments="%").ravel()
    if values.size < n_logical:
        raise ValueError(
            f"{poro_file.name} holds {values.size} values for {n_logical} "
            f"logical cells")
    return values[:n_logical][np.asarray(g.cells.indexMap, dtype=np.int64)]


def fault_faces(g) -> np.ndarray:
    """Faces MRST colours red: those carrying a positive fault tag."""
    tag = getattr(g.faces, "tag", None)
    if tag is None:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.asarray(tag).ravel() > 0)


if __name__ == "__main__":
    def test_dataset_is_available():
        assert find_johansen_sector() is not None, "NPD5.grdecl not found"
        print("ok test_dataset_is_available")

    def test_logical_dimensions_match_the_npd5_sector():
        case = load_johansen()
        assert case is not None, "Johansen dataset not found"
        assert tuple(case.grdecl["cartDims"]) == NPD5_CART_DIMS, (
            case.grdecl["cartDims"])
        assert case.actnum.size == int(np.prod(NPD5_CART_DIMS))
        print(f"ok test_logical_dimensions_match_the_npd5_sector {NPD5_CART_DIMS}")

    def test_two_grids_differ_in_exactly_the_inactive_cells():
        """The all-active grid must be larger by the inactive count.

        That relationship is what makes the first half of MRST's script a
        comparison rather than two pictures of the same thing, and it fails
        immediately if ACTNUM is mis-read.
        """
        case = load_johansen()
        assert case is not None
        n_active = int(np.count_nonzero(case.actnum))
        assert case.G.cells.num == n_active, (case.G.cells.num, n_active)
        assert case.G_full.cells.num > case.G.cells.num
        assert int(np.count_nonzero(case.active_mask_full)) == n_active
        print(f"ok test_two_grids_differ_in_exactly_the_inactive_cells "
              f"({n_active} active, {case.G_full.cells.num - n_active} inactive)")

    def test_faults_are_detected():
        """Johansen is faulted; no tagged faces means the model was mis-read."""
        case = load_johansen()
        assert case is not None
        faces = fault_faces(case.G_full)
        assert faces.size > 0, "no fault faces in a faulted model"
        print(f"ok test_faults_are_detected ({faces.size} fault faces)")

    def test_porosity_is_sampled_onto_the_active_cells():
        """One value per grid cell, drawn through `indexMap` from the full box.

        Taking the first `n_active` values instead, or forgetting the map,
        gives an array of the right length and the wrong contents, which no
        colour plot would reveal.
        """
        case = load_johansen()
        assert case is not None
        assert case.poro is not None, "NPD5_Porosity.txt not found"
        assert case.poro.shape == (case.G.cells.num,), case.poro.shape
        assert np.all((case.poro >= 0.0) & (case.poro <= 1.0)), (
            case.poro.min(), case.poro.max())
        # MRST's own colour-bar range for this field is [0.09, 0.31], so the
        # sampled values should sit inside it.
        assert 0.05 <= case.poro.mean() <= 0.35, case.poro.mean()
        print(f"ok test_porosity_is_sampled_onto_the_active_cells "
              f"(mean {case.poro.mean():.4f}, range "
              f"{case.poro.min():.3f} to {case.poro.max():.3f})")

    def test_porosity_matches_a_direct_lookup():
        """Recompute the mapping independently and require agreement."""
        case = load_johansen()
        assert case is not None and case.poro is not None
        grdecl_file = find_johansen_sector()
        values = np.loadtxt(grdecl_file.with_name("NPD5_Porosity.txt"),
                            comments="%").ravel()
        index_map = np.asarray(case.G.cells.indexMap, dtype=np.int64)
        assert np.allclose(case.poro, values[index_map]), "indexMap sampling differs"
        print("ok test_porosity_matches_a_direct_lookup")

    def test_active_geometry_is_physical():
        case = load_johansen()
        assert case is not None
        vol = np.asarray(case.G.cells.volumes, dtype=float)
        assert np.all(vol > 0.0), vol.min()
        z = np.asarray(case.G.cells.centroids, dtype=float)[:, 2]
        assert z.min() > 0.0, "Johansen lies below the surface"
        print(f"ok test_active_geometry_is_physical "
              f"(bulk {vol.sum():.4e} m3, depth {z.min():.0f} to {z.max():.0f} m)")

    test_dataset_is_available()
    test_logical_dimensions_match_the_npd5_sector()
    test_two_grids_differ_in_exactly_the_inactive_cells()
    test_faults_are_detected()
    test_porosity_is_sampled_onto_the_active_cells()
    test_porosity_matches_a_direct_lookup()
    test_active_geometry_is_physical()
    print("\nAll tests passed!")
