"""Introduction to the Norne Model.

Port of MRST ``examples/datasets/showNorne.m``.

Norne is a real North Sea field model on a corner-point grid with faults and a
large inactive region. MRST's example loads it, shows the fault faces, contrasts
active against inactive cells, and highlights two logical boxes.

One step is worth understanding before reading the code, because it looks
backwards. MRST saves `ACTNUM`, replaces it with all ones, and only then calls
`processGRDECL`:

    actnum        = grdecl.ACTNUM;
    grdecl.ACTNUM = ones(prod(grdecl.cartDims), 1);
    G             = processGRDECL(grdecl, 'checkgrid', false);

That is deliberate. Processing with the real `ACTNUM` would drop the inactive
cells and leave nothing to compare against; building the full grid keeps them,
so the example can colour active and inactive separately. The saved array is
then indexed through `G.cells.indexMap` to recover which is which. The port
keeps that structure rather than tidying it away, since the alternative reads
as a bug.

Plotting is optional, as elsewhere in this tree. What the port checks is the
structure the pictures are meant to convey: how many cells the model has, how
many are active, that faults were detected, and that a logical box selects the
cells it names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.utils.inout.convert_input_units import convert_input_units
from pymrst.modules.deckformat.deckinput.read_grid import read_grid
from pymrst.utils.units.get_unit_system import get_unit_system

# Norne's published logical dimensions.
NORNE_CART_DIMS = (46, 112, 22)


@dataclass
class NorneModel:
    """The full Norne grid, with activity kept alongside rather than applied."""

    grdecl: dict
    G: Any
    actnum: np.ndarray

    @property
    def active_mask(self) -> np.ndarray:
        """Per-cell activity, aligned to the grid rather than the logical box."""
        return np.asarray(self.actnum, dtype=bool)[
            np.asarray(self.G.cells.indexMap, dtype=np.int64)]

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    def logical_box(self, i_range, j_range, k_range) -> np.ndarray:
        """Grid cells inside a logical box, given as 1-based inclusive ranges.

        MRST writes `meshgrid(6:15, 80:100, 1:22)` and matches rows against
        `ind2sub` output, so the ranges are 1-based and inclusive. They are
        converted here rather than at the call sites.
        """
        dims = np.asarray(self.grdecl["cartDims"], dtype=np.int64)
        index_map = np.asarray(self.G.cells.indexMap, dtype=np.int64)
        i, j, k = np.unravel_index(index_map, tuple(dims), order="F")
        inside = (
            (i >= i_range[0] - 1) & (i <= i_range[1] - 1)
            & (j >= j_range[0] - 1) & (j <= j_range[1] - 1)
            & (k >= k_range[0] - 1) & (k <= k_range[1] - 1)
        )
        return np.flatnonzero(inside)


def find_norne_grdecl(path=None):
    """Locate ``NORNE.GRDECL``, or return ``None``. Never downloads."""
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("norne", download=False)
        if root is not None:
            for candidate in Path(root).rglob("NORNE.GRDECL"):
                return candidate
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "mrst-2024a-Geomech/examples/data/Norne/NORNE.GRDECL"
        if candidate.exists():
            return candidate
    return None


def load_norne(path=None, *, check_grid: bool = False) -> NorneModel | None:
    """Read Norne and build the **full** grid, inactive cells included.

    Returns ``None`` when the dataset is absent.
    """
    grdecl_file = find_norne_grdecl(path)
    if grdecl_file is None:
        return None

    # `read_grdecl` does not follow INCLUDE, and NORNE.GRDECL splits its
    # arrays across an INCLUDE directory, so the operators lose the
    # SPECGRID they need. `read_grid` splices first and parses once.
    grdecl = read_grid(grdecl_file)
    converted = convert_input_units(grdecl, get_unit_system("METRIC"))
    grdecl = converted[0] if isinstance(converted, tuple) else converted

    # Keep the real activity, then process with everything active so the
    # inactive cells survive into the grid and can be compared against.
    actnum = np.asarray(grdecl["ACTNUM"], dtype=np.int64).ravel().copy()
    full = dict(grdecl)
    full["ACTNUM"] = np.ones(int(np.prod(grdecl["cartDims"])), dtype=np.int64)
    g = process_grdecl(full, check_grid=check_grid)
    return NorneModel(grdecl=grdecl, G=g, actnum=actnum)


def fault_faces(model: NorneModel) -> np.ndarray:
    """Faces MRST colours red: those carrying a positive fault tag."""
    tag = getattr(model.G.faces, "tag", None)
    if tag is None:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.asarray(tag).ravel() > 0)


if __name__ == "__main__":
    def test_dataset_is_available():
        assert find_norne_grdecl() is not None, "NORNE.GRDECL not found"
        print("ok test_dataset_is_available")

    def test_logical_dimensions_match_the_published_model():
        model = load_norne()
        assert model is not None, "Norne dataset not found"
        assert tuple(model.grdecl["cartDims"]) == NORNE_CART_DIMS, (
            model.grdecl["cartDims"])
        assert model.actnum.size == int(np.prod(NORNE_CART_DIMS)), model.actnum.size
        print(f"ok test_logical_dimensions_match_the_published_model "
              f"{NORNE_CART_DIMS}")

    def test_grid_keeps_inactive_cells():
        """The point of overwriting ACTNUM before processing.

        Processing with the real ACTNUM would drop inactive cells and leave
        nothing to contrast. Both groups must be present and non-empty, or the
        example's central comparison is vacuous.
        """
        model = load_norne()
        assert model is not None
        active = model.active_mask
        assert active.size == model.G.cells.num, (active.size, model.G.cells.num)
        assert 0 < model.n_active < model.G.cells.num, (
            model.n_active, model.G.cells.num)
        inactive = int(model.G.cells.num - model.n_active)
        assert inactive > 0, "no inactive cells survived processing"
        print(f"ok test_grid_keeps_inactive_cells "
              f"({model.n_active} active, {inactive} inactive)")

    def test_faults_are_detected():
        """Norne is faulted; a model with no tagged faces has been mis-read."""
        model = load_norne()
        assert model is not None
        faces = fault_faces(model)
        assert faces.size > 0, "no fault faces found in a faulted model"
        assert faces.max() < model.G.faces.num
        print(f"ok test_faults_are_detected ({faces.size} fault faces)")

    def test_logical_box_selects_the_cells_it_names():
        """MRST's ranges are 1-based and inclusive; the conversion must hold.

        Every returned cell must lie inside the requested box, checked by
        recomputing its logical position rather than by trusting the selector.
        """
        model = load_norne()
        assert model is not None
        cells = model.logical_box((6, 15), (80, 100), (1, 22))
        assert cells.size > 0, "the box selected nothing"
        dims = tuple(np.asarray(model.grdecl["cartDims"], dtype=np.int64))
        index_map = np.asarray(model.G.cells.indexMap, dtype=np.int64)[cells]
        i, j, k = np.unravel_index(index_map, dims, order="F")
        assert np.all((i >= 5) & (i <= 14)), (i.min(), i.max())
        assert np.all((j >= 79) & (j <= 99)), (j.min(), j.max())
        assert np.all((k >= 0) & (k <= 21)), (k.min(), k.max())
        print(f"ok test_logical_box_selects_the_cells_it_names ({cells.size} cells)")

    def test_a_second_box_is_smaller_than_the_first():
        model = load_norne()
        assert model is not None
        big = model.logical_box((6, 15), (80, 100), (1, 22))
        small = model.logical_box((9, 11), (65, 67), (1, 22))
        assert 0 < small.size < big.size, (small.size, big.size)
        print("ok test_a_second_box_is_smaller_than_the_first")

    def test_degenerate_cells_are_confined_to_the_inactive_region():
        """Every *active* cell has positive volume; some inactive ones do not.

        This is why MRST passes `'checkgrid', false` here. Forcing all cells
        active to keep the inactive region visible also admits the degenerate
        geometry that region contains: 64 of Norne's cells have non-positive
        volume, and every one of them is inactive. Asserting that all volumes
        are positive would be wrong for this deliberately-full grid, while
        asserting nothing would miss a real defect if a degenerate cell ever
        appeared among the active ones.
        """
        from pymrst.gridprocessing import compute_geometry
        model = load_norne()
        assert model is not None
        g = compute_geometry(model.G)
        vol = np.asarray(g.cells.volumes, dtype=float)
        active = model.active_mask
        assert np.all(vol[active] > 0.0), vol[active].min()
        degenerate = vol <= 0.0
        assert not np.any(degenerate & active), (
            f"{int(np.count_nonzero(degenerate & active))} active cells are degenerate")
        print(f"ok test_degenerate_cells_are_confined_to_the_inactive_region "
              f"({int(degenerate.sum())} degenerate, all inactive; "
              f"active bulk {vol[active].sum():.4e} m3)")

    test_dataset_is_available()
    test_logical_dimensions_match_the_published_model()
    test_grid_keeps_inactive_cells()
    test_faults_are_detected()
    test_logical_box_selects_the_cells_it_names()
    test_a_second_box_is_smaller_than_the_first()
    test_degenerate_cells_are_confined_to_the_inactive_region()
    print("\nAll tests passed!")
