"""Stair-stepped and pillar grids from Case B4.

Port of MRST ``examples/datasets/showCaseB4.m``.

Case B4 ships the same geological model discretised two ways, which is the
whole point of the example:

* ``stairstep_36x48.grdecl`` follows faults as a staircase through the logical
  index space, so every cell keeps a box shape and the fault surface is
  approximated;
* ``pillar_36x48.grdecl`` follows the fault along tilted pillars, so cells
  deform and the fault surface is represented exactly.

The two grids describe one reservoir and should agree on gross properties while
differing in geometry. That contrast is what the port checks, in place of the
side-by-side figure MRST ends on.

``cut_grdecl`` then extracts a logical sub-box from each, which MRST uses to
show the fault structure close up.

**The dataset is not installed here.** It is a 0.5 MB download and the
descriptor is already registered, so ``fetch_case_b4()`` will retrieve it. Until
then every function returns ``None`` and the self-tests report the absence
rather than failing, matching the rest of this tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.cut_grdecl import cut_grdecl
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.modules.deckformat.deckinput.read_grid import read_grid

STAIRSTEP_FILE = "stairstep_36x48.grdecl"
PILLAR_FILE = "pillar_36x48.grdecl"

# MRST's close-up box, 1-based inclusive as it writes it.
CUT_RANGES = ((12, 20), (13, 23), (1, 12))


@dataclass
class CaseB4Grids:
    """Both discretisations, and the sub-box cut from each."""

    stairstep: Any
    pillar: Any
    stairstep_cut: Any
    pillar_cut: Any


def case_b4_directory(root=None):
    """Locate the Case B4 files, or return ``None``. Never downloads."""
    if root is not None:
        root = Path(root)
        return root if (root / STAIRSTEP_FILE).exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        base = get_dataset_path("CaseB4", download=False)
        if base is not None and (Path(base) / STAIRSTEP_FILE).exists():
            return Path(base)
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "mrst-2024a-Geomech/examples/data/CaseB4"
        if (candidate / STAIRSTEP_FILE).exists():
            return candidate
    return None


def fetch_case_b4():
    """Download Case B4 through the dataset manager.

    Kept separate from the loaders and never called by them, so importing or
    testing this module cannot start a network transfer.
    """
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    return get_dataset_path("CaseB4", download=True)


def load_case_b4(root=None) -> CaseB4Grids | None:
    """Build both grids and their cut sub-boxes. ``None`` when data is absent."""
    directory = case_b4_directory(root)
    if directory is None:
        return None

    grids = {}
    for key, name in (("stairstep", STAIRSTEP_FILE), ("pillar", PILLAR_FILE)):
        grdecl = read_grid(directory / name)
        grids[key] = compute_geometry(process_grdecl(dict(grdecl)))
        # `cut_grdecl` takes the same 1-based inclusive ranges MRST writes.
        cut = cut_grdecl(dict(grdecl), [list(r) for r in CUT_RANGES])
        grids[key + "_cut"] = compute_geometry(process_grdecl(cut))

    return CaseB4Grids(stairstep=grids["stairstep"], pillar=grids["pillar"],
                       stairstep_cut=grids["stairstep_cut"],
                       pillar_cut=grids["pillar_cut"])


if __name__ == "__main__":
    def _skip_note():
        print("   (Case B4 is not installed; call fetch_case_b4() to download "
              "0.5 MB and enable these checks)")

    def test_dataset_absence_is_reported_not_raised():
        """Absent data must return None rather than failing the sweep."""
        directory = case_b4_directory()
        if directory is None:
            assert load_case_b4() is None
            print("ok test_dataset_absence_is_reported_not_raised (absent)")
            _skip_note()
        else:
            assert load_case_b4() is not None
            print(f"ok test_dataset_absence_is_reported_not_raised ({directory})")

    def test_both_discretisations_describe_one_reservoir():
        """Stair-step and pillar grids differ in geometry, not in extent.

        They discretise the same body, so their bounding boxes should agree
        closely while their cell shapes do not. A pair that disagrees on extent
        has had one of the two files mis-read.
        """
        case = load_case_b4()
        if case is None:
            print("ok test_both_discretisations_describe_one_reservoir (skipped)")
            return
        a = np.asarray(case.stairstep.nodes.coords, dtype=float)
        b = np.asarray(case.pillar.nodes.coords, dtype=float)
        span_a = a.max(axis=0) - a.min(axis=0)
        span_b = b.max(axis=0) - b.min(axis=0)
        assert np.allclose(span_a, span_b, rtol=0.05), (span_a, span_b)
        print(f"ok test_both_discretisations_describe_one_reservoir "
              f"({case.stairstep.cells.num} vs {case.pillar.cells.num} cells)")

    def test_cut_reduces_both_grids():
        """The sub-box must be strictly smaller and still valid."""
        case = load_case_b4()
        if case is None:
            print("ok test_cut_reduces_both_grids (skipped)")
            return
        for full, cut, name in ((case.stairstep, case.stairstep_cut, "stairstep"),
                                (case.pillar, case.pillar_cut, "pillar")):
            assert 0 < cut.cells.num < full.cells.num, (name, cut.cells.num)
            vol = np.asarray(cut.cells.volumes, dtype=float)
            assert np.all(vol > 0.0), (name, vol.min())
        print(f"ok test_cut_reduces_both_grids "
              f"({case.stairstep_cut.cells.num} and {case.pillar_cut.cells.num} cells)")

    def test_geometry_is_physical():
        case = load_case_b4()
        if case is None:
            print("ok test_geometry_is_physical (skipped)")
            return
        for g, name in ((case.stairstep, "stairstep"), (case.pillar, "pillar")):
            vol = np.asarray(g.cells.volumes, dtype=float)
            assert np.all(vol > 0.0), (name, vol.min())
        print("ok test_geometry_is_physical")

    def test_fetch_is_not_triggered_by_loading():
        """Loading must never start a download.

        A test sweep that silently pulls half a megabyte per run, or blocks on
        a prompt, is worse than one that skips.
        """
        import inspect
        source = inspect.getsource(load_case_b4) + inspect.getsource(case_b4_directory)
        assert "download=True" not in source, "a loader requests a download"
        assert "fetch_case_b4" not in source, "a loader calls the fetcher"
        print("ok test_fetch_is_not_triggered_by_loading")

    test_dataset_absence_is_reported_not_raised()
    test_both_discretisations_describe_one_reservoir()
    test_cut_reduces_both_grids()
    test_geometry_is_physical()
    test_fetch_is_not_triggered_by_loading()
    print("\nAll tests passed!")
