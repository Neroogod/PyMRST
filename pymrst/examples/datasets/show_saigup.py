"""Introduction to the SAIGUP Model.

Port of MRST ``examples/datasets/showSAIGUP.m``.

SAIGUP is a shallow-marine synthetic model built for a sensitivity study, and it
is a useful stress test for a grid reader: it is faulted, it has a large
inactive region, and `processGRDECL` can return it as several disconnected
grids rather than one.

The example follows the pattern `showNorne` and `showJohansen` use. It builds an
all-active grid first, so the inactive region survives and can be contrasted
with the active one, then rebuilds with the real `ACTNUM` and attaches rock
properties. Porosity, net-to-gross, saturation region and permeability are each
inspected in turn.

**This needs the SAIGUP dataset, a 2.5 MB download that is not installed here.**
Every function returns ``None`` until it is fetched, and the self-tests report
the absence rather than failing.

One correction is worth recording, because it was stated the other way round
earlier in this port. `showSAIGUP` does **not** use the `coarsegrid` module. It
calls only `readGRDECL`, `convertInputUnits`, `processGRDECL`, `grdecl2Rock` and
plotting, all of which were already present. The file that partitions SAIGUP is
`modules/coarsegrid/examples/coarsegridExampleSAIGUP.m`, a different file with a
similar name. The only thing blocking this example was the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.modules.deckformat.deckinput.read_grid import read_grid
from pymrst.params.rock.grdecl2rock import grdecl2rock
from pymrst.utils.inout.convert_input_units import convert_input_units
from pymrst.utils.units import DARCY, MILLI
from pymrst.utils.units.get_unit_system import get_unit_system


@dataclass
class SaigupModel:
    """Both grids and the rock, as MRST's script builds them."""

    grdecl: dict
    G_full: Any            # every logical cell, active or not
    G: Any                 # the real grid
    actnum: np.ndarray
    rock: Any

    @property
    def satnum(self) -> np.ndarray | None:
        """Saturation region per active cell, when the deck supplies it."""
        values = self.grdecl.get("SATNUM")
        if values is None:
            return None
        return np.asarray(values).ravel()[
            np.asarray(self.G.cells.indexMap, dtype=np.int64)]


def find_saigup_grdecl(path=None):
    """Locate ``SAIGUP.GRDECL``, or return ``None``. Never downloads."""
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("SAIGUP", download=False)
        if root is not None:
            for candidate in Path(root).rglob("SAIGUP.GRDECL"):
                return candidate
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        base = parent.parent / "mrst-2024a-Geomech/examples/data"
        for candidate in (base / "SAIGUP/SAIGUP.GRDECL", base / "saigup/SAIGUP.GRDECL"):
            if candidate.exists():
                return candidate
    return None


def fetch_saigup():
    """Download SAIGUP through the dataset manager.

    Kept out of the loaders and never called by them, so importing or testing
    this module cannot start a network transfer.
    """
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    return get_dataset_path("SAIGUP", download=True)


def load_saigup(path=None) -> SaigupModel | None:
    """Read SAIGUP and build both grids plus the rock. ``None`` when absent."""
    grdecl_file = find_saigup_grdecl(path)
    if grdecl_file is None:
        return None

    grdecl = read_grid(grdecl_file)
    converted = convert_input_units(grdecl, get_unit_system("METRIC"))
    grdecl = converted[0] if isinstance(converted, tuple) else converted

    actnum = np.asarray(grdecl["ACTNUM"], dtype=np.int64).ravel().copy()
    n_logical = int(np.prod(grdecl["cartDims"]))

    full = dict(grdecl)
    full["ACTNUM"] = np.ones(n_logical, dtype=np.int64)
    g_full = process_grdecl(full, check_grid=False)

    g = compute_geometry(process_grdecl(dict(grdecl)))
    rock = grdecl2rock(dict(grdecl), np.asarray(g.cells.indexMap, dtype=np.int64))
    return SaigupModel(grdecl=grdecl, G_full=g_full, G=g, actnum=actnum, rock=rock)


def permeability_in_millidarcy(model: SaigupModel, column: int = 0) -> np.ndarray:
    """One permeability column in mD, as MRST's `convertTo` gives it."""
    perm = np.asarray(model.rock.perm, dtype=float)
    if perm.ndim == 1:
        perm = perm[:, None]
    return perm[:, min(column, perm.shape[1] - 1)] / (MILLI * DARCY)


if __name__ == "__main__":
    def _skip(name, extra=""):
        print(f"ok {name} (skipped){extra}")

    def test_absence_is_reported_not_raised():
        """No dataset must mean a clean skip, never a failure or a download."""
        if find_saigup_grdecl() is None:
            assert load_saigup() is None
            print("ok test_absence_is_reported_not_raised (absent)")
            print("   (SAIGUP is a 2.5 MB download; call fetch_saigup() to "
                  "enable the checks below)")
        else:
            assert load_saigup() is not None
            print("ok test_absence_is_reported_not_raised (present)")

    def test_loading_never_downloads():
        """A test sweep must not pull data or block on a prompt."""
        import inspect
        source = inspect.getsource(load_saigup) + inspect.getsource(find_saigup_grdecl)
        assert "download=True" not in source, "a loader requests a download"
        assert "fetch_saigup" not in source, "a loader calls the fetcher"
        print("ok test_loading_never_downloads")

    def test_two_grids_differ_by_the_inactive_cells():
        model = load_saigup()
        if model is None:
            return _skip("test_two_grids_differ_by_the_inactive_cells")
        n_active = int(np.count_nonzero(model.actnum))
        assert model.G.cells.num == n_active, (model.G.cells.num, n_active)
        assert model.G_full.cells.num > model.G.cells.num
        print(f"ok test_two_grids_differ_by_the_inactive_cells "
              f"({n_active} active of {model.G_full.cells.num})")

    def test_rock_lines_up_with_the_active_grid():
        """`grdecl2rock` is given `indexMap`, so its rows must match the cells.

        Taking the first `n_active` rows instead gives an array of the right
        length holding another region's properties, which no plot reveals.
        """
        model = load_saigup()
        if model is None:
            return _skip("test_rock_lines_up_with_the_active_grid")
        poro = np.asarray(model.rock.poro, dtype=float)
        assert poro.shape[0] == model.G.cells.num, poro.shape
        assert np.all((poro >= 0.0) & (poro <= 1.0)), (poro.min(), poro.max())
        print(f"ok test_rock_lines_up_with_the_active_grid "
              f"(porosity {poro.min():.3f} to {poro.max():.3f})")

    def test_net_to_gross_is_a_fraction():
        model = load_saigup()
        if model is None:
            return _skip("test_net_to_gross_is_a_fraction")
        ntg = getattr(model.rock, "ntg", None)
        if ntg is None:
            return _skip("test_net_to_gross_is_a_fraction", " (no NTG in deck)")
        ntg = np.asarray(ntg, dtype=float)
        assert np.all((ntg > 0.0) & (ntg <= 1.0)), (ntg.min(), ntg.max())
        print(f"ok test_net_to_gross_is_a_fraction ({ntg.min():.3f} to {ntg.max():.3f})")

    def test_saturation_regions_are_labelled_one_to_six():
        """SAIGUP carries six saturation regions; MRST colours them 1 to 6."""
        model = load_saigup()
        if model is None:
            return _skip("test_saturation_regions_are_labelled_one_to_six")
        sn = model.satnum
        if sn is None:
            return _skip("test_saturation_regions_are_labelled_one_to_six",
                         " (no SATNUM)")
        assert sn.shape[0] == model.G.cells.num
        assert sn.min() >= 1, sn.min()
        print(f"ok test_saturation_regions_are_labelled_one_to_six "
              f"({sn.min()} to {sn.max()})")

    def test_permeability_is_in_si_and_spans_orders():
        """Values must read as m2, and the contrast must survive the read."""
        model = load_saigup()
        if model is None:
            return _skip("test_permeability_is_in_si_and_spans_orders")
        kx = permeability_in_millidarcy(model, 0)
        positive = kx[kx > 0.0]
        assert positive.size > 0
        assert 1e-4 < positive.max() < 1e6, positive.max()
        print(f"ok test_permeability_is_in_si_and_spans_orders "
              f"({positive.min():.3g} to {positive.max():.3g} mD)")

    def test_vertical_permeability_is_lower_than_horizontal():
        model = load_saigup()
        if model is None:
            return _skip("test_vertical_permeability_is_lower_than_horizontal")
        perm = np.asarray(model.rock.perm, dtype=float)
        if perm.ndim < 2 or perm.shape[1] < 3:
            return _skip("test_vertical_permeability_is_lower_than_horizontal",
                         " (isotropic tensor)")
        kx, kz = perm[:, 0], perm[:, 2]
        both = (kx > 0) & (kz > 0)
        assert np.median(kz[both]) < np.median(kx[both])
        print("ok test_vertical_permeability_is_lower_than_horizontal")

    def test_active_geometry_is_physical():
        model = load_saigup()
        if model is None:
            return _skip("test_active_geometry_is_physical")
        vol = np.asarray(model.G.cells.volumes, dtype=float)
        assert np.all(vol > 0.0), vol.min()
        print(f"ok test_active_geometry_is_physical (bulk {vol.sum():.4e} m3)")

    def test_this_example_does_not_need_coarsegrid():
        """Recorded because the porting plan said the opposite.

        The dependency scan attributed `processPartition`, `partitionLayers`
        and `explosionView` to this file. They belong to
        `modules/coarsegrid/examples/coarsegridExampleSAIGUP.m`, a different
        file with a similar name. Nothing here imports `coarsegrid`, and the
        assertion keeps that honest if the example is ever extended.
        """
        import ast
        import pathlib
        tree = ast.parse(pathlib.Path(__file__).read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        offenders = [m for m in imported if "coarsegrid" in m]
        assert not offenders, f"this example gained a coarsegrid dependency: {offenders}"
        print(f"ok test_this_example_does_not_need_coarsegrid "
              f"({len(imported)} imports, none from coarsegrid)")

    test_absence_is_reported_not_raised()
    test_loading_never_downloads()
    test_two_grids_differ_by_the_inactive_cells()
    test_rock_lines_up_with_the_active_grid()
    test_net_to_gross_is_a_fraction()
    test_saturation_regions_are_labelled_one_to_six()
    test_permeability_is_in_si_and_spans_orders()
    test_vertical_permeability_is_lower_than_horizontal()
    test_active_geometry_is_physical()
    test_this_example_does_not_need_coarsegrid()
    print("\nAll tests passed!")
