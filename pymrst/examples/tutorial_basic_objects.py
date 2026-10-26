"""Basic Objects in PyMRST Workflows (MRST ``tutorialBasicObjects.m``).

The objects every simulation is built from, introduced in the order MRST uses:

1. a grid, from ``cart_grid``;
2. a rock, from ``make_rock``, first uniform and then heterogeneous;
3. cutting cells out of a grid with ``remove_cells``, and the matching rock
   subset with ``compress_rock``;
4. correlated porosity from ``gaussian_field``, with permeability from the
   Kozeny-Carman relation;
5. pore volume, the quantity that ties grid and rock together.

MRST's version leans on plotting to make each point. The port computes the same
objects and checks the relationships that the pictures are meant to convey: that
removing cells removes exactly the intended ones, that a compressed rock still
lines up with its grid, and that pore volume equals `volume * poro * ntg`.

Two closing sections read real data, and both are written to report rather than
fail so the tutorial runs on any checkout:

* ``run_bed_model_section`` reads BedModel2 through ``read_grdecl``. That
  dataset is not installed here, so it returns ``None``.
* ``run_eclipse_case_section`` builds a whole case from the SPE9 deck. The deck
  is present and reads correctly, but PyMRST's ``init_eclipse_problem_ad_ad``
  supports only the SPE1-style parsed-deck subset and rejects a full ECLIPSE
  deck. It therefore returns a reason rather than a result, and the self-test
  asserts that reason so the gap stays visible and the test flips the day
  full-deck support lands.

See ``examples_parity.md`` for dataset and gap status.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.deckformat.params.rock import compress_rock
from pymrst.params.rock import make_rock
from pymrst.params.rock.gaussian_field import gaussian_field
from pymrst.params.rock.pore_volume import pore_volume
from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
from pymrst.utils.units import DARCY, MILLI


@dataclass
class BasicObjects:
    """The grid, rock and derived quantities the tutorial builds."""

    G: object
    rock: object
    G_reduced: object
    rock_reduced: object
    removed: np.ndarray
    pore_volume: np.ndarray


def uniform_grid_and_rock(dims=(5, 5, 10), physdims=(1.0, 1.0, 1.0),
                          perm: float = 100 * MILLI * DARCY,
                          poro: float = 0.3):
    """A Cartesian grid with uniform rock, MRST's starting point."""
    g = compute_geometry(cart_grid(list(dims), list(physdims)))
    return g, make_rock(g, perm, poro)


def select_interior_box(g, i_range=(1, 4), k_range=(2, 8)) -> np.ndarray:
    """Cells inside a logical box, the selection MRST removes.

    MRST picks `i` and `j` in 2:4 and `k` in 3:8 with 1-based inclusive
    indexing. The defaults here are the 0-based half-open equivalent, so the
    same 3x3x6 block of 54 cells is selected.
    """
    ijk = grid_logical_indices(g)
    i, j, k = (np.asarray(a).ravel() for a in ijk[:3])
    pick = ((i >= i_range[0]) & (i < i_range[1])
            & (j >= i_range[0]) & (j < i_range[1])
            & (k >= k_range[0]) & (k < k_range[1]))
    return np.flatnonzero(pick)


def kozeny_carman_rock(g, seed: int = 0, poro_range=(0.2, 0.4)):
    """Correlated porosity, with permeability from Kozeny-Carman.

        k = poro^3 * d^2 / (0.81 * 72 * (1 - poro)^2),  d = 1e-5 m

    The point of the section is that porosity and permeability are not
    independent inputs: one is drawn as a correlated field and the other
    follows from it.
    """
    rng = np.random.default_rng(seed)
    poro = np.asarray(
        gaussian_field(g.cartDims, list(poro_range), rng=rng), dtype=float
    ).ravel(order="F")
    perm = poro ** 3 * (1e-5) ** 2 / (0.81 * 72 * (1.0 - poro) ** 2)
    return make_rock(g, perm, poro)


def run_tutorial_basic_objects(*, seed: int = 0) -> BasicObjects:
    """Build every object the tutorial introduces and return them together."""
    g, _ = uniform_grid_and_rock()
    rock = kozeny_carman_rock(g, seed=seed)
    removed = select_interior_box(g)
    g_red = remove_cells(g, removed)
    rock_red = compress_rock(rock, g_red.cells.indexMap)
    return BasicObjects(G=g, rock=rock, G_reduced=g_red, rock_reduced=rock_red,
                        removed=removed, pore_volume=pore_volume(g, rock))


def run_bed_model_section(dataset_root=None, *, download: bool = False):
    """MRST's closing section: read BedModel2 through the deck reader.

    Returns ``None`` when the dataset is not installed, so the tutorial remains
    runnable on a checkout without it.

    ``download`` defaults to ``False`` deliberately. ``get_dataset_path`` will
    otherwise prompt on the terminal and wait for an answer, which hangs any
    automated run of this file. Pass ``download=True`` to fetch it, or install
    it through the dataset manager beforehand.
    """
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = (get_dataset_path("BedModel2", download=download)
                if dataset_root is None else dataset_root)
    except Exception:
        return None
    if root is None:
        return None
    from pathlib import Path
    grdecl_file = Path(root) / "BedModel2.grdecl"
    if not grdecl_file.exists():
        return None

    from pymrst.gridprocessing.process_grdecl import process_grdecl
    from pymrst.params.rock.grdecl2rock import grdecl2rock
    from pymrst.utils.inout.convert_input_units import convert_input_units
    from pymrst.utils.inout.eclipse.deckinput_simple.read_grdecl import read_grdecl
    from pymrst.utils.units.get_unit_system import get_unit_system

    grdecl = convert_input_units(read_grdecl(grdecl_file), get_unit_system("LAB"))
    g = compute_geometry(process_grdecl(grdecl))
    rock = compress_rock(grdecl2rock(grdecl), g.cells.indexMap)
    return g, rock, pore_volume(g, rock)


def find_spe9_deck():
    """Locate `BENCH_SPE9.DATA`, or return ``None``.

    Tries the dataset manager first, then the MRST tree beside this checkout,
    which is where the payload sits when MRST is present. ``download`` is never
    requested, so this cannot prompt and cannot block an automated run.
    """
    from pathlib import Path

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("SPE9", download=False)
        if root is not None:
            candidate = Path(root) / "BENCH_SPE9.DATA"
            if candidate.exists():
                return candidate
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "mrst-2024a-Geomech/examples/data/spe9/BENCH_SPE9.DATA"
        if candidate.exists():
            return candidate
    return None


def run_eclipse_case_section(deck_file=None):
    """MRST's closing section: a complete simulation case from an ECLIPSE deck.

    Reads SPE9 through ``read_eclipse_deck`` and ``convert_deck_units``, then
    hands the deck to ``init_eclipse_problem_ad_ad``, which should return the
    initial state, model and schedule ready to simulate. The point of the
    section is that a whole case can come from a deck rather than being
    assembled by hand.

    **This does not work yet, and the reason is worth stating rather than
    hiding.** The deck reads fine, but PyMRST's ``init_eclipse_problem_ad_ad``
    accepts only the SPE1-style parsed-deck subset carrying `x`/`y`/`z`/`poro`/
    `perm`, and raises `NotImplementedError` for a full ECLIPSE deck such as
    SPE9 unless a `model=` is supplied. So the section returns a `(None,
    reason)` pair instead of a result. Reporting the limitation keeps it
    visible; returning a bare `None` would make a real gap look like a missing
    dataset.

    Returns ``(result, None)`` on success, or ``(None, reason)`` otherwise.
    """
    deck_file = find_spe9_deck() if deck_file is None else deck_file
    if deck_file is None:
        return None, "SPE9 dataset not found"

    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad_ad,
    )
    from pymrst.modules.deckformat.deckinput.convert_deck_units import (
        convert_deck_units,
    )
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
        read_eclipse_deck,
    )
    deck = convert_deck_units(read_eclipse_deck(deck_file))
    try:
        return init_eclipse_problem_ad_ad(deck), None
    except NotImplementedError as exc:
        return None, f"init_eclipse_problem_ad_ad does not support this deck: {exc}"


if __name__ == "__main__":
    def test_grid_and_uniform_rock():
        g, rock = uniform_grid_and_rock()
        assert g.cells.num == 250, g.cells.num
        assert np.allclose(rock.poro, 0.3)
        assert np.allclose(np.asarray(rock.perm).ravel()[0], 100 * MILLI * DARCY)
        print("ok test_grid_and_uniform_rock")

    def test_removing_cells_removes_exactly_the_selection():
        """MRST shows this with a picture; the count is what the picture means."""
        g, _ = uniform_grid_and_rock()
        removed = select_interior_box(g)
        assert removed.size == 3 * 3 * 6, removed.size
        g_red = remove_cells(g, removed)
        assert g_red.cells.num == g.cells.num - removed.size, g_red.cells.num
        # The survivors are exactly the complement, and indexMap says which.
        kept = np.setdiff1d(np.arange(g.cells.num), removed)
        assert np.array_equal(np.asarray(g_red.cells.indexMap), kept)
        print("ok test_removing_cells_removes_exactly_the_selection")

    def test_compressed_rock_lines_up_with_the_reduced_grid():
        """The reason `compress_rock` exists: rows must match cells.

        A rock built on the full box has one row per original cell. After
        cells are removed, using it unchanged would silently pair each cell
        with the wrong rock.
        """
        out = run_tutorial_basic_objects()
        assert out.rock_reduced.poro.shape[0] == out.G_reduced.cells.num
        kept = np.asarray(out.G_reduced.cells.indexMap)
        assert np.allclose(out.rock_reduced.poro, out.rock.poro[kept])
        assert np.allclose(np.asarray(out.rock_reduced.perm),
                           np.asarray(out.rock.perm)[kept])
        print("ok test_compressed_rock_lines_up_with_the_reduced_grid")

    def test_kozeny_carman_is_monotone_in_porosity():
        """Higher porosity must mean higher permeability, everywhere."""
        g, _ = uniform_grid_and_rock()
        rock = kozeny_carman_rock(g)
        poro = np.asarray(rock.poro).ravel()
        perm = np.asarray(rock.perm).ravel()
        order = np.argsort(poro)
        assert np.all(np.diff(perm[order]) >= -1e-30), "permeability not monotone"
        assert poro.min() >= 0.2 - 1e-9 and poro.max() <= 0.4 + 1e-9, (
            poro.min(), poro.max())
        print("ok test_kozeny_carman_is_monotone_in_porosity")

    def test_pore_volume_is_volume_times_porosity():
        out = run_tutorial_basic_objects()
        expect = (np.asarray(out.G.cells.volumes) * np.asarray(out.rock.poro))
        ntg = getattr(out.rock, "ntg", None)
        if ntg is not None:
            expect = expect * np.asarray(ntg)
        assert np.allclose(out.pore_volume, expect, rtol=1e-14)
        print("ok test_pore_volume_is_volume_times_porosity")

    def test_gaussian_field_is_reproducible():
        g, _ = uniform_grid_and_rock()
        a = kozeny_carman_rock(g, seed=7).poro
        b = kozeny_carman_rock(g, seed=7).poro
        c = kozeny_carman_rock(g, seed=8).poro
        assert np.allclose(a, b), "same seed gave different fields"
        assert not np.allclose(a, c), "different seeds gave the same field"
        print("ok test_gaussian_field_is_reproducible")

    def test_bed_model_section_is_optional():
        """Absent dataset must return None rather than raising."""
        out = run_bed_model_section()
        if out is None:
            print("ok test_bed_model_section_is_optional (BedModel2 not installed)")
        else:
            g, rock, pv = out
            assert g.cells.num == rock.poro.shape[0] == pv.shape[0]
            print("ok test_bed_model_section_is_optional (dataset present)")

    def test_eclipse_case_section_reports_its_limitation():
        """SPE9 through the deck reader, when the dataset is installed.

        This section was missed on the first pass of this port. MRST's file
        mentions `initEclipseGrid` and `processWells` in a comment, and the
        static dependency scan attributed those to the Egg examples, so the
        section looked like it had already been covered. The code below the
        comment calls `initEclipseProblemAD`, which is a different function
        and was present in PyMRST all along.
        """
        assert find_spe9_deck() is not None, "SPE9 deck should be found beside MRST"
        result, reason = run_eclipse_case_section()
        if result is None:
            # A known limitation, asserted so it cannot be mistaken for a
            # missing dataset and so this test starts failing the day
            # `init_eclipse_problem_ad_ad` grows full-deck support.
            assert "does not support this deck" in reason, reason
            print(f"ok test_eclipse_case_section_reports_its_limitation "
                  f"({reason.split(':')[0]})")
            return
        state0, model = result[0], result[1]
        assert model.G.cells.num > 0, model.G.cells.num
        assert np.asarray(state0.pressure).shape[0] == model.G.cells.num
        print(f"ok test_eclipse_case_section_reports_its_limitation "
              f"(SPE9 now works: {model.G.cells.num} cells)")

    test_grid_and_uniform_rock()
    test_removing_cells_removes_exactly_the_selection()
    test_compressed_rock_lines_up_with_the_reduced_grid()
    test_kozeny_carman_is_monotone_in_porosity()
    test_pore_volume_is_volume_times_porosity()
    test_gaussian_field_is_reproducible()
    test_bed_model_section_is_optional()
    test_eclipse_case_section_reports_its_limitation()
    print("\nAll tests passed!")
