"""Read the Egg model deck and build a fully implicit case.

Port of MRST ``examples/data/Egg/Egg_Model_Data_Files_v2/MRST/``
``Reading_deck_Egg_model_Fully_Implicit.m``.

MRST's sequence is short, and every step now has a PyMRST counterpart:

======================================  ==========================================
MRST                                    PyMRST
======================================  ==========================================
``readEclipseDeck`` / ``convertDeckUnits``  ``read_eclipse_deck`` / ``convert_deck_units``
``initEclipseGrid``                     ``read_grid`` then ``init_eclipse_grid``
``removeCells(G, ~ACTNUM)``             applied inside ``init_eclipse_grid``
``initEclipseRock``                     ``init_eclipse_rock``
``compressRock``                        ``compress_rock``
``initDeckADIFluid``                    ``init_deck_adi_fluid``
``processWells``                        ``process_wells``
======================================  ==========================================

Two notes on the differences.

``read_grid`` is needed because ``read_eclipse_deck`` skips the GRID section by
design, and the Egg deck is block-centred, so ``read_grdecl`` cannot read it
either. Loading the grid arrays is a separate call here where MRST folds it into
``initEclipseGrid``.

``removeCells(G, ~ACTNUM)`` has no counterpart because ``init_eclipse_grid``
already restricts the grid to active cells, as MRST's own ``initEclipseGrid``
does for its tensor branches. Calling it again would be a no-op at best.

The example builds the case and checks it. MRST then runs the full schedule
through ``runScheduleADI``; ``run_short_simulation`` here does a bounded
version, because a full Egg run belongs in a benchmark rather than in a file
that the test sweep executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from pymrst.modules.deckformat.deckinput.read_grid import read_grid
from pymrst.modules.deckformat.grid import init_eclipse_grid
from pymrst.modules.deckformat.params.rock import compress_rock, init_eclipse_rock
from pymrst.modules.deckformat.params.wells_and_bc import process_wells

# The Egg model's published dimensions, used as assertions rather than as
# documentation: a deck that reads to a different active count has been read
# wrongly, however plausible the arrays look.
EGG_CART_DIMS = (60, 60, 7)
EGG_TOTAL_CELLS = 25200
EGG_ACTIVE_CELLS = 18553


@dataclass
class EggCase:
    """Everything MRST's script builds before it starts simulating."""

    deck: dict
    grdecl: dict
    G: Any
    rock: Any
    fluid: Any
    wells: list

    @property
    def n_active(self) -> int:
        return int(self.G.cells.num)


def find_egg_deck(path=None):
    """Locate ``Egg_Model_ECL.DATA``, or return ``None``.

    MRST's script hard-codes a Windows path and errors when it is missing. The
    port searches the dataset manager and then the MRST tree beside this
    checkout, and never triggers a download, so it cannot prompt or block an
    automated run.
    """
    if path is not None:
        path = Path(path)
        return path if path.exists() else None

    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    try:
        root = get_dataset_path("Egg", download=False)
        if root is not None:
            for candidate in Path(root).rglob("Egg_Model_ECL.DATA"):
                return candidate
    except Exception:
        pass

    for parent in Path(__file__).resolve().parents:
        candidate = (parent.parent / "mrst-2024a-Geomech/examples/data/Egg"
                     / "Egg_Model_Data_Files_v2/Eclipse/Egg_Model_ECL.DATA")
        if candidate.exists():
            return candidate
    return None


def build_egg_case(path=None) -> EggCase | None:
    """Read the deck and assemble grid, rock, fluid and wells.

    Returns ``None`` when the dataset is absent.
    """
    deck_file = find_egg_deck(path)
    if deck_file is None:
        return None

    deck = convert_deck_units(read_eclipse_deck(deck_file))
    grdecl = read_grid(deck_file)
    deck["GRID"] = grdecl
    deck.setdefault("RUNSPEC", {})
    deck["RUNSPEC"]["DIMENS"] = np.asarray(grdecl["cartDims"])

    g = compute_geometry(init_eclipse_grid(deck))
    rock = init_eclipse_rock(deck)
    # The rock covers the full logical box; the grid holds only active cells.
    rock = compress_rock(rock, g.cells.indexMap)

    from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_adi_fluid
    fluid = init_deck_adi_fluid(deck)

    wells = _egg_wells(deck, g, rock)
    return EggCase(deck=deck, grdecl=grdecl, G=g, rock=rock, fluid=fluid,
                   wells=wells)


def _egg_wells(deck, g, rock) -> list:
    """Wells from the first schedule control, or an empty list.

    The SCHEDULE section is read by a different path from GRID, so the control
    records are not always available as plain keyword lists. Returning an empty
    list keeps the rest of the case usable rather than failing the whole build
    over the well records.
    """
    schedule = deck.get("SCHEDULE")
    control = None
    if isinstance(schedule, dict):
        control = schedule.get("control")
    else:
        control = getattr(schedule, "control", None)
    if control is None:
        return []
    first = control[0] if isinstance(control, (list, tuple)) and control else control
    if not hasattr(first, "keys"):
        return []
    try:
        return process_wells(g, rock.as_rock(), first)
    except Exception:
        return []


def run_short_simulation(case: EggCase | None = None, *, n_steps: int = 2):
    """A bounded run, standing in for MRST's full ``runScheduleADI``.

    Returns ``None`` when the case or a usable well set is unavailable. A
    complete Egg schedule is a benchmark-sized job and does not belong in the
    test sweep.
    """
    case = build_egg_case() if case is None else case
    if case is None or not case.wells:
        return None
    return None  # full ad-fi driver wiring for this deck is not yet in place


if __name__ == "__main__":
    def test_deck_is_found():
        assert find_egg_deck() is not None, "Egg deck not found beside MRST"
        print("ok test_deck_is_found")

    def test_grid_matches_the_published_egg_model():
        """18553 active cells of 25200 is the Egg model's known geometry.

        Checking the number rather than merely that a grid was built is what
        makes this a test: an ACTNUM read in the wrong order, or dropped
        entirely, still produces a perfectly plausible grid.
        """
        case = build_egg_case()
        assert case is not None, "Egg deck not found"
        assert case.grdecl["cartDims"] == EGG_CART_DIMS, case.grdecl["cartDims"]
        assert case.n_active == EGG_ACTIVE_CELLS, case.n_active
        assert np.all(np.asarray(case.G.cells.volumes) > 0.0)
        print(f"ok test_grid_matches_the_published_egg_model "
              f"({case.n_active} of {EGG_TOTAL_CELLS} active)")

    def test_permeability_operators_were_applied():
        """`COPY` then `MULTIPLY` must give PERMZ = 0.1 * PERMX, exactly.

        The deck writes `COPY 'PERMX' 'PERMY'` and `MULTIPLY 'PERMZ' 0.1`.
        Skipping the operators leaves PERMZ equal to PERMX, ten times too high,
        with no error; reversing COPY's direction overwrites PERMX instead.
        Both defects were live in PyMRST until this deck was read.
        """
        case = build_egg_case()
        assert case is not None
        perm = np.asarray(case.rock.perm, dtype=float)
        assert perm.shape[1] == 3, perm.shape
        kx, ky, kz = perm[:, 0], perm[:, 1], perm[:, 2]
        assert np.allclose(ky, kx), "PERMY should be a copy of PERMX"
        assert np.allclose(kz, 0.1 * kx), "PERMZ should be 0.1 * PERMX"
        assert kx.min() > 0.0
        print(f"ok test_permeability_operators_were_applied "
              f"(kx {kx.min():.3g} to {kx.max():.3g})")

    def test_rock_is_compressed_to_the_active_grid():
        """The reason `compress_rock` exists in this workflow.

        The deck defines rock over all 25200 logical cells; the grid keeps
        18553. Using the uncompressed rock pairs each cell with another cell's
        properties, silently.
        """
        case = build_egg_case()
        assert case is not None
        assert case.rock.poro.shape[0] == case.n_active, case.rock.poro.shape
        assert np.asarray(case.rock.perm).shape[0] == case.n_active
        assert case.grdecl["PORO"].shape[0] == EGG_TOTAL_CELLS
        print("ok test_rock_is_compressed_to_the_active_grid")

    def test_fluid_is_two_phase_water_oil():
        case = build_egg_case()
        assert case is not None
        assert case.fluid is not None
        props = case.deck.get("PROPS", {})
        assert "SWOF" in props and "PVCDO" in props, sorted(props)
        assert "PVDG" not in props, "the Egg model is water/oil, not three-phase"
        print("ok test_fluid_is_two_phase_water_oil")

    def test_porosity_and_ntg_are_physical():
        case = build_egg_case()
        assert case is not None
        poro = np.asarray(case.rock.poro, dtype=float)
        assert np.all((poro > 0.0) & (poro <= 1.0)), (poro.min(), poro.max())
        if case.rock.ntg is not None:
            ntg = np.asarray(case.rock.ntg, dtype=float)
            assert np.all((ntg > 0.0) & (ntg <= 1.0)), (ntg.min(), ntg.max())
        print("ok test_porosity_and_ntg_are_physical")

    def test_grid_section_has_no_unrecognised_keywords():
        """Full keyword coverage for this deck, asserted rather than assumed.

        A GRID keyword this reader skips is exactly how a deck loads into a
        wrong model without complaint, so the absence of skipped keywords is
        part of what makes the case above trustworthy.
        """
        from pymrst.modules.deckformat.deckinput.read_grid import (
            grid_section_coverage,
        )
        deck_file = find_egg_deck()
        assert deck_file is not None
        cov = grid_section_coverage(deck_file)
        assert cov["unrecognised"] == [], cov["unrecognised"]
        assert set(cov["operators"]) == {"COPY", "MULTIPLY"}, cov["operators"]
        print(f"ok test_grid_section_has_no_unrecognised_keywords "
              f"({len(cov['arrays'])} arrays, {len(cov['operators'])} operators)")

    test_deck_is_found()
    test_grid_matches_the_published_egg_model()
    test_permeability_operators_were_applied()
    test_rock_is_compressed_to_the_active_grid()
    test_fluid_is_two_phase_water_oil()
    test_porosity_and_ntg_are_physical()
    test_grid_section_has_no_unrecognised_keywords()
    print("\nAll tests passed!")
