"""Read the Egg model deck and build a sequential water-flood case.

Port of MRST ``examples/data/Egg/Egg_Model_Data_Files_v2/MRST/``
``Reading_deck_Egg_model_Sequential.m``.

The same deck as the fully implicit example, set up differently. Two changes
matter:

* the fluid is a Corey model written in the script rather than the deck's PVT
  tables, so the case runs under the incompressible sequential solver;
* the wells are placed explicitly at eight `(I, J)` pairs rather than being read
  from the SCHEDULE section, each injecting 79.5 m3/day through the full column.

Building the wells by hand is what makes this example useful as a check: the
injector positions are known, so the completions can be verified cell by cell,
which a deck-driven well set cannot offer without trusting the reader under
test.

Grid and rock construction is shared with the fully implicit example and
imported from it rather than repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.examples.data.egg.reading_deck_egg_model_fully_implicit import (
    EGG_ACTIVE_CELLS,
    EggCase,
    build_egg_case,
    find_egg_deck,
)
from pymrst.modules.incomp.fluid.incompressible.init_corey_fluid import (
    init_corey_fluid,
)
from pymrst.params.wells_and_bc.vertical_well import vertical_well
from pymrst.utils.units import CENTI, DAY, METER, POISE

# MRST's eight injector locations, one-based as the script writes them.
INJECTOR_I = (5, 30, 2, 27, 50, 8, 32, 57)
INJECTOR_J = (57, 53, 35, 29, 35, 9, 2, 6)
WATER_INJECTION_RATE = 79.5 * METER ** 3 / DAY
WELL_DIAMETER = 0.2


@dataclass
class EggSequentialCase:
    """The Egg case set up for the sequential incompressible solver."""

    base: EggCase
    fluid: Any
    wells: list

    @property
    def G(self):
        return self.base.G

    @property
    def rock(self):
        return self.base.rock


def corey_fluid():
    """MRST's hand-written Corey fluid for this example.

    Water and oil at 1 and 5 cP, densities 1000 and 900, Corey exponents 3 and
    4, residual saturations 0.2 and 0.1, and end-point relperms 0.749 and 0.8.
    """
    return init_corey_fluid(
        mu=[1.0 * CENTI * POISE, 5.0 * CENTI * POISE],
        rho=[1000.0, 900.0],
        n=[3.0, 4.0],
        sr=[0.2, 0.1],
        kwm=[0.749, 0.8],
    )


def place_injectors(g, rock, *, rate: float = WATER_INJECTION_RATE) -> list:
    """Eight vertical injectors at MRST's `(I, J)` pairs.

    The deck indices are one-based; `vertical_well` takes zero-based logical
    indices, so each is decremented. Getting that wrong places every injector
    one cell off in both directions, and the flood still looks plausible.
    """
    wells: list = []
    for index, (i, j) in enumerate(zip(INJECTOR_I, INJECTOR_J), start=1):
        wells = vertical_well(
            wells, g, rock, int(i) - 1, int(j) - 1, None,
            well_type="rate", val=float(rate), radius=WELL_DIAMETER / 2.0,
            compi=np.array([1.0, 0.0]), name=f"I{index}",
        )
    return wells


def build_egg_sequential_case(path=None) -> EggSequentialCase | None:
    """Grid and rock from the deck, with a Corey fluid and explicit injectors."""
    base = build_egg_case(path)
    if base is None:
        return None
    fluid = corey_fluid()
    wells = place_injectors(base.G, base.rock.as_rock())
    return EggSequentialCase(base=base, fluid=fluid, wells=wells)


if __name__ == "__main__":
    def test_case_builds_on_the_egg_grid():
        case = build_egg_sequential_case()
        assert case is not None, "Egg deck not found"
        assert case.G.cells.num == EGG_ACTIVE_CELLS, case.G.cells.num
        assert case.rock.poro.shape[0] == case.G.cells.num
        print(f"ok test_case_builds_on_the_egg_grid ({case.G.cells.num} cells)")

    def test_eight_injectors_are_placed():
        case = build_egg_sequential_case()
        assert case is not None
        assert len(case.wells) == 8, len(case.wells)
        assert [w.name for w in case.wells] == [f"I{i}" for i in range(1, 9)]
        for w in case.wells:
            assert w.type == "rate", w.type
            assert np.isclose(float(w.val), WATER_INJECTION_RATE)
        print("ok test_eight_injectors_are_placed")

    def test_injectors_land_on_the_named_columns():
        """One-based deck indices become zero-based logical indices.

        Every completion of injector `n` must sit in logical column
        `(I[n]-1, J[n]-1)`. An off-by-one here shifts all eight injectors and
        the water flood still looks entirely reasonable.
        """
        from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
        case = build_egg_sequential_case()
        assert case is not None
        ijk = grid_logical_indices(case.G)
        gi, gj = np.asarray(ijk[0]).ravel(), np.asarray(ijk[1]).ravel()
        for n, well in enumerate(case.wells):
            cells = np.asarray(well.cells, dtype=np.int64)
            assert cells.size > 0, well.name
            assert np.all(gi[cells] == INJECTOR_I[n] - 1), (well.name, gi[cells])
            assert np.all(gj[cells] == INJECTOR_J[n] - 1), (well.name, gj[cells])
        print("ok test_injectors_land_on_the_named_columns")

    def test_completions_skip_inactive_layers():
        """The Egg model is channelised, so a column may not be fully active.

        A vertical well spans the layers that exist in that column, which is at
        most the seven of the full model and often fewer.
        """
        case = build_egg_sequential_case()
        assert case is not None
        counts = [np.asarray(w.cells).size for w in case.wells]
        assert all(1 <= c <= 7 for c in counts), counts
        print(f"ok test_completions_skip_inactive_layers (per-well cells: {counts})")

    def test_corey_fluid_endpoints():
        """Relperms must vanish at residual saturation and cap at kwm."""
        fluid = corey_fluid()
        kr_lo = np.atleast_1d(np.asarray(fluid.relperm(0.2), dtype=float)).ravel()
        kr_hi = np.atleast_1d(np.asarray(fluid.relperm(0.9), dtype=float)).ravel()
        assert np.isclose(kr_lo[0], 0.0, atol=1e-12), kr_lo
        assert np.isclose(kr_hi[1], 0.0, atol=1e-12), kr_hi
        assert 0.0 < kr_hi[0] <= 0.749 + 1e-12, kr_hi[0]
        print("ok test_corey_fluid_endpoints")

    def test_well_indices_are_positive():
        case = build_egg_sequential_case()
        assert case is not None
        for w in case.wells:
            wi = np.asarray(w.WI, dtype=float)
            assert wi.size == np.asarray(w.cells).size, (w.name, wi.size)
            assert np.all(wi > 0.0), (w.name, wi)
        print("ok test_well_indices_are_positive")

    def test_deck_is_shared_with_the_fully_implicit_example():
        assert find_egg_deck() is not None
        print("ok test_deck_is_shared_with_the_fully_implicit_example")

    test_case_builds_on_the_egg_grid()
    test_eight_injectors_are_placed()
    test_injectors_land_on_the_named_columns()
    test_completions_skip_inactive_layers()
    test_corey_fluid_endpoints()
    test_well_indices_are_positive()
    test_deck_is_shared_with_the_fully_implicit_example()
    print("\nAll tests passed!")
