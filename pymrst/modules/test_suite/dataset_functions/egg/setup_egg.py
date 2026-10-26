"""The Egg case (MRST ``test-suite/dataset-functions/egg/setupEGG.m``).

Line for line, MRST's is::

    deck  = getDeckEGG(...);
    G     = computeGeometry(initEclipseGrid(deck));
    rock  = compressRock(initEclipseRock(deck), G.cells.indexMap);
    fluid = initDeckADIFluid(deck);
    gravity reset on
    pr     = 400*barsa;
    rz     = G.cells.centroids(1,3);
    dz     = G.cells.centroids(:,3) - rz;
    rhoO   = fluid.bO(400*barsa)*fluid.rhoOS;
    rhoW   = fluid.bW(400*barsa)*fluid.rhoWS;
    rhoMix = .1*rhoW + .9*rhoO;
    p0     = pr + norm(gravity)*rhoMix*dz;
    state  = initResSol(G, p0, [0.1, .90]);

Two details in that hydrostatic column are easy to get wrong and neither is
visible in the result:

* the reference depth is **the first cell's centroid**, not the deck's
  equilibrium datum. The datum is 4000 m; the shallowest cell centre is
  4002 m. Using the datum offsets the whole field by one half-layer of head;
* the density is the **mixture** at the initial saturations, `0.1*rhoW +
  0.9*rhoO`, not the oil density. Oil alone makes the column about 2% light,
  which at 24 m of relief is small - and stays small right up until someone
  reuses the function on a thicker reservoir.

`setupEGG` returns this state (MRST's `initResSol` struct) and turns gravity
on globally, as MRST does; `eggExample` does not use the state: it
equilibrates from `EQUIL` through `initEclipseProblemAD` instead.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
    init_deck_adi_fluid,
)
from pymrst.modules.deckformat.grid import init_eclipse_grid
from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
from pymrst.modules.deckformat.params.rock.init_eclipse_rock import (
    init_eclipse_rock,
)
from pymrst.modules.test_suite.dataset_functions.egg.get_deck_egg import (
    get_deck_egg,
)
from pymrst.utils.units import BARSA

def setup_egg(realization: int = 0, *, useACTNUM: bool = True,
              data_root=None) -> Tuple[Any, Any, Any, dict, Any]:
    """MRST `[G, rock, fluid, deck, state] = setupEGG('realization', n)`.

    `useACTNUM` is MRST's option, which `setupEGG` accepts and does not use;
    `data_root` (a PyMRST addition) points at another copy of the dataset.
    """
    from pymrst.params.gravity.gravity import gravity
    from pymrst.solvers.init_res_sol import init_res_sol
    deck = get_deck_egg(realization, data_root=data_root)
    G = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), G.cells.indexMap)
    fluid = init_deck_adi_fluid(deck)
    gravity("reset", "on")
    pr = 400*BARSA
    z = np.asarray(G.cells.centroids, dtype=float)[:, 2]
    rz = z[0]
    dz = z - rz
    rhoO = float(np.ravel(_call(fluid, "bO")(400*BARSA))[0])*_scalar(fluid, "rhoOS")
    rhoW = float(np.ravel(_call(fluid, "bW")(400*BARSA))[0])*_scalar(fluid, "rhoWS")
    rhoMix = .1*rhoW + .9*rhoO
    p0 = pr + np.linalg.norm(gravity())*rhoMix*dz
    state = init_res_sol(G, p0, np.array([0.1, .90]))
    return G, rock, fluid, deck, state


setupEGG = setup_egg


def _call(fluid: Any, name: str):
    """A fluid's property function, whichever spelling the fluid uses."""
    if isinstance(fluid, dict):
        if name in fluid:
            return fluid[name]
        raise KeyError(f"the Egg fluid has no {name!r}")
    found = getattr(fluid, name, None) or getattr(fluid, _snake(name), None)
    if found is None:
        raise AttributeError(f"the Egg fluid has no {name!r}")
    return found


def _snake(name: str) -> str:
    return name[0] + "_" + name[1:].lower()


def _scalar(fluid: Any, name: str) -> float:
    """A surface density, which a deck states per PVT region.

    The Egg deck has one region, so the array has one entry - but it *is* an
    array, and multiplying a `b` factor by it would give the whole column a
    one-element density rather than a number.
    """
    value = fluid[name] if isinstance(fluid, dict) else getattr(fluid, name)
    return float(np.ravel(np.asarray(value, dtype=float))[0])


__all__ = ["setupEGG", "setup_egg"]


if __name__ == "__main__":
    from pymrst.utils.units import DARCY, MILLI

    def _available() -> bool:
        from pymrst.modules.deckformat.egg.egg_dataset import (
            egg_data_directory,
        )
        try:
            egg_data_directory()
            return True
        except FileNotFoundError:
            return False

    if not _available():
        print("skip (Egg dataset not installed)")
        raise SystemExit(0)

    from pymrst.params.gravity.gravity import isolated_gravity
    with isolated_gravity():
        G, ROCK, FLUID, DECK, STATE = setup_egg()

    def test_the_grid_is_the_active_part_of_the_cartesian_box():
        """60x60x7 is 25200 cells; the Egg model's channel geometry leaves
        18553 of them active."""
        assert G.cells.num == 18553, G.cells.num
        assert np.asarray(G.cells.indexMap).size == 18553
        print("ok test_the_grid_is_the_active_part_of_the_cartesian_box")

    def test_the_cells_are_eight_by_eight_by_four_metres():
        assert np.allclose(G.cells.volumes, 8.0*8.0*4.0)
        print("ok test_the_cells_are_eight_by_eight_by_four_metres")

    def test_the_reservoir_sits_between_four_thousand_and_four_thousand_and_twenty_eight():
        z = np.asarray(G.cells.centroids)[:, 2]
        assert np.isclose(z.min(), 4002.0) and np.isclose(z.max(), 4026.0), (
            z.min(), z.max())
        print("ok test_the_reservoir_sits_between_four_thousand_and_four_thousand_and_twenty_eight")

    def test_the_rock_is_compressed_onto_the_active_cells():
        """`initEclipseRock` returns one row per Cartesian cell. Left
        uncompressed it is 25200 rows against a 18553-cell grid, and the
        permeability of every cell past the first inactive one is wrong."""
        assert np.asarray(ROCK.perm).shape[0] == G.cells.num
        assert np.asarray(ROCK.poro).size == G.cells.num
        assert np.allclose(ROCK.poro, 0.2)
        print("ok test_the_rock_is_compressed_onto_the_active_cells")

    def test_the_permeability_follows_the_deck_through_the_index_map():
        deck_kx = np.asarray(DECK["GRID"]["PERMX"], float)
        active = np.asarray(G.cells.indexMap, dtype=int)
        assert np.allclose(np.asarray(ROCK.perm)[:, 0], deck_kx[active])
        assert np.isclose(np.asarray(ROCK.perm)[:, 0].max()/(MILLI*DARCY),
                          7000.0)
        print("ok test_the_permeability_follows_the_deck_through_the_index_map")

    def test_the_fluid_carries_the_decks_surface_densities():
        assert np.isclose(_scalar(FLUID, "rhoOS"), 900.0)
        assert np.isclose(_scalar(FLUID, "rhoWS"), 1000.0)
        print("ok test_the_fluid_carries_the_decks_surface_densities")

    def test_the_initial_saturation_is_connate_water():
        assert np.allclose(np.asarray(STATE.s)[:, 0], 0.1)
        assert np.allclose(np.asarray(STATE.s).sum(axis=1), 1.0)
        print("ok test_the_initial_saturation_is_connate_water")

    def test_the_first_cell_sits_exactly_at_the_reference_pressure():
        """The reference depth is the first cell's centroid, so its own
        pressure is `pr` with no head at all. Using the deck's 4000 m datum
        instead puts 2 m of head on it."""
        assert np.isclose(float(STATE.pressure[0]), 400*BARSA), (
            float(STATE.pressure[0])/BARSA)
        print("ok test_the_first_cell_sits_exactly_at_the_reference_pressure")

    def test_the_column_uses_the_mixture_density_not_the_oil_density():
        """`0.1*rhoW + 0.9*rhoO` = 910, against 900 for oil alone. Over the
        24 m of relief that is 2.4 kPa - small here, and the reason the error
        survives unnoticed until the function meets a thicker reservoir."""
        z = np.asarray(G.cells.centroids)[:, 2]
        deepest = int(np.argmax(z))
        head = float(STATE.pressure[deepest]) - 400*BARSA
        expected = 9.80665*910.0*(z[deepest] - z[0])
        assert np.isclose(head, expected, rtol=1e-3), (head, expected)
        assert not np.isclose(head, 9.80665*900.0*(z[deepest] - z[0]),
                              rtol=1e-4)
        print("ok test_the_column_uses_the_mixture_density_not_the_oil_density")

    def test_pressure_increases_with_depth():
        z = np.asarray(G.cells.centroids)[:, 2]
        p = np.asarray(STATE.pressure)
        order = np.argsort(z)
        assert np.all(np.diff(p[order]) >= -1e-9)
        print("ok test_pressure_increases_with_depth")

    def test_a_realization_changes_the_rock_and_nothing_else():
        """Compared in millidarcy: a permeability in SI is of order 1e-13 and
        `allclose`'s default `atol` is 1e-8, so any two permeability fields
        compare equal in SI whatever the reader did."""
        with isolated_gravity():
            _, rock1, _, _, state1 = setup_egg(realization=1)
        assert not np.allclose(np.asarray(rock1.perm)[:, 0]/(MILLI*DARCY),
                               np.asarray(ROCK.perm)[:, 0]/(MILLI*DARCY))
        assert np.allclose(np.asarray(state1.pressure),
                           np.asarray(STATE.pressure))
        print("ok test_a_realization_changes_the_rock_and_nothing_else")

    test_the_grid_is_the_active_part_of_the_cartesian_box()
    test_the_cells_are_eight_by_eight_by_four_metres()
    test_the_reservoir_sits_between_four_thousand_and_four_thousand_and_twenty_eight()
    test_the_rock_is_compressed_onto_the_active_cells()
    test_the_permeability_follows_the_deck_through_the_index_map()
    test_the_fluid_carries_the_decks_surface_densities()
    test_the_initial_saturation_is_connate_water()
    test_the_first_cell_sits_exactly_at_the_reference_pressure()
    test_the_column_uses_the_mixture_density_not_the_oil_density()
    test_pressure_increases_with_depth()
    test_a_realization_changes_the_rock_and_nothing_else()
    print("\nAll tests passed!")
