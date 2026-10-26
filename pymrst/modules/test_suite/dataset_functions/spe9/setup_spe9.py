"""The SPE9 case (MRST ``test-suite/dataset-functions/spe9/setupSPE9.m``).

MRST's, line for line::

    deck  = convertDeckUnits(readEclipseDeck(fullfile(pth, 'BENCH_SPE9.DATA')));
    G     = computeGeometry(initEclipseGrid(deck));
    rock  = compressRock(initEclipseRock(deck), G.cells.indexMap);
    fluid = initDeckADIFluid(deck);
    gravity reset on
    state = struct('s', [sw0, 1-sw0-sg0, sg0], 'rs', rs0, 'rv', rv0, ...
                   'pressure', deck.SOLUTION.PRESSURE);

`rs`/`rv` are the deck's RS/RV arrays, or 0 when the deck has none (SPE9
has RS and no RV). The SOLUTION arrays are used as read, without
`indexMap`: every SPE9 cell is active.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import local_dataset_path
from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import init_deck_adi_fluid
from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
from pymrst.modules.deckformat.grid import init_eclipse_grid
from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
from pymrst.modules.deckformat.params.rock.init_eclipse_rock import init_eclipse_rock
from pymrst.params.gravity.gravity import gravity


def setup_spe9() -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    """MRST `setupSPE9`: ``(G, rock, fluid, deck, state)`` for SPE9."""
    fn = local_dataset_path("spe9", "BENCH_SPE9.DATA")/"BENCH_SPE9.DATA"
    deck = convert_deck_units(read_eclipse_deck(fn))
    G = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), G.cells.indexMap)
    fluid = init_deck_adi_fluid(deck)
    gravity("reset", "on")
    solution = deck["SOLUTION"]
    p0 = _column(solution["PRESSURE"])
    sw0 = _column(solution["SWAT"])
    sg0 = _column(solution["SGAS"])
    s0 = np.column_stack([sw0, 1 - sw0 - sg0, sg0])
    rs0 = _column(solution["RS"]) if "RS" in solution else 0.0
    rv0 = _column(solution["RV"]) if "RV" in solution else 0.0
    state = {"s": s0, "rs": rs0, "rv": rv0, "pressure": p0}
    return G, rock, fluid, deck, state


def _column(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


setupSPE9 = setup_spe9

__all__ = ["setupSPE9", "setup_spe9"]


if __name__ == "__main__":
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.utils.units import DARCY, FT, MILLI, PSIA, STB

    with isolated_gravity():
        G, ROCK, FLUID, DECK, STATE = setup_spe9()
        GRAVITY_AFTER_SETUP = np.array(gravity(), dtype=float)

    def test_the_grid_is_the_published_24_by_25_by_15():
        assert tuple(DECK["RUNSPEC"]["cartDims"]) == (24, 25, 15)
        assert G.cells.num == 9000, G.cells.num
        print("ok test_the_grid_is_the_published_24_by_25_by_15")

    def test_the_state_is_mrsts_struct():
        assert set(STATE) == {"s", "rs", "rv", "pressure"}
        assert STATE["s"].shape == (9000, 3)
        np.testing.assert_allclose(STATE["s"].sum(axis=1), 1.0)
        assert STATE["rs"].shape == (9000,), "SPE9 has dissolved gas"
        assert STATE["rv"] == 0.0, "SPE9 has no RV, so rv0 = 0"
        print("ok test_the_state_is_mrsts_struct")

    def test_the_initial_solution_is_the_decks_in_si():
        """INITSOL.INC starts PRESSURE 3590.254639 3610.877930 (psia), SWAT
        0.165092 and RS 1.387076 (Mscf/stb)."""
        np.testing.assert_allclose(STATE["pressure"][:2]/PSIA, [3590.254639, 3610.877930])
        assert np.isclose(STATE["s"][0, 0], 0.165092)
        assert np.isclose(STATE["rs"][0], 1.387076*1000*FT**3/STB)
        print("ok test_the_initial_solution_is_the_decks_in_si")

    def test_the_permeability_is_converted_to_si():
        kx = np.asarray(ROCK.perm)[:, 0]/(MILLI*DARCY)
        assert 0.003 < kx.min() and kx.max() < 1.1e4, (kx.min(), kx.max())
        print("ok test_the_permeability_is_converted_to_si")

    def test_gravity_is_switched_on():
        """MRST's `gravity reset on` is a global side effect the case relies on."""
        assert np.isclose(np.linalg.norm(GRAVITY_AFTER_SETUP), 9.80665)
        print("ok test_gravity_is_switched_on")

    test_the_grid_is_the_published_24_by_25_by_15()
    test_the_state_is_mrsts_struct()
    test_the_initial_solution_is_the_decks_in_si()
    test_the_permeability_is_converted_to_si()
    test_gravity_is_switched_on()
    print("\nAll tests passed!")
