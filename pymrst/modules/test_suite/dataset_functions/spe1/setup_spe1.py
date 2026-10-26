"""The SPE1 case (MRST ``test-suite/dataset-functions/spe1/setupSPE1.m``).

MRST's, line for line::

    deck  = convertDeckUnits(readEclipseDeck(fullfile(pth, 'BENCH_SPE1.DATA')));
    G     = computeGeometry(initEclipseGrid(deck));
    rock  = compressRock(initEclipseRock(deck), G.cells.indexMap);
    fluid = initDeckADIFluid(deck);
    gravity reset on
    p0    = [329.7832774859256; 330.2313357125603; 330.9483500720813];  % per layer
    state = struct('s', repmat([0.12, 0.88, 0.0], [G.cells.num, 1]), ...
                   'rs', repmat(226.1966570852417, [G.cells.num, 1]), ...
                   'rv', 0, 'pressure', convertFrom(p0(k), barsa));

The three pressures and the solution gas-oil ratio are **not computed here**;
they are the equilibrated values MRST wrote into the file to fifteen digits, so
that every case built on SPE1 starts from bit-identical numbers rather than
from whatever the equilibration happens to produce. They are transcribed for
the same reason.

Two details are easy to lose:

* **the pressures are per layer, not per cell.** `k` is the layer index, and
  `p0(k)` broadcasts one of three values onto each cell. The layers differ by
  about a bar, so a single value is not visibly wrong - it is just not SPE1;
* **`rv = 0` means dry gas**, not "vaporised oil is zero everywhere by
  coincidence". SPE1 has no vaporised oil at all, and a model built with
  `vapoil` on would carry an unknown the case does not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_props.props.init_deck_adi_fluid_dispatch import (
    init_deck_adi_fluid,
)
from pymrst.modules.deckformat.deckinput.convert_deck_units import (
    convert_deck_units,
)
from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
    read_eclipse_deck,
)
from pymrst.modules.deckformat.grid import init_eclipse_grid
from pymrst.modules.deckformat.params.rock.compress_rock import compress_rock
from pymrst.modules.deckformat.params.rock.init_eclipse_rock import (
    init_eclipse_rock,
)
from pymrst.params.gravity.gravity import gravity
from pymrst.utils.units import BARSA

#: MRST's `gravity reset on`.
NORM_GRAVITY = 9.80665

#: The equilibrated layer pressures, in bar, transcribed from `setupSPE1.m`.
LAYER_PRESSURE_BAR = (329.7832774859256,     # top
                      330.2313357125603,     # middle
                      330.9483500720813)     # bottom

#: The equilibrated solution gas-oil ratio, likewise transcribed.
RS0 = 226.1966570852417

#: `[sW, sO, sG]` at t = 0.
S0 = (0.12, 0.88, 0.0)


def spe1_data_directory(root=None) -> Path:
    """Where `BENCH_SPE1.DATA` lives.

    MRST reads `getDatasetPath('spe1')`, which looks under
    `mrstDataDirectory()`. The SPE1 data ships inside the MRST tree in this
    workspace, so that location is tried first and the bundled copy second -
    the same rule `egg_data_directory` follows.
    """
    candidates = []
    if root is not None:
        candidates.append(Path(root))
    else:
        try:
            from pymrst.utils.dataset_manager.get_dataset_path import (
                get_dataset_path,
            )
            candidates.append(Path(get_dataset_path(
                "spe1", download=False, skip_available_check=True)))
        except Exception:
            pass
        here = Path(__file__).resolve().parents[5]
        candidates.append(here.parent/"mrst-2024a-Geomech"/"examples"/"data"/"SPE1")
    for candidate in candidates:
        if (candidate/"BENCH_SPE1.DATA").is_file():
            return candidate
    raise FileNotFoundError(
        "BENCH_SPE1.DATA was not found. Looked in "
        + ", ".join(str(path) for path in candidates))


def setup_spe1(*, data_root=None) -> Tuple[Any, Any, Any, Dict[str, Any], Any]:
    """MRST `setupSPE1`: ``(G, rock, fluid, deck, state)`` for SPE1, with
    `state = struct('s', s0, 'rs', rs0, 'rv', 0, 'pressure', p0)`."""
    path = spe1_data_directory(data_root)/"BENCH_SPE1.DATA"
    deck = convert_deck_units(read_eclipse_deck(path))
    G = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), G.cells.indexMap)
    fluid = init_deck_adi_fluid(deck)
    gravity("reset", "on")
    # `[k, k] = ind2sub([prod(cartDims(1:2)), cartDims(3)], indexMap)`: the
    # layer of each active cell (0-based here).
    dims = np.asarray(deck["RUNSPEC"]["cartDims"], dtype=np.int64)
    k = np.asarray(G.cells.indexMap, dtype=np.int64).ravel()//int(dims[0]*dims[1])
    p0 = np.asarray(LAYER_PRESSURE_BAR, dtype=float)[k]*BARSA
    s0 = np.tile(S0, (G.cells.num, 1))
    rs0 = np.full(G.cells.num, RS0)
    rv0 = 0.0                            # dry gas
    state = {"s": s0, "rs": rs0, "rv": rv0, "pressure": p0}
    return G, rock, fluid, deck, state


def setup_spe1_ad(*, data_root=None) -> Tuple[Any, Any, Any, Dict[str, Any], Any]:
    """`setup_spe1` with the state in the legacy models' layout:
    `sW`/`sO`/`sG` and a `status` flag instead of MRST's `s` matrix."""
    from pymrst.modules.ad_core import State

    path = spe1_data_directory(data_root)/"BENCH_SPE1.DATA"
    deck = convert_deck_units(read_eclipse_deck(path))

    grid = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), grid.cells.indexMap)
    fluid = init_deck_adi_fluid(deck)
    # MRST's `gravity reset on`: a global side effect every SPE1 case relies
    # on. Models built after this call (MRST's, and the AD-OO ports) read
    # gravity from the global setting; leaving it off silently removes gravity
    # from the reservoir flux, the boundary conditions and the wellbore drop.
    gravity("reset", "on")

    # `ind2sub([prod(cartDims(1:2)), cartDims(3)], indexMap)` - the layer each
    # active cell sits in. MRST is 1-based; the index below is 0-based, which
    # is what `LAYER_PRESSURE_BAR` is indexed with.
    dims = np.asarray(deck["RUNSPEC"]["cartDims"], dtype=np.int64)
    per_layer = int(dims[0]*dims[1])
    index_map = np.asarray(grid.cells.indexMap, dtype=np.int64).ravel()
    layer = index_map // per_layer

    pressure = np.asarray(LAYER_PRESSURE_BAR, dtype=float)[layer]*BARSA
    n = int(grid.cells.num)
    sw = np.full(n, S0[0])
    so = np.full(n, S0[1])
    sg = np.full(n, S0[2])
    state = State(pressure=pressure, sW=sw, sO=so, sG=sg,
                  rs=np.full(n, RS0),
                  rv=np.zeros(n))          # dry gas
    # MRST's models derive the phase-appearance flag the first time they need
    # it; PyMRST's read it off the state, so it is set here from the same
    # saturations rather than left for the first assembly to invent.
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        blackoil_status,
    )
    state["status"] = blackoil_status(sw, so, sg, disgas=True, vapoil=False)
    return grid, rock, fluid, deck, state


setupSPE1 = setup_spe1

__all__ = ["LAYER_PRESSURE_BAR", "NORM_GRAVITY", "RS0", "S0", "setupSPE1",
           "setup_spe1", "setup_spe1_ad", "spe1_data_directory"]


if __name__ == "__main__":
    from pymrst.utils.units import DARCY, MILLI

    try:
        spe1_data_directory()
    except FileNotFoundError:
        print("skip (SPE1 dataset not installed)")
        raise SystemExit(0)

    G, ROCK, FLUID, DECK, STATE = setup_spe1()

    def test_the_grid_is_the_published_ten_by_ten_by_three():
        assert tuple(DECK["RUNSPEC"]["cartDims"]) == (10, 10, 3)
        assert G.cells.num == 300, G.cells.num
        print("ok test_the_grid_is_the_published_ten_by_ten_by_three")

    def test_the_permeability_is_the_three_published_layers():
        kx = np.asarray(ROCK.perm)[:, 0]/(MILLI*DARCY)
        assert sorted(set(np.round(kx, 6))) == [50.0, 200.0, 500.0], set(kx)
        print("ok test_the_permeability_is_the_three_published_layers")

    def test_the_deck_is_read_in_field_units_and_converted():
        """SPE1 is a FIELD deck. Unconverted, 500 mD reads as 500 m2."""
        assert DECK["RUNSPEC"].get("SI") is True, DECK["RUNSPEC"]
        assert np.isclose(np.max(np.asarray(ROCK.perm)[:, 0])/(MILLI*DARCY),
                          500.0)
        print("ok test_the_deck_is_read_in_field_units_and_converted")

    def test_the_pressures_are_per_layer_not_per_cell():
        """`p0(k)` broadcasts three values onto the layers. One value for the
        whole field is about a bar out at the bottom - not visibly wrong, and
        not SPE1."""
        p = np.asarray(STATE["pressure"])/BARSA
        assert sorted(set(np.round(p, 10))) == sorted(
            np.round(LAYER_PRESSURE_BAR, 10)), set(np.round(p, 6))
        print("ok test_the_pressures_are_per_layer_not_per_cell")

    def test_each_layer_gets_its_own_pressure_in_depth_order():
        z = np.asarray(G.cells.centroids)[:, 2]
        p = np.asarray(STATE["pressure"])
        order = np.argsort(z)
        assert p[order[0]] < p[order[-1]], "the deeper layer must be heavier"
        print("ok test_each_layer_gets_its_own_pressure_in_depth_order")

    def test_the_saturations_and_ratio_are_the_transcribed_ones():
        assert np.array_equal(np.asarray(STATE["s"]).shape, (300, 3))
        assert np.allclose(STATE["s"], [0.12, 0.88, 0.0])
        assert np.allclose(STATE["rs"], RS0)
        print("ok test_the_saturations_and_ratio_are_the_transcribed_ones")

    def test_the_state_is_mrsts_struct():
        """MRST's fields, and nothing else: the models derive `status`."""
        assert sorted(STATE) == ["pressure", "rs", "rv", "s"]
        print("ok test_the_state_is_mrsts_struct")

    def test_the_legacy_state_agrees_with_the_saturations():
        """The legacy models read the phase-appearance flag off the state; a
        `status` that disagrees with the saturations is a contradiction the
        solver resolves by discarding one of them."""
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            blackoil_status,
        )
        legacy = setup_spe1_ad()[4]
        assert np.allclose(legacy["sW"], STATE["s"][:, 0])
        assert np.allclose(legacy["pressure"], STATE["pressure"])
        expected = blackoil_status(np.full(300, S0[0]), np.full(300, S0[1]),
                                   np.full(300, S0[2]), disgas=True,
                                   vapoil=False)
        assert np.array_equal(np.asarray(legacy["status"]), expected)
        print("ok test_the_legacy_state_agrees_with_the_saturations")

    def test_the_gas_is_dry():
        """`rv = 0` in MRST is a statement about the fluid, not a coincidence:
        SPE1 has no vaporised oil, and a `vapoil` model would carry an unknown
        the case does not have."""
        assert np.allclose(STATE["rv"], 0.0)
        assert "rvSat" not in FLUID, sorted(FLUID)
        print("ok test_the_gas_is_dry")

    def test_the_fluid_is_live_oil():
        assert "rsSat" in FLUID, sorted(FLUID)
        assert DECK["RUNSPEC"].get("DISGAS") is True
        print("ok test_the_fluid_is_live_oil")

    def test_the_schedule_is_the_decks():
        schedule = DECK["SCHEDULE"]
        assert len(schedule.step["val"]) == 120, len(schedule.step["val"])
        assert len(schedule.control) >= 1
        print("ok test_the_schedule_is_the_decks")

    test_the_grid_is_the_published_ten_by_ten_by_three()
    test_the_permeability_is_the_three_published_layers()
    test_the_deck_is_read_in_field_units_and_converted()
    test_the_pressures_are_per_layer_not_per_cell()
    test_each_layer_gets_its_own_pressure_in_depth_order()
    test_the_saturations_and_ratio_are_the_transcribed_ones()
    test_the_state_is_mrsts_struct()
    test_the_legacy_state_agrees_with_the_saturations()
    test_the_gas_is_dry()
    test_the_fluid_is_live_oil()
    test_the_schedule_is_the_decks()
    print("\nAll tests passed!")
