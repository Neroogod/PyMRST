"""MRST ``ad-blackoil/examples/utils/setupAquifertest.m``.

In MRST 2024a this function cannot finish: it reads the deck, builds the grid,
rock and fluid and processes the aquifers, then calls `AquiferBlackOilModel`,
which no longer exists anywhere in MRST ("Undefined function
'AquiferBlackOilModel' for input arguments of type 'struct'"). Nothing in
MRST calls it; MRST's `aquifertest` example sets the cases up with
`initEclipseProblemAD`. The port does what MRST does, and stops where MRST
stops, with the same message.

`setup_aquifertest_ad` is PyMRST's earlier, working replacement (a legacy
`GenericBlackOilModelAD` with the `AQUFETP`/`AQUANCON` aquifers attached),
kept for `aquifertest.py`. Its notes follow.

Port of MRST ``ad-blackoil/examples/utils/setupAquifertest.m``.

MRST's, line for line::

    deck   = convertDeckUnits(readEclipseDeck(fn));
    G      = computeGeometry(initEclipseGrid(deck));
    rock   = compressRock(initEclipseRock(deck), G.cells.indexMap);
    fluid  = initDeckADIFluid(deck, 'G', G);
    output = processAquifer(deck, G);
    model  = AquiferBlackOilModel(G, rock, fluid, output.aquifers, ...
                                  output.aquind, output.aquiferprops, ...
                                  'modeltype', 'oilwater');

It takes **a deck file name**, and that is the whole difference from the
version this replaces: that one built a synthetic 10x1x2 Cartesian case with a
hand-written aquifer, where MRST reads one of the two published
`2D_OILWATER_AQ_ECLIPSE_*.DATA` decks - a 20x1x5 corner-point model with two
Fetkovich aquifers attached to opposite faces of the box, and an ECLIPSE result
alongside it to compare against. A synthetic case has nothing to compare to,
which is what the example is for.

The aquifer wiring is the part worth stating. `AQUFETP` gives each aquifer its
productivity index, compressibility, initial pressure and volume; `AQUANCON`
says which *cells* it attaches to and through which face. Reading only the
first gives an aquifer with no connections - which contributes no influx, runs
perfectly, and answers as though the aquifer were not there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections, AquiferModelAD,
)
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst_ad,
)
from pymrst.modules.ad_core.initialization.deck import init_state_deck_ad
from pymrst.modules.ad_props.props.deck_fluid import deck_fluid_object
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
from pymrst.modules.deckformat.params.rock.process_aquifer import (
    process_aquifer,
)
from pymrst.utils.units import BARSA

#: MRST's `gravity on`.
NORM_GRAVITY = 9.80665

#: The two published cases. `_1` has the aquifers at their initial pressure,
#: `_2` starts them out of equilibrium.
CASES = ("2D_OILWATER_AQ_ECLIPSE_1", "2D_OILWATER_AQ_ECLIPSE_2")


def aquifertest_data_directory(root=None) -> Path:
    """Where the published `2D_OILWATER_AQ_ECLIPSE_*` files live."""
    candidates = []
    if root is not None:
        candidates.append(Path(root))
    else:
        try:
            from pymrst.utils.dataset_manager.get_dataset_path import (
                get_dataset_path,
            )
            candidates.append(Path(get_dataset_path(
                "aquifertest", download=False, skip_available_check=True)))
        except Exception:
            pass
        here = Path(__file__).resolve().parents[5]
        candidates.append(here.parent/"mrst-2024a-Geomech"/"examples"/"data"
                          / "aquifertest")
    for candidate in candidates:
        if (candidate/f"{CASES[0]}.DATA").is_file():
            return candidate
    raise FileNotFoundError(
        "the aquifertest dataset is not installed. Looked in "
        + ", ".join(str(path) for path in candidates)
        + ". MRST downloads it from "
        "https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129"
        "/aquifertestdata.zip")


def setup_aquifertest(fn: Any) -> Tuple[Any, Any, Any]:
    """MRST's `[model, initState, schedule] = setupAquifertest(fn)`: raises
    where MRST 2024a does, after the same steps."""
    deck = read_eclipse_deck(Path(fn))
    deck = convert_deck_units(deck)
    G = init_eclipse_grid(deck)
    G = compute_geometry(G)
    rock = init_eclipse_rock(deck)
    rock = compress_rock(rock, G.cells.indexMap)
    init_deck_adi_fluid(deck, grid=G)
    process_aquifer(deck, G)
    raise NameError("Undefined function 'AquiferBlackOilModel' for input arguments of "
                    "type 'struct'. (MRST 2024a has no AquiferBlackOilModel.)")


def setup_aquifertest_ad(fn=None, *, case: int = 1, data_root=None):
    """Return ``(model, init_state, schedule)`` for one published case."""
    if fn is None:
        fn = aquifertest_data_directory(data_root)/f"{CASES[case - 1]}.DATA"
    deck = convert_deck_units(read_eclipse_deck(Path(fn)))

    grid = compute_geometry(init_eclipse_grid(deck))
    rock = compress_rock(init_eclipse_rock(deck), grid.cells.indexMap)
    fluid = deck_fluid_object(init_deck_adi_fluid(deck))

    model = GenericBlackOilModelAD(grid, rock, fluid, [],
                                 gravity=NORM_GRAVITY)
    model.input_data = deck
    model.aquifer = aquifer_from_deck(deck, grid)

    schedule = convert_deck_schedule_to_mrst_ad(
        model, {"step": deck["SCHEDULE"].step,
                "control": deck["SCHEDULE"].control})
    if schedule.controls and schedule.controls[0].W:
        model.facility.wells[:] = list(schedule.controls[0].W)

    state = init_state_deck_ad(model, deck, gravity=NORM_GRAVITY)
    model.aquifer.init_state(state)
    _seed_well_unknowns(model, state, model.facility.wells)
    return model, state, schedule


setupAquifertest = setup_aquifertest


def aquifer_from_deck(deck: Any, grid: Any) -> AquiferModelAD:
    """The `AQUFETP`/`AQUANCON` aquifers, as an `AquiferModel`."""
    output = process_aquifer(deck, grid)
    table = np.atleast_2d(np.asarray(output.aquifers, dtype=float))
    index = output.aquind
    if table.size == 0:
        raise ValueError(
            "the deck declares no aquifer connections; AQUANCON is what "
            "attaches an AQUFETP aquifer to the grid, and without it the "
            "aquifer contributes no influx at all")

    aquid = table[:, index.aquid].astype(np.int64)
    connections = AquiferConnections(
        aquid=aquid,
        cells=table[:, index.conn].astype(np.int64),
        j=table[:, index.J],
        c=table[:, index.C],
        alpha=table[:, index.alpha],
        depth_conn=table[:, index.depthconn],
        depth_aq=table[:, index.depthaq],
        c_aq=np.asarray(output.aquiferprops.C, dtype=float),
    )
    return AquiferModelAD(connections,
                        np.asarray(output.initval["pressures"], dtype=float),
                        np.asarray(output.initval["volumes"], dtype=float))


def _seed_well_unknowns(model: Any, state: Any, wells) -> None:
    """The well primary variables the black-oil models read off the state."""
    wells = list(wells or ())
    phases = [str(name).upper()[0] for name in model.get_phase_names()]
    n = len(wells)
    p_mean = float(np.mean(np.asarray(state["pressure"], dtype=float)))
    rates = {f"q{phase}s": np.zeros(n) for phase in phases}
    bhp = np.zeros(n)
    for position, well in enumerate(wells):
        composition = np.atleast_1d(np.asarray(well.compi, dtype=float))
        if str(well.type).lower() == "bhp":
            bhp[position] = float(well.val)
        else:
            bhp[position] = p_mean + 10*BARSA*int(well.sign or 1)
            for k, phase in enumerate(phases):
                if k < composition.size:
                    rates[f"q{phase}s"][position] = float(well.val)*composition[k]
    for name, values in rates.items():
        state[name] = values
    state["bhp"] = bhp


__all__ = ["CASES", "NORM_GRAVITY", "aquifer_from_deck",
           "aquifertest_data_directory", "setupAquifertest", "setup_aquifertest",
           "setup_aquifertest_ad"]


if __name__ == "__main__":
    try:
        aquifertest_data_directory()
    except FileNotFoundError as missing:
        print(f"skip ({missing})")
        raise SystemExit(0)

    def test_it_stops_where_mrst_2024a_stops():
        fn = aquifertest_data_directory()/f"{CASES[0]}.DATA"
        try:
            setup_aquifertest(fn)
        except NameError as undefined:
            assert "AquiferBlackOilModel" in str(undefined)
            print("ok test_it_stops_where_mrst_2024a_stops")
            return
        raise AssertionError("MRST 2024a cannot build an AquiferBlackOilModel")

    test_it_stops_where_mrst_2024a_stops()
    MODEL, STATE, SCHEDULE = setup_aquifertest_ad()

    def test_the_grid_is_the_published_twenty_by_one_by_five():
        """A corner-point deck, not the synthetic 10x1x2 Cartesian box the
        previous version built."""
        assert tuple(MODEL.G.cartDims) == (20, 1, 5), MODEL.G.cartDims
        assert MODEL.G.cells.num == 96, MODEL.G.cells.num
        print("ok test_the_grid_is_the_published_twenty_by_one_by_five")

    def test_it_is_an_oil_water_case():
        assert MODEL.n_phases == 2, MODEL.n_phases
        assert MODEL.disgas is False and MODEL.vapoil is False
        print("ok test_it_is_an_oil_water_case")

    def test_there_are_two_aquifers():
        """`AQUFETP` declares two; `AQUANCON` attaches them to the I- and I+
        faces, at opposite ends of the model."""
        conn = MODEL.aquifer.connections
        assert conn.n_aq == 2, conn.n_aq
        assert conn.n_conn == 10, conn.n_conn
        assert set(np.unique(conn.aquid).tolist()) == {0, 1}
        print(f"ok test_there_are_two_aquifers ({conn.n_conn} connections)")

    def test_the_aquifers_attach_to_opposite_ends():
        """Reading `AQUFETP` alone gives an aquifer with no connections: it
        contributes no influx, runs perfectly, and answers as though it were
        not there."""
        conn = MODEL.aquifer.connections
        cells = np.asarray(conn.cells)
        i = np.asarray(MODEL.G.cells.indexMap)[cells] % 20
        first = i[np.asarray(conn.aquid) == 0]
        second = i[np.asarray(conn.aquid) == 1]
        assert first.max() < second.min(), (first, second)
        print("ok test_the_aquifers_attach_to_opposite_ends")

    def test_the_aquifer_properties_are_the_decks():
        """`AQUFETP`: J = 100, C = 3e-5/bar, p = 300 bar, V = 318000 m3."""
        aquifer = MODEL.aquifer
        assert np.allclose(aquifer.pressures/BARSA, 300.0), aquifer.pressures
        assert np.allclose(aquifer.volumes, 318000.0), aquifer.volumes
        assert np.allclose(aquifer.connections.c_aq*BARSA, 3e-5)
        print("ok test_the_aquifer_properties_are_the_decks")

    def test_the_state_carries_the_aquifer():
        """`aquifer_p`/`aquifer_V` ride on the reservoir state so they survive
        Newton iterations and timestep cuts."""
        assert "aquifer_p" in STATE and "aquifer_V" in STATE
        assert np.allclose(STATE["aquifer_p"]/BARSA, 300.0)
        print("ok test_the_state_carries_the_aquifer")

    def test_the_schedule_is_the_decks():
        from pymrst.utils.units import DAY
        assert SCHEDULE.n_steps == 21, SCHEDULE.n_steps
        assert np.isclose(float(np.sum(SCHEDULE.step_val))/DAY, 1101.0)
        assert len(SCHEDULE.controls[0].W) == 2
        print("ok test_the_schedule_is_the_decks")

    def test_the_second_case_differs_from_the_first():
        """Two published cases, and they are not the same run."""
        _model2, state2, _schedule2 = setup_aquifertest_ad(case=2)
        assert not np.allclose(np.asarray(state2["pressure"]),
                               np.asarray(STATE["pressure"]))
        print("ok test_the_second_case_differs_from_the_first")

    test_the_grid_is_the_published_twenty_by_one_by_five()
    test_it_is_an_oil_water_case()
    test_there_are_two_aquifers()
    test_the_aquifers_attach_to_opposite_ends()
    test_the_aquifer_properties_are_the_decks()
    test_the_state_carries_the_aquifer()
    test_the_schedule_is_the_decks()
    test_the_second_case_differs_from_the_first()
    print("\nAll tests passed!")
