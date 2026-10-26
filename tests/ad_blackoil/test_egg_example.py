"""The Egg benchmark, against MRST's own reference.

`eggExample.m` is now ported literally - `setupEGG` reads the deck with the
general reader, `initEclipseProblemAD` equilibrates from `EQUIL`, and the
schedule is the deck's own 123 steps. The previous port read the deck with a
bespoke Egg-only parser, built the grid from `cart_grid` with hand-written
cell sizes, and initialised pressure from a formula of its own; those tests
checked that dictionary rather than the deck.

`ref_ad_blackoil_egg.mat` holds MRST's grid, rock, initial state and the state
after the first report step. The setup quantities are checked to machine
precision, because they are the *same numbers* read a different way and
nothing in between is iterative. The post-step quantities are checked to the
tolerance the two solvers' well models actually agree to, which is measured
rather than assumed.
"""

import numpy as np
import pytest
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.egg import run_egg_example
from pymrst.modules.test_suite.dataset_functions.egg import (
    get_deck_egg, setup_egg,
)
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

REFERENCE = "examples/validation/ref_ad_blackoil_egg.mat"


def _dataset_available() -> bool:
    from pymrst.modules.deckformat.egg.egg_dataset import egg_data_directory
    try:
        egg_data_directory()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _dataset_available(),
                                reason="the Egg dataset is not installed")


@pytest.fixture(scope="module")
def reference():
    return loadmat(REFERENCE, squeeze_me=True)


@pytest.fixture(scope="module")
def case():
    return setup_egg()


@pytest.fixture(scope="module")
def first_step():
    return run_egg_example(1, verbose=False)


def _relative(actual, expected) -> float:
    """Worst relative difference, floored at the *quantity's own* scale.

    An entry that is exactly zero in the reference has no relative error, and
    dividing by a fixed `1e-30` invents one. MRST writes a shut producer's rate
    as `0.0`; PyMRST leaves it at about `1e-34` m3/s - denormal noise, and
    physically the same statement - but `1e-34 / 1e-30` reads as a 1e-4
    disagreement. That is the mirror of the `allclose` atol trap: there a fixed
    floor was too *large* for SI permeabilities, here it is far too small for
    surface rates.

    The floor is `1e-12` of the reference's own largest magnitude, so it always
    means "negligible for this quantity" rather than "negligible in the
    abstract".
    """
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    assert actual.shape == expected.shape
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    floor = scale*1e-12 if scale > 0.0 else 1e-30
    return float(np.max(np.abs(actual - expected)
                        / np.maximum(np.abs(expected), floor)))


# -- the deck ----------------------------------------------------------------

def test_the_deck_reads_as_a_deck_not_a_dictionary():
    """RUNSPEC, GRID, PROPS, SOLUTION and SCHEDULE - the sections MRST's
    readers fill, and the four functions of `setupEGG` all take a deck."""
    deck = get_deck_egg()
    assert tuple(deck["RUNSPEC"]["cartDims"]) == (60, 60, 7)
    assert deck["RUNSPEC"]["START"].isoformat() == "2011-06-15"
    assert set(deck["GRID"]) >= {"ACTNUM", "PORO", "PERMX", "PERMZ", "TOPS"}
    assert deck["SOLUTION"]["EQUIL"].shape == (1, 11)
    assert len(deck["SCHEDULE"].step["val"]) == 123


def test_the_schedule_is_one_control_period_of_ten_years():
    schedule = get_deck_egg()["SCHEDULE"]
    assert len(schedule.control) == 1
    assert np.isclose(sum(schedule.step["val"])/DAY, 3600.0)


def test_a_realization_replaces_the_permeability():
    base = np.asarray(get_deck_egg()["GRID"]["PERMX"], float)/(MILLI*DARCY)
    drawn = np.asarray(get_deck_egg(1)["GRID"]["PERMX"], float)/(MILLI*DARCY)
    assert drawn.shape == base.shape
    # In millidarcy: an SI permeability is of order 1e-13 and `allclose`'s
    # default `atol` is 1e-8, so any two fields compare equal in SI.
    assert not np.allclose(drawn, base)


# -- the grid, rock and initial state, against MRST --------------------------

def test_the_grid_matches_mrsts(case, reference):
    G = case[0]
    assert G.cells.num == int(reference["nc"])
    # MRST's indexMap is 1-based.
    assert np.array_equal(np.asarray(G.cells.indexMap) + 1,
                          np.asarray(reference["indexMap"]))
    assert np.allclose(np.asarray(G.cells.centroids),
                       np.asarray(reference["centroids"], dtype=float))


def test_the_rock_matches_mrsts(case, reference):
    rock = case[1]
    assert _relative(rock.perm, reference["perm"]) < 1e-12
    assert np.allclose(np.asarray(rock.poro).ravel(), reference["poro"])


def test_the_equilibrated_pressure_matches_mrsts(first_step, reference):
    """To machine precision: both integrate the same hydrostatic column from
    the same `EQUIL` record, and nothing in between iterates."""
    assert _relative(first_step.state0["pressure"],
                     reference["pressure0"]) < 1e-12


def test_the_initial_saturation_is_connate_water(first_step, reference):
    """The water-oil contact is at 5000 m, below a reservoir ending at
    4028 m, so the whole field starts at `swco`."""
    assert np.allclose(np.asarray(first_step.state0["sW"]),
                       np.asarray(reference["sw0"]))
    assert np.allclose(np.asarray(first_step.state0["sW"]), 0.1)


# -- the wells ---------------------------------------------------------------

def test_the_wells_match_mrsts(first_step, reference):
    wells = list(first_step.schedule.controls[0].W)
    assert [w.name for w in wells] == list(reference["well_names"])
    assert [int(w.sign) for w in wells] == list(
        np.asarray(reference["well_sign"], dtype=int))
    assert [len(np.atleast_1d(w.cells)) for w in wells] == list(
        np.asarray(reference["well_cells"], dtype=int))


def test_the_well_targets_match_mrsts(first_step, reference):
    """Injectors at 79.5 m3/day, producers at 395 bar - the `WCONINJE` and
    `WCONPROD` records, through the unit conversion."""
    values = np.asarray([w.val for w in first_step.schedule.controls[0].W])
    expected = np.asarray(reference["well_val"], dtype=float)
    assert np.allclose(values[:8], expected[:8], rtol=1e-12)
    assert np.allclose(values[8:], 395*BARSA)


def test_the_step_lengths_match_mrsts(first_step, reference):
    full = get_deck_egg()["SCHEDULE"].step
    assert np.allclose(np.asarray(full["val"], dtype=float),
                       np.asarray(reference["step_val"], dtype=float))
    assert np.array_equal(np.asarray(full["control"], dtype=int),
                          np.asarray(reference["step_control"], dtype=int))


# -- the first report step ---------------------------------------------------

def test_the_first_step_converges(first_step):
    assert all(step.converged for step in first_step.report.steps)


def test_the_pressure_after_a_day_matches_mrsts(first_step, reference):
    assert _relative(first_step.states[0]["pressure"],
                     reference["pressure1"]) < 1e-3


def test_the_injection_rates_match_mrsts(first_step, reference):
    """The injectors are rate-controlled and away from their pressure limit,
    so this is the target itself and the agreement is exact.

    The producers are shut in this step: MRST writes their rate as `0.0` and
    PyMRST leaves denormal noise around 1e-34 m3/s. `_relative` floors at the
    quantity's own scale so that reads as agreement, which it is.
    """
    assert _relative(first_step.states[0]["qWs"], reference["qWs1"]) < 1e-9
    injectors = np.asarray(first_step.states[0]["qWs"], dtype=float)[:8]
    assert np.allclose(injectors, np.asarray(reference["qWs1"], float)[:8],
                       rtol=1e-12, atol=0.0), "the injectors must match exactly"


def test_the_bottom_hole_pressures_match_mrsts(first_step, reference):
    assert _relative(first_step.states[0]["bhp"], reference["bhp1"]) < 1e-3


def test_the_production_rates_match_mrsts(first_step, reference):
    """The loosest of the comparisons, and the one that measures the two well
    models against each other: a producer's rate is the drawdown times a
    mobility, so it carries every small difference in the pressure solution.
    Measured at 0.7%.
    """
    actual = np.asarray(first_step.states[0]["qOs"], dtype=float)[8:]
    expected = np.asarray(reference["qOs1"], dtype=float)[8:]
    assert np.all(actual < 0.0), "producers must produce"
    assert np.max(np.abs(actual - expected)/np.abs(expected)) < 0.012


def test_the_injectors_produce_no_oil(first_step):
    assert np.allclose(np.asarray(first_step.states[0]["qOs"])[:8], 0.0,
                       atol=1e-15)


# -- the example's own outputs -----------------------------------------------

def test_the_example_returns_mrsts_four_curves(first_step):
    """`eggExample` plots injector BHP, producer oil rate, producer water rate
    and water cut. Those are the four arrays."""
    assert first_step.bhp.shape == (1, 8)
    assert first_step.orat.shape == (1, 4)
    assert first_step.wrat.shape == (1, 4)
    assert first_step.water_cut.shape == (1, 4)
    assert first_step.injectors == [f"INJECT{i}" for i in range(1, 9)]
    assert first_step.producers == [f"PROD{i}" for i in range(1, 5)]


def test_the_water_cut_starts_at_zero(first_step):
    assert np.allclose(first_step.water_cut, 0.0)


def test_the_well_output_is_the_solved_state_not_the_seed(first_step):
    """PyMRST keeps the well unknowns on the state; the `wellSol` rows a state
    carries are `init_well_sol_ad`'s seeds and the solve does not update them.
    Reading those would give every producer a rate of zero."""
    rows = first_step.well_sols[0]
    assert np.allclose([row["bhp"] for row in rows],
                       np.asarray(first_step.states[0]["bhp"]))
    assert all(row["qOs"] < 0.0 for row in rows if row["sign"] < 0)


@pytest.mark.slow
def test_the_full_schedule_runs(reference):
    """All 123 report steps - ten years of the published case."""
    result = run_egg_example(verbose=False)
    assert len(result.states) == 123
    assert all(step.converged for step in result.report.steps)
    assert np.isclose(result.T[-1], 3600.0*DAY/(365.2425*DAY), rtol=1e-3)
    # Water breaks through: the cut must rise from zero over ten years.
    assert np.allclose(result.water_cut[0], 0.0)
    assert result.water_cut[-1].max() > 0.1, result.water_cut[-1]
