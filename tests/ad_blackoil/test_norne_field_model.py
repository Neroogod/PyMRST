import numpy as np
import pytest

from pymrst.modules.ad_blackoil.examples.field_model_norne_example import (
    setup_field_model_norne,
)
from pymrst.modules.deckformat.norne import (
    norne_field_directory, read_norne_field_deck,
)


def _available():
    try:
        norne_field_directory()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="official Norne archive absent")


def test_complete_norne_deck_schedule_fault_and_well_inventory():
    deck = read_norne_field_deck()
    assert len(deck.schedule.dates) == 247
    assert len(deck.schedule.wells) == 36
    assert deck.schedule.control_change_count == 4533
    assert len(deck.faults) == 977
    assert len(deck.fault_multipliers) == 60
    assert deck.schedule.dates[0] == np.datetime64("1997-11-14")
    assert deck.schedule.dates[-1] == np.datetime64("2006-12-01")


def test_norne_imported_model_initialization_and_regions():
    case = setup_field_model_norne()
    assert case.active_cells == 44431
    assert [part.grid.cells.num for part in case.components] == [44420, 11]
    assert len(case.outputs.opm.states) == 65
    assert case.initial_state.pressure.size == case.active_cells
    state = case.initial_blackoil_state()
    np.testing.assert_allclose(state["sW"]+state["sO"]+state["sG"], 1.0)
    assert set(state["status"]) == {1, 2, 3}
    assert all("SATNUM" in part.regions and "PVTNUM" in part.regions
               for part in case.components)
    definition = case.well_definition("B-1H")
    assert definition["WELSPECS"] and definition["COMPDAT"]


def test_norne_native_schedule_and_well_conversion(tmp_path):
    case = setup_field_model_norne()
    grid, rock, schedule = case.build_native_schedule()
    assert grid.cells.num == 44431
    assert rock.perm.shape == (44431, 3)
    assert rock.poro.shape == (44431,)
    assert len(schedule.step_val) == 247
    assert len(schedule.controls) == 247
    np.testing.assert_allclose(np.sum(schedule.step_val)/86400.0, 3312.0)
    for control in schedule.controls:
        assert len(control.W) == 36
        assert all(well.type in {"bhp", "rate", "orat", "wrat", "grat", "lrat",
                                 "vrat", "resv_history"}
                   for well in control.W)
    final = {well.name: well for well in schedule.controls[-1].W}
    assert len(final["C-4H"].cells) == 7
    assert len(final["B-2H"].cells) == 8
    assert len(final["D-1H"].cells) == 8
    injector, producer = case.build_native_vfp_tables()
    assert len(injector) == 20 and sum(table is not None for table in injector) == 11
    assert len(producer) == 48 and sum(table is not None for table in producer) == 24
    assert injector[9].bhp_table.shape == (14, 9)
    assert producer[0].bhp_table.shape == (18, 8, 6, 8)
    np.testing.assert_allclose(
        injector[9].evaluate_bhp(injector[9].flow[0], injector[9].thp[0]),
        139.37e5,
    )
    operators = case.build_reference_operators()
    assert operators.N.shape == (134622, 2)
    assert np.count_nonzero(operators.T) == 132531
    np.testing.assert_allclose(np.max(operators.T), 1.403368462456597e-9)
    model, initial, model_schedule = case.build_native_model()
    assert model.pv_ref.shape == (44431,)
    assert model.n_phases == 3 and len(model.wells) == 36
    assert model.operators.N.shape == (134622, 2)
    assert len(model_schedule.controls) == 247
    initialized = model.init_state_ad(initial)
    assert initialized["pressure"].val.shape == (44431,)
    assert initialized["bhp"].val.shape == (36,)
    checkpoint = case.save_native_checkpoint(
        tmp_path / "norne-period-0.npz", initial, 0
    )
    restored, completed = case.load_native_checkpoint(checkpoint)
    assert completed == 0
    for key in ("pressure", "sW", "sO", "sG", "rs", "rv", "status",
                "qWs", "qOs", "qGs", "bhp"):
        np.testing.assert_array_equal(restored[key], initial[key])


def test_norne_field_scale_reference_signature():
    signature = setup_field_model_norne().regression_signature()
    np.testing.assert_array_equal(signature["dimensions"], [44431, 2, 36, 247, 65])
    np.testing.assert_allclose(signature["states"], [
        [0, 27592890.01332968, 649178.0517648754, 0.6009614956071463, 0.07484685003640765],
        [141, 27257607.128585216, 3847766.445978753, 0.6580012712043841, 0.1060686599160825],
        [241, 30324153.52905683, 5866149.061853466, 0.7091273453176077, 0.08675091219273653],
    ], rtol=2e-12)
    np.testing.assert_allclose(signature["time"], [0, 1548, 3312])
    np.testing.assert_allclose(signature["rates"], [
        [8.63999983e-10, 2.52447500e4, 7.25554541e3],
        [8.63999983e-10, 3.61549414e3, 1.67914648e4],
        [8.63999983e-10, 5.74811950e6, 2.27945525e6],
        [0, 3.75179375e4, 2.18840078125e4],
        [0, 4.15303025e6, 2.08381140625e5],
    ], rtol=2e-7, atol=1e-8)
