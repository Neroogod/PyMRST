from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.models.aquifers.select_aquifer_from_deck import (
    aquifer_model_from_processed,
    select_aquifer_from_deck_ad,
)
from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.utils.cfl import estimate_composition_cfl_ad
from pymrst.modules.ad_core.utils.schedule import Control, Schedule, subset_schedule
from pymrst.modules.ad_core.utils.standalone_solve import standalone_solve
from pymrst.modules.deckformat.params.rock.process_aquifer import (
    AquiferDeckOutput,
    AquiferIndices,
    AquiferProps,
)


def test_physical_model_protocol_resolves_state_and_rejects_abstract_steps():
    model = PhysicalModel()
    state = {"pressure": np.array([10.0])}
    assert model.get_props(state, "pressure") is state["pressure"]
    for method, args in (
        (model.get_equations, (state, state, 1.0)),
        (model.update_state, (state, np.ones(1))),
        (model.get_convergence, ([], 1.0)),
    ):
        try:
            method(*args)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("PhysicalModel protocol method must be abstract")


def test_select_aquifer_from_deck_guards_and_processed_mapping():
    assert select_aquifer_from_deck_ad({}, SimpleNamespace(G=None)) is None
    assert select_aquifer_from_deck_ad(
        {"SOLUTION": {"AQUANCON": []}}, SimpleNamespace(G=None)
    ) is None

    rows = np.array([[0, 3, 0, 2.0, 0.1, 1.0, 1200.0, 1100.0]])
    processed = AquiferDeckOutput(
        aquifers=rows,
        aquind=AquiferIndices(),
        initval={"pressures": np.array([2.0e7]), "volumes": np.array([4.0e5])},
        aquiferprops=AquiferProps(
            depthaq=np.array([1100.0]), C=np.array([0.1]),
            J=np.array([2.0]), pvttbl=np.array([0]),
        ),
    )
    aquifer = aquifer_model_from_processed(processed)
    assert aquifer.connections.cells.tolist() == [3]
    assert np.allclose(aquifer.pressures, [2.0e7])
    assert np.allclose(aquifer.volumes, [4.0e5])


def test_filter_schedule_removes_dead_controls_and_renumbers():
    """MRST filterSchedule drops unused controls and remaps survivors."""
    schedule = Schedule(
        step_val=np.array([1.0, 2.0, 3.0, 4.0]),
        step_control=np.array([0, 1, 2, 1]),
        controls=[Control(W=["a"]), Control(W=["b"]), Control(W=["c"])],
    )
    filtered = subset_schedule(schedule, [1, 3])
    assert filtered.step_val.tolist() == [2.0, 4.0]
    assert filtered.step_control.tolist() == [0, 0]
    assert len(filtered.controls) == 1 and filtered.controls[0].W == ["b"]


def test_standalone_solve_delegates_one_complete_timestep():
    """MRST standaloneSolveAD is exactly one complete nonlinear timestep."""
    expected_state, expected_report = object(), object()

    class Solver:
        def solve_timestep(self, model, state, dt):
            assert model == "model" and state == "state" and dt == 2.5
            return expected_state, expected_report

    assert standalone_solve("state", "model", 2.5, Solver()) == (
        expected_state, expected_report
    )


def test_estimate_composition_cfl_direction_crossflow_and_mass_guard():
    model = SimpleNamespace(
        operators=SimpleNamespace(N=np.array([[0, 1], [1, 2]])),
        G=SimpleNamespace(cells=SimpleNamespace(num=3)),
    )
    mass = np.array([[2.0, 4.0], [4.0, 0.0], [8.0, 2.0]])
    q = np.array([[8.0, 4.0], [4.0, 2.0]])
    phase_flux = np.array([[3.0, 1.0], [2.0, -3.0]])

    outflow = estimate_composition_cfl_ad(
        model, {"flux": phase_flux}, 2.0,
        component_flux=q, component_mass=mass,
    )
    # Face 0 leaves cell 0. Face 1 is crossflow and counts for both cells.
    assert np.allclose(outflow[0], [8.0, 2.0])
    assert np.allclose(outflow[1], [1.0, 2.0])
    assert np.allclose(outflow[2], [1.0, 2.0])
    assert np.all(np.isfinite(outflow))

    inflow = estimate_composition_cfl_ad(
        model, {"flux": phase_flux}, 2.0, use_inflow=True,
        component_flux=q, component_mass=mass,
    )
    assert np.allclose(inflow[0], [0.0, 0.0])
    assert np.allclose(inflow[1], [9.0, 4.0])
    assert np.allclose(inflow[2], [1.0, 2.0])
