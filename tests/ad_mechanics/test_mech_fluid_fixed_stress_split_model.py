import numpy as np
import pytest

from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.mech_fluid_fixed_stress_split_model import (
    MechFluidFixedStressSplitModel,
    simulate_fixed_stress_schedule,
)
from pymrst.modules.ad_mechanics.poromech import FixedStressSplitModel


def test_fixed_stress_layout_owner_is_operational_subclass():
    assert issubclass(MechFluidFixedStressSplitModel, FixedStressSplitModel)
    assert MechFluidFixedStressSplitModel is not FixedStressSplitModel


def test_compute_mech_term_exposes_mrst_field_names():
    class Mechanics:
        @staticmethod
        def fixed_stress_terms(xd, pressure):
            return np.asarray(xd) - pressure, np.ones_like(np.asarray(pressure)) * 2

    model = object.__new__(MechFluidFixedStressSplitModel)
    model.mech = Mechanics()
    result = model.compute_mech_term({"xd": np.array([3.0]), "pressure": np.array([1.0])})
    assert result["sTerm"] == pytest.approx([2.0])
    assert result["pTerm"] == pytest.approx([2.0])


def test_compute_mech_term_validates_state_contract():
    model = object.__new__(MechFluidFixedStressSplitModel)
    with pytest.raises(KeyError, match="pressure"):
        model.compute_mech_term({"xd": np.array([0.0])})


def test_step_function_reports_nested_iterations_and_rolls_back_failure():
    model = object.__new__(MechFluidFixedStressSplitModel)
    model.last_split_report = {}

    def fail(state0, dt):
        model.last_split_report = {
            "converged": False,
            "iterations": 2,
            "failure": "fluid Newton solve did not converge",
            "outer": [
                {"iteration": 1, "mechanics_residual": 3.0,
                 "flow": {"iterations": 4, "converged": True}},
                {"iteration": 2, "mechanics_residual": None,
                 "flow": {"iterations": 5, "converged": False}},
            ],
        }
        return {"pressure": np.array([999.0])}, False, 1

    model.step = fail
    trial = {"pressure": np.array([2.0])}
    old = {"pressure": np.array([1.0])}
    accepted, report = model.step_function(trial, old, 10.0)
    assert accepted is not trial
    assert accepted["pressure"] == pytest.approx([2.0])
    assert report["Failure"]
    assert report["FailureMsg"] == "fluid Newton solve did not converge"
    assert report["Iterations"] == 2
    assert report["Residuals"] == [3.0]
    assert len(report["SplittingIterations"]) == 2


def test_step_function_returns_converged_state_and_residual_history():
    model = object.__new__(MechFluidFixedStressSplitModel)

    def succeed(state0, dt):
        model.last_split_report = {
            "converged": True, "iterations": 1, "failure": "",
            "outer": [{"iteration": 1, "mechanics_residual": 1e-12,
                       "flow": {"iterations": 2, "converged": True}}],
        }
        return {"pressure": np.array([4.0])}, True, 1

    model.step = succeed
    accepted, report = model.step_function(
        {"pressure": np.array([2.0])}, {"pressure": np.array([1.0])}, 10.0
    )
    assert accepted["pressure"] == pytest.approx([4.0])
    assert report["Converged"]
    assert report["ResidualsConverged"]
    assert report["Residuals"] == [1e-12]


def test_schedule_cuts_failed_step_rolls_back_and_preserves_attempts():
    class Model:
        def __init__(self):
            self.calls = []
            self.last_split_report = {}

        def step(self, state, dt):
            self.calls.append((float(state["value"]), dt))
            state["value"] = 999.0  # rejected mutation must never leak
            ok = dt <= 4.0
            self.last_split_report = {
                "converged": ok, "iterations": 2,
                "failure": "" if ok else "forced cut",
                "outer": [],
            }
            candidate = {"value": self.calls[-1][0] + dt}
            return candidate, ok, 2

    initial = {"value": 0.0}
    model = Model()
    states, reports = simulate_fixed_stress_schedule(
        model, initial, [8.0], max_timestep_cuts=2
    )
    assert initial == {"value": 0.0}
    assert model.calls == [(0.0, 8.0), (0.0, 4.0), (4.0, 4.0)]
    assert states == [{"value": 8.0}]
    assert reports[0]["Cuts"] == 1
    assert [a["Converged"] for a in reports[0]["Attempts"]] == [False, True, True]
    assert reports[0]["AcceptedTime"] == pytest.approx(8.0)


def test_schedule_enforces_cut_and_minimum_timestep_limits():
    class Failure:
        last_split_report = {"failure": "always fails"}

        @staticmethod
        def step(state, dt):
            return {"corrupt": True}, False, 1

    with pytest.raises(RuntimeError, match="after 1 timestep cuts"):
        simulate_fixed_stress_schedule(Failure(), {}, [4.0], max_timestep_cuts=1)
    with pytest.raises(RuntimeError, match="below min_timestep"):
        simulate_fixed_stress_schedule(
            Failure(), {}, [4.0], max_timestep_cuts=5, min_timestep=3.0
        )
