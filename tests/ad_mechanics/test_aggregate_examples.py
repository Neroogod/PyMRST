import numpy as np
import pytest
from importlib import import_module

adjoint_module = import_module("pymrst.modules.ad_mechanics.examples.run_adjoint_example")
all_2d_module = import_module("pymrst.modules.ad_mechanics.examples.run_all_2d_cases")
all_norne_module = import_module("pymrst.modules.ad_mechanics.examples.run_all_norne_examples")
norne_module = import_module("pymrst.modules.ad_mechanics.examples.run_norne_example")


def test_run_all_2d_cases_dispatches_complete_six_case_matrix(monkeypatch):
    calls = []

    def run_2d_case(**kwargs):
        calls.append((kwargs["fluid_model"], kwargs["method"]))
        return kwargs

    monkeypatch.setattr(all_2d_module, "run_2d_case", run_2d_case)
    result = all_2d_module.run_all_2d_cases()
    assert len(result) == 6
    assert {fluid for fluid, _ in calls} == {"water", "oil water", "blackoil"}
    assert {method for _, method in calls} == {"fully coupled", "fixed stress splitting"}


def test_run_norne_example_maps_case_name_to_script_subset(monkeypatch):
    calls = []
    script = type("Script", (), {"run": staticmethod(lambda **kw: calls.append(kw) or kw)})()
    monkeypatch.setattr(norne_module, "_load_script", lambda: script)
    assert norne_module.run_norne_example("mini Norne", n_steps=2)["subset"] == "mini"
    assert norne_module.run_norne_example("full", n_steps=3)["subset"] == "full"
    assert calls == [
        {"subset": "mini", "fluid_model": "oil water", "method": "fully coupled",
         "splitting_tolerance": 1.0e-6, "n_steps": 2},
        {"subset": "full", "fluid_model": "oil water", "method": "fully coupled",
         "splitting_tolerance": 1.0e-6, "n_steps": 3},
    ]
    norne_module.run_norne_example("mini Norne", fluid_model="water", n_steps=1)
    assert calls[-1] == {
        "subset": "mini", "fluid_model": "water", "method": "fully coupled",
        "splitting_tolerance": 1.0e-6, "n_steps": 1,
    }
    norne_module.run_norne_example(
        "mini Norne", fluid_model="water", method="fixed stress splitting",
    )
    assert calls[-1]["method"] == "fixed stress splitting"
    norne_module.run_norne_example(
        "mini Norne", fluid_model="oil water", method="fixed stress splitting",
    )
    assert calls[-1]["fluid_model"] == "oil water"
    norne_module.run_norne_example(
        "mini Norne", fluid_model="blackoil", method="fully coupled",
    )
    norne_module.run_norne_example(
        "mini Norne", fluid_model="blackoil", method="fixed stress splitting",
        splitting_tolerance=1.0e-3,
    )
    assert calls[-1]["splitting_tolerance"] == 1.0e-3


def test_run_all_norne_examples_has_ci_and_optional_field_modes(monkeypatch):
    calls = []
    def runner(case, **options):
        calls.append((case, options))
        return options

    mini = all_norne_module.run_all_norne_examples(False, runner=runner, n_steps=1)
    assert mini.norne_case == "mini Norne"
    assert len(mini.runs) == 6 and mini.converged
    assert {run.spec.fluid_model for run in mini.runs} == {
        "water", "oil water", "blackoil"
    }
    assert {run.spec.method for run in mini.runs} == {
        "fully coupled", "fixed stress splitting"
    }
    assert mini.runs[-1].spec.splitting_tolerance == 1e-3

    calls.clear()
    both = all_norne_module.run_all_norne_examples(True, runner=runner, n_steps=1)
    assert [sweep.norne_case for sweep in both] == ["mini Norne", "full"]
    assert len(calls) == 12


def test_norne_sweep_can_capture_case_failure_and_continue():
    def runner(case, **options):
        if options["fluid_model"] == "blackoil":
            raise RuntimeError("not available")
        return options

    sweep = all_norne_module.run_all_norne_examples(
        runner=runner, continue_on_error=True, n_steps=1
    )
    assert len(sweep.runs) == 6
    assert sum(run.converged for run in sweep.runs) == 4
    assert all("not available" in run.error for run in sweep.runs if not run.converged)
    assert len(sweep.summary()) == 6


def test_norne_summary_extracts_engineering_metrics():
    state = {
        "pressure": np.array([1.0e7, 2.0e7]),
        "sW": np.array([0.2, 0.8]),
        "u": np.array([0.0, -0.01, 0.02]),
        "bhp": np.array([3.0e7]),
        "qWs": np.array([0.1]),
    }
    grid = type("Grid", (), {"cells": type("Cells", (), {"num": 2})()})()
    model = type("Model", (), {
        "last_split_report": {"iterations": 7},
    })()
    spec = all_norne_module.NorneCaseSpec("water", "fixed stress splitting")
    sweep = all_norne_module.run_all_norne_examples(
        cases=[spec], runner=lambda *_args, **_kwargs: (grid, model, [{}, state]),
        n_steps=1,
    )
    summary = sweep.summary()[0]
    assert summary["pressure_mean_pa"] == 1.5e7
    assert summary["water_saturation_max"] == 0.8
    assert summary["splitting_iterations_last_step"] == 7
    assert summary["water_surface_rates_m3_s"] == [0.1]


def test_norne_acceptance_summary_records_reproducible_field_evidence():
    states = [
        {"pressure": np.array([1.0, 2.0]), "u": np.zeros(3),
         "bhp": np.array([3.0]), "qWs": np.array([0.1])},
        {"pressure": np.array([2.0, 4.0]), "u": np.ones(3),
         "bhp": np.array([5.0]), "qWs": np.array([0.2])},
    ]
    grid = type("Grid", (), {
        "cells": type("Cells", (), {"num": 2, "volumes": np.ones(2)})(),
        "nodes": type("Nodes", (), {"coords": np.zeros((2, 3))})(),
    })()
    rock = type("Rock", (), {
        "poro": np.array([0.2, 0.3]),
        "perm": np.ones((2, 3)),
    })()
    reports = [
        {"Converged": True, "Iterations": 4, "Cuts": 1,
         "Attempts": [{"Converged": False}, {"Converged": True}]},
    ]
    model = type("Model", (), {
        "rock": rock,
        "acceptance_reports": reports,
        "acceptance_timesteps": [86400.0],
    })()
    metrics = all_norne_module._result_metrics((grid, model, states))
    assert len(metrics["grid_rock_sha256"]) == 64
    assert metrics["pressure_mean_history_pa"] == [1.5, 3.0]
    assert metrics["bhp_history_pa"] == [[3.0], [5.0]]
    assert metrics["water_surface_rate_history_m3_s"] == [[0.1], [0.2]]
    assert metrics["timestep_iterations"] == [4]
    assert metrics["timestep_cuts"] == [1]
    assert metrics["all_timesteps_converged"] is True
    assert metrics["timesteps_seconds"] == [86400.0]


def test_adjoint_example_returns_forward_objective_and_gradient(monkeypatch):
    objective = object()
    states = [object(), object()]
    adjoint = object()
    monkeypatch.setattr(adjoint_module, "uplift_objective", lambda model, node, direction: objective)
    monkeypatch.setattr(adjoint_module, "simulate_forward", lambda model, state0, dts, controls: states)
    monkeypatch.setattr(adjoint_module, "evaluate_objective", lambda obj, got_states, dts: 7.0)
    monkeypatch.setattr(
        adjoint_module,
        "compute_gradient_adjoint",
        lambda model, state0, dts, controls, obj, states: adjoint,
    )
    result = adjoint_module.run_adjoint_example(
        object(), object(), np.array([1.0, 2.0]), np.ones((2, 1)), 3
    )
    assert result == {"states": states, "objective": 7.0, "adjoint": adjoint}
