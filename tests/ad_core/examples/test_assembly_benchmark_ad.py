"""Tests for the MRST ``assemblyBenchmarkAD`` counterpart."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.examples import (
    AssemblyBenchmarkCase,
    assembly_benchmark_ad,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import init_variables_adi


class LinearModel:
    G = SimpleNamespace(cells=SimpleNamespace(num=4))

    def init_state_ad(self, state):
        (x,) = init_variables_adi(state["x"])
        return State(x=x)

    def get_model_equations(self, state0, state, dt):
        return [(state["x"] - state0["x"]) / dt + 2.0 * state["x"]]

    def get_number_of_components(self):
        return 1


def test_times_state_and_equation_assembly_separately():
    case = AssemblyBenchmarkCase(
        LinearModel(),
        State(x=np.arange(4.0)),
        State(x=np.arange(4.0) + 1.0),
        2.0,
        physics="linear",
        backend="sparse-test",
    )
    result = assembly_benchmark_ad(case, iterations=3)[0]

    assert result.physics == "linear"
    assert result.backend == "sparse-test"
    assert result.cell_count == result.degrees_of_freedom == 4
    assert result.state_time > 0.0
    assert result.equation_time > 0.0
    assert result.total_time == result.state_time + result.equation_time
    assert result.seconds_per_cell > 0.0
    assert result.seconds_per_dof > 0.0
    assert 0.0 < result.state_fraction < 1.0


def test_multiple_cases_and_custom_reducer_are_supported():
    case = AssemblyBenchmarkCase(
        LinearModel(), State(x=np.ones(4)), State(x=np.ones(4)), 1.0
    )
    results = assembly_benchmark_ad(
        [case, case], iterations=2, reducer=np.min
    )
    assert len(results) == 2
    assert all(result.iterations == 2 for result in results)


def test_real_reservoir_model_assembles_during_benchmark():
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock

    grid = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
    rock = make_rock(grid, 1e-13, 0.25)
    fluid = init_simple_adi_fluid(
        mu=[1.0, 2.0], rho=[1000.0, 800.0], n=[2, 2]
    )
    model = GenericBlackOilModel(grid, rock, fluid, [])
    state = State(
        pressure=np.full(grid.cells.num, 1e7),
        sW=np.full(grid.cells.num, 0.3),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )

    result = assembly_benchmark_ad(
        AssemblyBenchmarkCase(
            model, state, state, 86400.0, physics="immiscible"
        ),
        iterations=1,
    )[0]

    assert result.physics == "immiscible"
    assert result.backend == "Sparse"
    assert result.cell_count == 6
    assert result.degrees_of_freedom == 12
    assert result.equation_time > 0.0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        assembly_benchmark_ad([], iterations=0)
    with pytest.raises(TypeError):
        assembly_benchmark_ad([object()])
