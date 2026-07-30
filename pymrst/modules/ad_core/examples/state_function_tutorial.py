"""Runnable, non-plotting counterpart of MRST ``stateFunctionTutorial``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.adi import value


@dataclass(frozen=True)
class StateFunctionTutorialResult:
    grouping_names: tuple[str, ...]
    flow_properties: tuple[str, ...]
    mobility_dependencies: tuple[tuple[str, str], ...]
    relative_permeability: tuple[np.ndarray, ...]
    mobility: tuple[np.ndarray, ...]
    mobility_saturation_jacobians: tuple[np.ndarray, ...]
    cached_flow_properties: tuple[str, ...]


def state_function_tutorial(
    saturation=(0.2, 0.4, 0.6, 0.8),
) -> StateFunctionTutorialResult:
    """Inspect and evaluate a real model's lazy AD state-function graph."""

    saturation = np.asarray(saturation, dtype=float).ravel()
    if saturation.size == 0:
        raise ValueError("saturation cannot be empty")
    grid = compute_geometry(
        cart_grid([saturation.size, 1, 1], [float(saturation.size), 1.0, 1.0])
    )
    rock = make_rock(grid, 1e-13, 0.25)
    fluid = init_simple_adi_fluid(
        mu=[1.0, 2.0], rho=[1000.0, 800.0], n=[2.0, 2.0]
    )
    model = GenericBlackOilModel(grid, rock, fluid, [])
    state = State(
        pressure=np.full(grid.cells.num, 1e7),
        sW=saturation,
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    state_ad = model.init_state_ad(state)
    relative_permeability = model.get_props(
        state_ad, "RelativePermeability"
    )
    # The direct grouping call is deliberately repeated: it must return the
    # same cached objects, matching the tutorial's two evaluation surfaces.
    direct = model.FlowPropertyFunctions.get(
        model, state_ad, "RelativePermeability"
    )
    if any(a is not b for a, b in zip(relative_permeability, direct)):
        raise RuntimeError("state-function cache did not reuse evaluated values")
    mobility = model.get_props(state_ad, "Mobility")
    mobility_function = model.FlowPropertyFunctions.get_state_function("Mobility")
    return StateFunctionTutorialResult(
        grouping_names=tuple(model.groupings),
        flow_properties=tuple(model.FlowPropertyFunctions.get_names()),
        mobility_dependencies=tuple(
            (dependency.grouping, dependency.name)
            for dependency in mobility_function.dependencies
        ),
        relative_permeability=tuple(
            np.asarray(value(item)) for item in relative_permeability
        ),
        mobility=tuple(np.asarray(value(item)) for item in mobility),
        mobility_saturation_jacobians=tuple(
            item.jac[1].toarray() for item in mobility
        ),
        cached_flow_properties=tuple(
            state_ad.containers[model.FlowPropertyFunctions.struct_name]
        ),
    )


stateFunctionTutorial = state_function_tutorial

__all__ = [
    "StateFunctionTutorialResult",
    "state_function_tutorial",
    "stateFunctionTutorial",
]
