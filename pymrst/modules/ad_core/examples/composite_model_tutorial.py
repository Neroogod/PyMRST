"""Runnable coupled-reservoir counterpart of MRST ``compositeModelTutorial``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.models.composite.composite_model import CompositeModel
from pymrst.modules.ad_core.statefunctions.composite.coupling_term import (
    LinearPressureCoupling,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock


@dataclass(frozen=True)
class CompositeTutorialResult:
    residual_without_coupling: np.ndarray
    residual_with_coupling: np.ndarray
    jacobian_shape: tuple[int, int]
    off_diagonal_nonzeros: int
    conservation_error: float
    layout: dict


def _submodel(pressure: float):
    grid = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    rock = make_rock(grid, 1e-13, 0.25)
    fluid = init_simple_adi_fluid(
        mu=[1.0, 2.0], rho=[1000.0, 800.0], n=[2.0, 2.0]
    )
    model = GenericBlackOilModel(grid, rock, fluid, [])
    raw = State(
        pressure=np.full(grid.cells.num, pressure),
        sW=np.full(grid.cells.num, 0.4),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    return model, raw.copy(), raw.copy()


def composite_model_tutorial(
    *,
    pressure_a: float = 2.0e7,
    pressure_b: float = 1.0e7,
    coupling_coefficient: float = 1e-10,
    dt: float = 86400.0,
) -> CompositeTutorialResult:
    """Assemble two reservoirs linked by a conservative pressure coupling."""

    model_a, state_a, state0_a = _submodel(pressure_a)
    model_b, state_b, state0_b = _submodel(pressure_b)
    composite = CompositeModel({"LayerA": model_a, "LayerB": model_b})
    states = {"LayerA": state_a, "LayerB": state_b}
    states0 = {"LayerA": state0_a, "LayerB": state0_b}
    uncoupled, uncoupled_jacobian, layout = composite.assemble_coupled_system(
        states0, states, dt
    )
    coupling = LinearPressureCoupling(
        "LayerA", "LayerB", coupling_coefficient
    )
    coupled, jacobian, layout = composite.assemble_coupled_system(
        states0, states, dt, couplings=[coupling]
    )

    row_a, _, size_a = layout["LayerA"]
    row_b, _, size_b = layout["LayerB"]
    off_ab = jacobian[row_a:row_a + size_a, row_b:row_b + size_b]
    off_ba = jacobian[row_b:row_b + size_b, row_a:row_a + size_a]
    delta_a = coupled[row_a:row_a + size_a] - uncoupled[
        row_a:row_a + size_a
    ]
    delta_b = coupled[row_b:row_b + size_b] - uncoupled[
        row_b:row_b + size_b
    ]
    return CompositeTutorialResult(
        residual_without_coupling=uncoupled,
        residual_with_coupling=coupled,
        jacobian_shape=jacobian.shape,
        off_diagonal_nonzeros=int(off_ab.nnz + off_ba.nnz),
        conservation_error=float(np.max(np.abs(delta_a + delta_b))),
        layout=layout,
    )


compositeModelTutorial = composite_model_tutorial

__all__ = [
    "CompositeTutorialResult",
    "composite_model_tutorial",
    "compositeModelTutorial",
]
