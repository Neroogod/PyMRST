"""Compact AD comparison of TPFA and MPFA pressure discretizations."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.statefunctions.flow_discretization import (
    FlowDiscretization,
)
from pymrst.modules.ad_core.utils import set_mpfa_discretization
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.params import make_rock
from pymrst.utils.adi import init_variables_adi, value


@dataclass(frozen=True)
class MPFATPFAComparison:
    tpfa_gradient: np.ndarray
    mpfa_gradient: np.ndarray
    tpfa_jacobian: np.ndarray
    mpfa_jacobian: np.ndarray
    tpfa_constant_error: float
    mpfa_constant_error: float

    @property
    def gradient_difference(self) -> float:
        return float(np.linalg.norm(self.mpfa_gradient - self.tpfa_gradient))

    @property
    def jacobian_difference(self) -> float:
        return float(np.linalg.norm(self.mpfa_jacobian - self.tpfa_jacobian))


def _comparison_model():
    grid = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    # Full off-diagonal permeability makes TPFA inconsistent and ensures the
    # multipoint stencil is genuinely different.
    permeability = np.tile(np.array([100.0, 35.0, 55.0]) * 1e-15,
                           (grid.cells.num, 1))
    rock = make_rock(grid, permeability, 0.2)
    model = SimpleNamespace(
        G=grid,
        rock=rock,
        operators=setup_operators_tpfa(grid, rock),
        fluid=SimpleNamespace(has_pc=False),
        component_state_keys=lambda: [],
    )
    model.FlowDiscretization = FlowDiscretization(model)
    return model


def mpfa_vs_tpfa_with_ad_example() -> MPFATPFAComparison:
    """Run MRST ``MPFAvsTPFAwithADExample`` at the operator/AD level."""

    tpfa_model = _comparison_model()
    mpfa_model = set_mpfa_discretization(_comparison_model())
    pressure_values = (
        2.0 * mpfa_model.G.cells.centroids[:, 0]
        - 0.75 * mpfa_model.G.cells.centroids[:, 1]
    )
    (pressure,) = init_variables_adi(pressure_values)

    tpfa = tpfa_model.FlowDiscretization.get_state_function("PressureGradient")
    mpfa = mpfa_model.FlowDiscretization.get_state_function("PressureGradient")
    tpfa_gradient = tpfa.Grad(pressure)
    mpfa_gradient = mpfa.Grad(pressure)
    constants = np.ones(mpfa_model.G.cells.num)
    return MPFATPFAComparison(
        tpfa_gradient=np.asarray(value(tpfa_gradient)),
        mpfa_gradient=np.asarray(value(mpfa_gradient)),
        tpfa_jacobian=tpfa_gradient.jac[0].toarray(),
        mpfa_jacobian=mpfa_gradient.jac[0].toarray(),
        tpfa_constant_error=float(np.max(np.abs(tpfa.Grad(constants)))),
        mpfa_constant_error=float(np.max(np.abs(mpfa.Grad(constants)))),
    )


MPFAvsTPFAwithADExample = mpfa_vs_tpfa_with_ad_example

__all__ = [
    "MPFATPFAComparison",
    "mpfa_vs_tpfa_with_ad_example",
    "MPFAvsTPFAwithADExample",
]
