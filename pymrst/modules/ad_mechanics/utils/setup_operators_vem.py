"""VEM mechanics operators (MRST ``setupOperatorsVEM.m`` layout)."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel, MechanicOperators
from pymrst.modules.vemmech.vem import ElasticBC


def setup_operators_vem(
    g: Grid,
    C: np.ndarray,
    alpha: np.ndarray | float,
    el_bc: ElasticBC,
    load=None,
    alpha_scaling: float = 1.0,
    S=None,
    stabilization=None,
    initial_stress=None,
) -> MechanicOperators:
    """Assemble the VEM stiffness/divergence operators used by mechanics."""
    model = MechanicModel(
        g, C, alpha, el_bc, load=load, alpha_scaling=alpha_scaling,
        S=S, stabilization=stabilization,
        initial_stress=initial_stress,
    )
    return model.operators


__all__ = ["MechanicModel", "MechanicOperators", "setup_operators_vem"]
