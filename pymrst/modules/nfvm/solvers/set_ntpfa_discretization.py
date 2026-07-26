"""Install NTPFA on an AD-core model (MRST ``setNTPFADiscretization.m``)."""

import numpy as np

from pymrst.modules.nfvm.solvers.ntpfa import NTPFA
from pymrst.modules.nfvm.solvers.utils.setup_gdz import setup_gdz


def _unwrap(model):
    return getattr(model, "parent_model", model)


def set_ntpfa_discretization(model, **options):
    """Replace the model's pressure-gradient operator with scaled NTPFA."""
    target = _unwrap(model)
    discretization = NTPFA.from_grid(target.G, target.rock, **options)
    scale = np.divide(
        1.0,
        target.operators.T,
        out=np.zeros_like(target.operators.T, dtype=float),
        where=np.asarray(target.operators.T) != 0,
    )
    target.operators.grad = lambda pressure: (
        discretization.unscaled_gradient(pressure)*scale
    )
    target.gdz = setup_gdz(target, target.operators.grad)
    target.operators.gdz = target.gdz
    target.nfvm_discretization = discretization
    return model


setNTPFADiscretization = set_ntpfa_discretization

__all__ = ["set_ntpfa_discretization", "setNTPFADiscretization"]
