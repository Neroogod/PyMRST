"""Install AvgMPFA (MRST ``setAvgMPFADiscretization.m``)."""

import numpy as np

from pymrst.modules.nfvm.solvers.avg_mpfa import AvgMPFA
from pymrst.modules.nfvm.solvers.set_ntpfa_discretization import _unwrap
from pymrst.modules.nfvm.solvers.utils.setup_gdz import setup_gdz


def set_avg_mpfa_discretization(model, **options):
    target = _unwrap(model)
    discretization = AvgMPFA.from_grid(target.G, target.rock, **options)
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


setAvgMPFADiscretization = set_avg_mpfa_discretization

__all__ = ["set_avg_mpfa_discretization", "setAvgMPFADiscretization"]
