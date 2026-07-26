"""Install a facewise hybrid gradient (MRST ``setHybridDiscretization.m``)."""

from pymrst.modules.nfvm.solvers.hybrid import Hybrid
from pymrst.modules.nfvm.solvers.set_ntpfa_discretization import _unwrap
from pymrst.modules.nfvm.solvers.utils.setup_gdz import setup_gdz


class _ModelGradient:
    def __init__(self, model):
        self.model = _unwrap(model)

    def gradient(self, pressure):
        return self.model.operators.grad(pressure)


def set_hybrid_discretization(model, models, faceblocks, **_options):
    target = _unwrap(model)
    candidates = [_ModelGradient(candidate) for candidate in models]
    hybrid = Hybrid(target.G, candidates, list(faceblocks))
    target.operators.grad = hybrid.gradient
    target.gdz = setup_gdz(target, target.operators.grad)
    target.operators.gdz = target.gdz
    target.nfvm_discretization = hybrid
    return model


setHybridDiscretization = set_hybrid_discretization

__all__ = ["set_hybrid_discretization", "setHybridDiscretization"]
