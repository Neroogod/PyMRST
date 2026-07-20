"""MRST ``transportEquationCompositional.m`` compatibility helper."""


def transport_equation_compositional(model, state0, state, dt, *args, **kwargs):
    """Dispatch to PyMRST's sequential transport step."""
    return model.transport_step(state0, state, dt, *args, **kwargs)


transportEquationCompositional = transport_equation_compositional

__all__ = ["transport_equation_compositional", "transportEquationCompositional"]
