"""MRST ``pressureEquationCompositional.m`` compatibility helper."""


def pressure_equation_compositional(model, state, dt, *args, **kwargs):
    """Dispatch to PyMRST's sequential pressure step."""
    return model.pressure_step(state, dt, *args, **kwargs)


pressureEquationCompositional = pressure_equation_compositional

__all__ = ["pressure_equation_compositional", "pressureEquationCompositional"]
