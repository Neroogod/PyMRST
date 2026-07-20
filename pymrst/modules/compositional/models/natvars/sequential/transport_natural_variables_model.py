"""MRST ``TransportNaturalVariablesModel.m`` compatibility class."""

from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel


class TransportNaturalVariablesModel(NaturalVariablesCompositionalModel):
    """Sequential transport natural-variable model surface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.staticUpwind = False
        self.upwindType = "potential"
        self.allowLargeSaturations = True
        self.maxPhaseChangesNonLinear = float("inf")

    def getEquations(self, state0, state, dt, drivingForces=None, *args, **kwargs):
        return transport_equations_natural_vars(self, state0, state, dt, drivingForces, *args, **kwargs)


def transport_equations_natural_vars(model, state0, state, dt, drivingForces=None, *args, **kwargs):
    """Compatibility dispatcher for natural-variable transport equations."""
    del args, kwargs
    if hasattr(model, "transport_step"):
        try:
            return model.transport_step(state0, state, dt, drivingForces=drivingForces)
        except TypeError:
            return model.transport_step(state0, state, dt)
    return {"model": model, "state0": state0, "state": state, "dt": dt, "drivingForces": drivingForces}, state


__all__ = ["TransportNaturalVariablesModel", "transport_equations_natural_vars"]
