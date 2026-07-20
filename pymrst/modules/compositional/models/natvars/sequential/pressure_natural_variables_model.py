"""MRST ``PressureNaturalVariablesModel.m`` compatibility class."""

from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel


class PressureNaturalVariablesModel(NaturalVariablesCompositionalModel):
    """Sequential pressure natural-variable model surface."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.useIncTolPressure = True
        self.useIncTolComposition = True
        self.allowLargeSaturations = True
        self.maxPhaseChangesNonLinear = 20
        self.outputFluxes = True
        self.singlePhaseStrategy = "numerical"
        self.twoPhaseStrategy = "numerical"
        self.singlePhaseDifferentiation = "numerical"
        self.twoPhaseDifferentiation = "numerical"

    def getEquations(self, state0, state, dt, drivingForces=None, *args, **kwargs):
        return equations_natural_variables_pressure(self, state0, state, dt, drivingForces, *args, **kwargs)

    def updateState(self, state, problem=None, dv=None, drivingForces=None):
        if dv:
            state.dpressure = dv[0]
        return state, None

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        if hasattr(state, "switchCount"):
            state.switchCount_p = state.switchCount
        return state, None


def equations_natural_variables_pressure(model, state0, state, dt, drivingForces=None, *args, **kwargs):
    """Compatibility dispatcher for reduced pressure equations."""
    del args, kwargs
    if hasattr(model, "pressure_step"):
        try:
            return model.pressure_step(state, dt, drivingForces=drivingForces)
        except TypeError:
            return model.pressure_step(state, dt)
    return {"model": model, "state0": state0, "state": state, "dt": dt, "drivingForces": drivingForces, "reduceToPressure": True}, state


__all__ = ["PressureNaturalVariablesModel", "equations_natural_variables_pressure"]
