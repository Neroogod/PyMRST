"""MRST ``models/GenericOverallCompositionModel.m`` compatibility class."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.components import ImmiscibleComponent
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.modules.compositional.properties.components import EquationOfStateComponent
from pymrst.utils.units import ATM


class GenericOverallCompositionModel(OverallCompositionModel):
    """Overall-composition model with MRST generic-model utility surface."""

    def __init__(self, *args, Components=None, components=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.OutputStateFunctions = ["ComponentTotalMass", "Density"]
        self.Components = list(Components if Components is not None else (components or []))

    @property
    def output_state_functions(self) -> list[str]:
        return self.OutputStateFunctions

    @output_state_functions.setter
    def output_state_functions(self, value) -> None:
        self.OutputStateFunctions = list(value)

    def getPhaseNames(self) -> tuple[str, ...]:
        """Return active phase labels in MRST order."""
        return (("W",) if self.water else tuple()) + ("O", "G")

    def getNumberOfPhases(self) -> int:
        return len(self.getPhaseNames())

    def getLiquidIndex(self) -> int:
        return int(bool(self.water))

    def getVaporIndex(self) -> int:
        return int(bool(self.water)) + 1

    def getNonEoSPhaseNames(self) -> tuple[str, ...]:
        return tuple(ph for ph in self.getPhaseNames() if ph not in {"O", "G"})

    def getComponentNames(self) -> tuple[str, ...]:
        """Return MRST generic component names."""
        if self.Components:
            return tuple(str(getattr(component, "name", component)) for component in self.Components)
        return tuple(str(name) for name in self.mix.names)

    def validateModel(self, *args, **kwargs):
        """Populate generic components if needed and return ``self``."""
        del args, kwargs
        if not self.Components:
            self.Components = []
            if self.water:
                self.Components.append(ImmiscibleComponent("water", 0))
            p_ref = float(np.mean(getattr(self, "T", np.array([ATM])) * 0.0 + ATM))
            t_ref = float(np.mean(getattr(self, "T", np.array([300.0]))))
            for ix, name in enumerate(self.mix.names):
                self.Components.append(
                    EquationOfStateComponent(
                        str(name),
                        pressure=p_ref,
                        T=t_ref,
                        component_index=ix,
                        surface_phase_mass_fractions=np.ones(2) / 2.0,
                        surface_phase_density_pure=np.asarray(self.rho_s[:2], dtype=float),
                        molar_mass=float(self.mix.molarMass[ix]),
                    )
                )
        return self

    def getPrimaryVariables(self, state):
        """Return MRST overall-composition primary variables."""
        pressure = np.asarray(state.pressure, dtype=float)
        z = np.asarray(state.z, dtype=float)
        component_names = tuple(str(name) for name in self.mix.names)
        names = ["pressure", *component_names[1:]]
        variables = [pressure, *[z[:, i] for i in range(1, z.shape[1])]]
        if self.water:
            names.append("sW")
            variables.append(np.asarray(state.sW, dtype=float))
        origin = [self.__class__.__name__] * len(names)
        return variables, names, origin

    def initStateAD(self, state, vars, names, origin=None):
        """MRST-style state initialization from primary variables."""
        del origin
        out = state.copy() if hasattr(state, "copy") else state
        names = list(names)
        if "pressure" in names:
            out.pressure = np.asarray(vars[names.index("pressure")], dtype=float)
        z = np.asarray(out.z, dtype=float).copy()
        component_names = tuple(str(name) for name in self.mix.names)
        missing = []
        remainder = np.ones(z.shape[0])
        for ix, cname in enumerate(component_names):
            if ix == 0:
                continue
            if cname in names:
                z[:, ix] = np.asarray(vars[names.index(cname)], dtype=float)
            missing.append(ix)
            remainder -= z[:, ix]
        z[:, 0] = remainder
        out.z = np.maximum(z, self.min_z)
        out.z = out.z / out.z.sum(axis=1, keepdims=True)
        if self.water and "sW" in names:
            out.sW = np.asarray(vars[names.index("sW")], dtype=float)
        return out

    def updateState(self, state, problem=None, dx=None, forces=None):
        """Remove transient fractional derivatives before returning state."""
        del problem, dx, forces
        if isinstance(state, dict):
            state.pop("FractionalDerivatives", None)
        elif hasattr(state, "FractionalDerivatives"):
            delattr(state, "FractionalDerivatives")
        return state, None

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        """MRST-compatible convergence cleanup."""
        del state0, dt, drivingForces
        for field in ("dz", "eos", "dpRel", "dpAbs", "switched", "switchCount"):
            if isinstance(state, dict):
                state.pop(field, None)
            elif hasattr(state, field):
                delattr(state, field)
        return state, None


__all__ = ["GenericOverallCompositionModel"]
