"""MRST ``models/ThreePhaseCompositionalModel.m`` compatibility base."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.backend import backend_flash
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.models.eos import EquationOfStateModel


class ThreePhaseCompositionalModel:
    """Base compositional model surface shared by MRST formulations.

    MRST uses this as an abstract superclass for natural-variable and
    overall-composition models.  PyMRST's concrete solvers live elsewhere; this
    class ports the common MRST-facing utility API so examples and validation
    code can depend on the same names without changing solver internals.
    """

    def __init__(self, G: Any = None, rock: Any = None, fluid: Any = None, compFluid: Any = None, **kwargs):
        if compFluid is None:
            compFluid = kwargs.pop("comp_fluid", kwargs.pop("mixture", None))
        if isinstance(compFluid, EquationOfStateModel):
            eos_model = compFluid
        elif isinstance(compFluid, Mixture):
            eos_model = EquationOfStateModel(G, compFluid)
        elif compFluid is None:
            eos_model = kwargs.pop("eos_model", None)
            if eos_model is None:
                raise ValueError("compFluid, mixture, or eos_model must be supplied")
        else:
            raise TypeError("compFluid must be a Mixture or EquationOfStateModel")

        self.G = G
        self.rock = rock
        self.fluid = fluid
        self.EOSModel = eos_model
        self.EOSNonLinearSolver = kwargs.pop("EOSNonLinearSolver", kwargs.pop("eos_nonlinear_solver", None))
        self.nonlinearTolerance = float(kwargs.pop("nonlinearTolerance", kwargs.pop("nonlinear_tolerance", 1.0e-3)))
        self.incTolPressure = float(kwargs.pop("incTolPressure", kwargs.pop("inc_tol_pressure", 1.0e-3)))
        self.incTolComposition = float(kwargs.pop("incTolComposition", kwargs.pop("inc_tol_composition", 1.0e-3)))
        self.useIncTolComposition = bool(kwargs.pop("useIncTolComposition", kwargs.pop("use_inc_tol_composition", False)))
        self.fugacityTolerance = float(kwargs.pop("fugacityTolerance", kwargs.pop("fugacity_tolerance", 1.0e-3)))
        self.liquidPhase = str(kwargs.pop("liquidPhase", kwargs.pop("liquid_phase", "O")))
        self.vaporPhase = str(kwargs.pop("vaporPhase", kwargs.pop("vapor_phase", "G")))
        self.water = bool(kwargs.pop("water", True))
        self.oil = bool(kwargs.pop("oil", True))
        self.gas = bool(kwargs.pop("gas", True))
        self.dzMaxAbs = float(kwargs.pop("dzMaxAbs", kwargs.pop("dz_max_abs", 0.1)))
        self.dsMaxAbs = float(kwargs.pop("dsMaxAbs", kwargs.pop("ds_max_abs", 0.1)))
        self.minimumPressure = float(kwargs.pop("minimumPressure", kwargs.pop("minimum_pressure", 0.0)))
        self.dpMaxRel = float(kwargs.pop("dpMaxRel", kwargs.pop("dp_max_rel", 0.25)))
        self.Components = kwargs.pop("Components", kwargs.pop("components", []))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")

    @property
    def mix(self) -> Mixture | None:
        return getattr(self.EOSModel, "CompositionalMixture", getattr(self.EOSModel, "mixture", None))

    @property
    def PropertyModel(self):
        """MRST dependent property model alias."""
        return self.EOSModel.PropertyModel

    def getComponentNames(self) -> tuple[str, ...]:
        """Return active component names with EOS components appended."""
        names = []
        if self.water:
            names.append("water")
        names.extend(str(name) for name in self.EOSModel.getComponentNames())
        return tuple(names)

    def getEoSComponentMask(self) -> np.ndarray:
        """Return boolean mask for EOS-governed components."""
        if self.Components:
            return np.array([
                comp.__class__.__name__ == "EquationOfStateComponent"
                for comp in self.Components
            ], dtype=bool)
        ncomp = self.EOSModel.getNumberOfComponents()
        if self.water:
            return np.array([False] + [True] * ncomp, dtype=bool)
        return np.ones(ncomp, dtype=bool)

    def getPhaseNames(self) -> tuple[str, ...]:
        """Return active phase labels in MRST order."""
        phases = []
        if self.water:
            phases.append("W")
        if self.oil:
            phases.append("O")
        if self.gas:
            phases.append("G")
        return tuple(phases)

    def getNumberOfPhases(self) -> int:
        return len(self.getPhaseNames())

    def getLiquidIndex(self) -> int:
        """Zero-based index of MRST's liquid/oil EOS phase."""
        return self.getPhaseNames().index(self.liquidPhase)

    def getVaporIndex(self) -> int:
        """Zero-based index of MRST's vapour/gas EOS phase."""
        return self.getPhaseNames().index(self.vaporPhase)

    def getEoSPhaseIndices(self) -> tuple[int, int]:
        return self.getLiquidIndex(), self.getVaporIndex()

    def getNonEoSPhaseNames(self) -> tuple[str, ...]:
        liquid = self.liquidPhase
        vapor = self.vaporPhase
        return tuple(ph for ph in self.getPhaseNames() if ph not in {liquid, vapor})

    def getVariableField(self, name: str, *args):
        """Map MRST variable aliases to state fields and component indices."""
        del args
        key = str(name).lower()
        if key in {"z", "components"}:
            return "components", slice(None)
        if key in {"x", "liquidmf", "liquidmolefractions"}:
            return "x", slice(None)
        if key in {"y", "vapormf", "vapormolefractions"}:
            return "y", slice(None)
        if key in {"l", "liquid"}:
            return "L", 0
        if key == "z_l":
            return "Z_L", 0
        if key == "z_v":
            return "Z_V", 0
        if key in {"k", "equilibriumconstants"}:
            return "K", slice(None)
        if key == "sl":
            return "s", self.getLiquidIndex()
        if key == "sv":
            return "s", self.getVaporIndex()

        eos_names = tuple(str(x).lower() for x in self.EOSModel.getComponentNames())
        if key in eos_names:
            return "components", eos_names.index(key)
        if key == "temperature":
            return "T", 0
        if key == "pressure":
            return "pressure", 0
        raise KeyError(f"unknown variable field {name!r}")

    def validateState(self, state):
        """MRST-style state validation and scalar-temperature expansion."""
        ncell = self._num_cells(state)
        components = _state_get(state, "components", _state_get(state, "z", None))
        if components is None:
            raise ValueError("state must contain components or z")
        components = np.asarray(components, dtype=float)
        if components.ndim != 2:
            raise ValueError("components must be a 2-D array")
        expected = self.EOSModel.getNumberOfComponents()
        if self.water and components.shape[1] == expected + 1:
            hydrocarbon = components[:, 1:]
        else:
            hydrocarbon = components
        if hydrocarbon.shape != (ncell, expected):
            raise ValueError(f"hydrocarbon composition must have shape {(ncell, expected)}")
        if np.any(hydrocarbon > 1.0 + 1.0e-12):
            raise AssertionError("Initial mole fractions are larger than unity.")
        hydrocarbon = _normalize_rows(np.maximum(hydrocarbon, self.EOSModel.minimumComposition))
        if self.water and components.shape[1] == expected + 1:
            components = np.column_stack([components[:, 0], hydrocarbon])
        else:
            components = hydrocarbon
        _state_set(state, "components", components)

        T = _state_get(state, "T", _state_get(state, "temperature", None))
        if T is not None:
            T = np.asarray(T, dtype=float)
            if T.ndim == 0 or T.size == 1:
                T = np.full(ncell, float(T.reshape(-1)[0]))
            else:
                T = np.asarray(T, dtype=float).reshape(-1)
                if T.size != ncell:
                    raise ValueError(f"Temperature must have {ncell} entries")
            _state_set(state, "T", T)

        if _state_get(state, "x", None) is None or _state_get(state, "K", None) is None:
            state = self.computeFlash(state, np.inf)
        return state

    def computeFlash(self, state, dt=np.inf, iteration: int = 1):
        """Flash a state with this model's EOS."""
        del dt, iteration
        pressure = _state_get(state, "pressure")
        T = _state_get(state, "T", _state_get(state, "temperature", None))
        if T is None:
            T = np.full(self._num_cells(state), 300.0)
        z = _state_get(state, "components", _state_get(state, "z", None))
        z = np.asarray(z, dtype=float)
        if self.water and z.shape[1] == self.EOSModel.getNumberOfComponents() + 1:
            z_hc = z[:, 1:]
        else:
            z_hc = z

        if hasattr(self.EOSModel, "flash"):
            flashed = self.EOSModel.flash(pressure, T, z_hc)
            _state_set(state, "x", flashed.x)
            _state_set(state, "y", flashed.y)
            _state_set(state, "K", flashed.K)
            _state_set(state, "L", 1.0 - flashed.V)
            _state_set(state, "Z_L", flashed.Z_L)
            _state_set(state, "Z_V", flashed.Z_V)
        else:
            flashed = backend_flash(self.mix, self.EOSModel, pressure, T, z_hc)
            _state_set(state, "x", flashed.x)
            _state_set(state, "y", flashed.y)
            _state_set(state, "K", flashed.K)
            _state_set(state, "L", flashed.L)
            _state_set(state, "Z_L", flashed.Z_L)
            _state_set(state, "Z_V", flashed.Z_V)
        return state

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        """Remove temporary nonlinear fields after a converged step."""
        del state0, dt, drivingForces
        for field in ("dz", "eos", "dpRel", "dpAbs", "switched", "switchCount"):
            _state_del(state, field)
        return state, None

    def _num_cells(self, state) -> int:
        if self.G is not None and hasattr(self.G, "cells") and hasattr(self.G.cells, "num"):
            return int(self.G.cells.num)
        pressure = _state_get(state, "pressure", None)
        if pressure is not None:
            return int(np.asarray(pressure).size)
        components = _state_get(state, "components", _state_get(state, "z", None))
        if components is not None:
            return int(np.asarray(components).shape[0])
        raise ValueError("could not infer cell count")


def _normalize_rows(x):
    x = np.maximum(np.asarray(x, dtype=float), 1.0e-300)
    return x / x.sum(axis=1, keepdims=True)


def _state_get(state, name: str, default=None):
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _state_set(state, name: str, value) -> None:
    if isinstance(state, dict):
        state[name] = value
    else:
        setattr(state, name, value)


def _state_del(state, name: str) -> None:
    if isinstance(state, dict):
        state.pop(name, None)
    elif hasattr(state, name):
        delattr(state, name)


__all__ = ["ThreePhaseCompositionalModel"]
