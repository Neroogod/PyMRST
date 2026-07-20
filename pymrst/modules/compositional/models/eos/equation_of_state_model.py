"""Compatibility surface for MRST ``EquationOfStateModel.m``."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.eos import (
    OMEGA_A,
    OMEGA_B,
    SQRT2,
    Mixture,
    _cubic_roots,
)
from pymrst.modules.compositional.eos_ad import (
    equations_equilibrium,
    flash_ad,
    phase_compressibility_ad,
    phase_fraction_derivatives,
)
from pymrst.modules.compositional.fluids.compositional_property_model import (
    CompositionalPropertyModel,
)
from pymrst.modules.compositional.flow_props import (
    mass_fraction,
    mole_fraction,
    phase_saturations,
)
from pymrst.modules.compositional.utils.estimate_equilibrium_wilson import (
    estimateEquilibriumWilson,
)
from pymrst.modules.compositional.utils.phase_stability_test import phaseStabilityTest
from pymrst.modules.compositional.utils.solve_rachford_rice_vle import solveRachfordRiceVLE
from pymrst.utils.adi import value


class EquationOfStateModel:
    """MRST-style cubic EOS model shell.

    The full nonlinear flash/equation assembly remains in PyMRST's existing
    EOS and AD helpers. This class mirrors MRST's object API for examples and
    facility/property code that expect ``EquationOfStateModel`` fields such as
    ``CompositionalMixture``, ``PropertyModel``, ``omegaA`` and ``omegaB``.
    """

    def __init__(self, G: Any = None, fluid: Mixture | None = None, eosname: str = "PR", **kwargs):
        if fluid is None:
            fluid = kwargs.pop("mixture", None)
        if "eos_name" in kwargs:
            eosname = kwargs.pop("eos_name")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        if fluid is not None and not isinstance(fluid, Mixture):
            raise TypeError("fluid must be a compositional Mixture")

        self.G = G
        self.CompositionalMixture = fluid
        self.nonlinearTolerance = 1.0e-4
        self.method = "ssi"
        self.maxSSI = np.inf
        self.selectGibbsMinimum = True
        self.alpha = None
        self.minimumComposition = 1.0e-8
        self.minimumSaturation = 1.0e-8
        self.equilibriumConstantFunctions: list[Any] = []
        self.extraOutput = 0
        self.PropertyModel = CompositionalPropertyModel(fluid)
        self.eosA = None
        self.eosB = None
        self.eosType: int | str | None = None
        self.setType(eosname)

    @property
    def mixture(self) -> Mixture | None:
        """Pythonic alias for MRST's ``CompositionalMixture`` property."""
        return self.CompositionalMixture

    @mixture.setter
    def mixture(self, value: Mixture | None) -> None:
        if value is not None and not isinstance(value, Mixture):
            raise TypeError("mixture must be a compositional Mixture")
        self.CompositionalMixture = value
        self.PropertyModel.fluid = value

    @property
    def property_model(self) -> CompositionalPropertyModel:
        """Pythonic alias for MRST's ``PropertyModel`` property."""
        return self.PropertyModel

    @property_model.setter
    def property_model(self, value: CompositionalPropertyModel) -> None:
        if not isinstance(value, CompositionalPropertyModel):
            raise TypeError("property_model must be a CompositionalPropertyModel")
        self.PropertyModel = value

    def getNumberOfComponents(self) -> int:
        """Return the number of mixture components."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before querying components")
        return self.CompositionalMixture.getNumberOfComponents()

    def getComponentNames(self) -> tuple[str, ...]:
        """Return component names in MRST order."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before querying components")
        return tuple(self.CompositionalMixture.names)

    @property
    def equilibrium_constant_functions(self):
        """Pythonic alias for MRST's ``equilibriumConstantFunctions`` field."""
        return self.equilibriumConstantFunctions

    @equilibrium_constant_functions.setter
    def equilibrium_constant_functions(self, value):
        if callable(value):
            self.equilibriumConstantFunctions = [value]
        else:
            functions = list(value)
            if not functions or not all(callable(fn) for fn in functions):
                raise TypeError(
                    "equilibrium_constant_functions must be a callable or a non-empty iterable of callables"
                )
            self.equilibriumConstantFunctions = functions

    @staticmethod
    def _as_2d(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 1-D or 2-D array")
        return arr

    @staticmethod
    def _broadcast_temperature(P: np.ndarray, T) -> np.ndarray:
        t = np.asarray(T, dtype=float)
        if t.ndim == 0:
            return np.full(P.shape, float(t))
        return np.broadcast_to(t.reshape(-1), P.shape).astype(float)

    @staticmethod
    def _expand_matrix_to_columns(z):
        if isinstance(z, (list, tuple)):
            return list(z), True
        arr = np.asarray(value(z), dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [arr[:, i] for i in range(arr.shape[1])], False

    @staticmethod
    def _call_equilibrium_function(fn, P, T, z_columns):
        for args in ((P, T, z_columns), (P, T), (P, z_columns), (P,)):
            try:
                return fn(*args)
            except TypeError:
                continue
        return fn(P, T, z_columns)

    def evaluateEquilibriumConstants(self, P, T, z):
        """Evaluate MRST-style user supplied equilibrium-constant functions."""
        if not self.equilibriumConstantFunctions:
            raise AssertionError(
                "Equilibrium constants not defined. This function is only valid when using K-value formulation."
            )
        P = np.asarray(value(P), dtype=float).reshape(-1)
        T = self._broadcast_temperature(P, T)
        z_columns, is_cell_input = self._expand_matrix_to_columns(z)

        functions = self.equilibriumConstantFunctions
        if len(functions) == 1:
            out = np.asarray(value(self._call_equilibrium_function(functions[0], P, T, z_columns)), dtype=float)
            if out.ndim == 2:
                out = np.clip(out, 1.0e-8, 1.0e8)
                if out.shape[0] != P.size:
                    raise ValueError(f"K has {out.shape[0]} rows, expected {P.size}")
                if is_cell_input:
                    return [out[:, i] for i in range(out.shape[1])]
                return out

        columns = []
        for fn in functions:
            col = np.asarray(value(self._call_equilibrium_function(fn, P, T, z_columns)), dtype=float).reshape(-1)
            col = np.broadcast_to(col, P.shape).astype(float)
            col = np.clip(col, 1.0e-8, 1.0e8)
            columns.append(col)
        if is_cell_input:
            return columns
        return np.column_stack(columns)

    evaluate_equilibrium_constants = evaluateEquilibriumConstants

    def setType(self, arg: str | int, c1: float | None = None, c2: float | None = None) -> "EquationOfStateModel":
        """Set EOS constants using MRST's string choices."""
        if isinstance(arg, str):
            key = arg.lower()
            if key in {"pr", "peng-robinson"}:
                self.eosA = 1.0 + SQRT2
                self.eosB = 1.0 - SQRT2
                self.omegaA = OMEGA_A
                self.omegaB = OMEGA_B
                self.eosType = 1
            elif key in {"srk", "soave-redlich-kwong"}:
                self.eosA = 0.0
                self.eosB = 1.0
                self.omegaA = 0.427480
                self.omegaB = 0.086640
                self.eosType = 2
            elif key in {"zj", "zudkevitch-joffe"}:
                self.eosA = 0.0
                self.eosB = 1.0
                self.omegaA = 0.427480
                self.omegaB = 0.086640
                self.eosType = 3
                raise NotImplementedError("Zudkevitch-Joffe EOS is not implemented")
            elif key in {"rk", "redlich-kwong"}:
                self.eosA = 0.0
                self.eosB = 1.0
                self.omegaA = 0.427480
                self.omegaB = 0.086640
                self.eosType = 4
            elif key in {"prcorr", "peng-robinson-corrected"}:
                self.eosA = 1.0 + SQRT2
                self.eosB = 1.0 - SQRT2
                self.omegaA = OMEGA_A
                self.omegaB = OMEGA_B
                self.eosType = 5
            else:
                raise ValueError(f"invalid EOS type {arg!r}")
        else:
            self.eosType = arg

        if c1 is not None:
            self.eosA = float(c1)
        if c2 is not None:
            self.eosB = float(c2)
        return self

    def shortname(self) -> str | int:
        """Return MRST's compact EOS type label."""
        return {
            1: "pr",
            2: "srk",
            3: "zj",
            4: "rk",
            5: "prcorr",
        }.get(self.eosType, self.eosType)

    def getEOSCoefficients(self) -> tuple[float, float]:
        """Return MRST's two cubic EOS coefficients ``m1`` and ``m2``."""
        return float(self.eosA), float(self.eosB)

    get_eos_coefficients = getEOSCoefficients

    def getCubicCoefficients(self, A, B):
        """Return PR/SRK cubic coefficients in MRST's ``[E2, E1, E0]`` order."""
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        m1 = float(self.eosA)
        m2 = float(self.eosB)
        e0 = -(A * B + m1 * m2 * B**2 * (B + 1.0))
        e1 = A - (m1 + m2 - m1 * m2) * B**2 - (m1 + m2) * B
        e2 = (m1 + m2 - 1.0) * B - 1.0
        return e2, e1, e0

    def solveCubicEOS(self, A, B):
        """Solve the cubic EOS and return positive real roots padded with NaN."""
        A = np.asarray(A, float).reshape(-1)
        B = np.asarray(B, float).reshape(-1)
        if self.shortname() == "pr":
            roots = _cubic_roots(A, B)
        else:
            e2, e1, e0 = self.getCubicCoefficients(A, B)
            roots = np.full((A.size, 3), np.nan)
            for i in range(A.size):
                row = np.roots([1.0, e2[i], e1[i], e0[i]])
                row = row[np.abs(row.imag) < 1e-10].real
                roots[i, :row.size] = np.sort(row)
        roots[roots <= 0.0] = np.nan
        return roots

    def checkZ(self, Z) -> None:
        """Raise if compressibility factors are negative or non-finite."""
        z = np.asarray(value(Z), dtype=float)
        n_negative = int(np.count_nonzero(z < 0.0))
        if n_negative:
            raise ValueError(f"{n_negative} negative Z-factors detected")
        n_bad = int(np.count_nonzero(~np.isfinite(z)))
        if n_bad:
            raise ValueError(f"{n_bad} non-finite Z-factors detected")

    check_z = checkZ

    def _roots_above_b(self, A, B):
        A = np.asarray(value(A), dtype=float).reshape(-1)
        B = np.asarray(value(B), dtype=float).reshape(-1)
        roots = self.solveCubicEOS(A, B)
        roots[roots < B[:, None]] = np.nan
        return roots

    def computeLiquidZ(self, A, B):
        """Pick the smallest physically valid Z-factor for the liquid phase."""
        roots = self._roots_above_b(A, B)
        with np.errstate(all="ignore"):
            z = np.nanmin(roots, axis=1)
        self.checkZ(z)
        return z

    compute_liquid_z = computeLiquidZ

    def computeVaporZ(self, A, B):
        """Pick the largest physically valid Z-factor for the vapour phase."""
        roots = self._roots_above_b(A, B)
        with np.errstate(all="ignore"):
            z = np.nanmax(roots, axis=1)
        self.checkZ(z)
        return z

    compute_vapor_z = computeVaporZ

    @staticmethod
    def _as_matrix(x) -> np.ndarray:
        if isinstance(x, (list, tuple)):
            return np.column_stack([np.asarray(value(item), dtype=float).reshape(-1) for item in x])
        return np.asarray(value(x), dtype=float)

    @staticmethod
    def _state_get(state, name: str, default=None):
        if isinstance(state, dict):
            return state.get(name, default)
        return getattr(state, name, default)

    @staticmethod
    def _state_has(state, name: str) -> bool:
        if isinstance(state, dict):
            return name in state
        return hasattr(state, name)

    @staticmethod
    def _state_set(state, name: str, item) -> None:
        if isinstance(state, dict):
            state[name] = item
        else:
            setattr(state, name, item)

    @staticmethod
    def _state_copy(item):
        if hasattr(item, "copy"):
            return item.copy()
        return np.array(item, copy=True)

    def validateState(self, state):
        """Fill missing EOS state fields using MRST's ``validateState`` rules."""
        pressure = np.asarray(value(self._state_get(state, "pressure")), dtype=float).reshape(-1)
        n_cell = pressure.size
        if n_cell == 0:
            raise ValueError("state.pressure must contain at least one cell")

        if not self._state_has(state, "L"):
            self._state_set(state, "L", np.full(n_cell, np.nan))

        if not self._state_has(state, "K"):
            if self._state_has(state, "x") and self._state_has(state, "y"):
                x = np.asarray(value(self._state_get(state, "x")), dtype=float)
                y = np.asarray(value(self._state_get(state, "y")), dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    K = y / x
            else:
                temperature = self._state_get(state, "T", self._state_get(state, "temperature"))
                if temperature is None:
                    raise KeyError("state must contain T or temperature when K is missing")
                K = estimateEquilibriumWilson(self, pressure, temperature)
            self._state_set(state, "K", K)

        if not self._state_has(state, "Z_V"):
            self._state_set(state, "Z_V", np.ones(n_cell))
        if not self._state_has(state, "Z_L"):
            self._state_set(state, "Z_L", np.ones(n_cell))

        if not self._state_has(state, "x"):
            components = self._state_get(state, "components")
            if components is None:
                raise KeyError("state must contain components when x is missing")
            self._state_set(state, "x", self._state_copy(components))
        if not self._state_has(state, "y"):
            components = self._state_get(state, "components")
            if components is None:
                raise KeyError("state must contain components when y is missing")
            self._state_set(state, "y", self._state_copy(components))
        return state

    validate_state = validateState

    def computeVapor(self, L=None, K=None, z=None):
        """Compute vapor composition from Rachford-Rice variables."""
        if K is None and z is None and L is not None and (
            isinstance(L, dict) or hasattr(L, "L")
        ):
            state = L
            return self.computeVapor(
                self._state_get(state, "L"),
                self._state_get(state, "K"),
                self._state_get(state, "components"),
            )
        L = np.asarray(value(L), dtype=float).reshape(-1, 1)
        K = np.asarray(value(K), dtype=float)
        z = np.asarray(value(z), dtype=float)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if L.shape[0] == 1 and z.shape[0] != 1:
            L = np.full((z.shape[0], 1), float(L[0, 0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            y = K * z / (L + (1.0 - L) * K)
            y[~np.isfinite(K)] = z[~np.isfinite(K)]
            y = y / np.sum(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError("non-finite vapor composition computed")
        return y

    compute_vapor = computeVapor

    def solveRachfordRice(self, L, K, z):
        """Solve Rachford-Rice and return MRST's liquid molar fraction ``L``."""
        del L
        vapor_fraction = solveRachfordRiceVLE(z, K)
        return 1.0 - vapor_fraction

    solve_rachford_rice = solveRachfordRice

    def computeLiquid(self, L=None, K=None, z=None):
        """Compute liquid composition from Rachford-Rice variables."""
        if K is None and z is None and L is not None and (
            isinstance(L, dict) or hasattr(L, "L")
        ):
            state = L
            return self.computeLiquid(
                self._state_get(state, "L"),
                self._state_get(state, "K"),
                self._state_get(state, "components"),
            )
        L = np.asarray(value(L), dtype=float).reshape(-1, 1)
        K = np.asarray(value(K), dtype=float)
        z = np.asarray(value(z), dtype=float)
        if K.ndim == 1:
            K = K.reshape(1, -1)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if L.shape[0] == 1 and z.shape[0] != 1:
            L = np.full((z.shape[0], 1), float(L[0, 0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            tmp = (1.0 - L) * K
            x = z / (L + tmp)
            x[~np.isfinite(K)] = 0.0
            sv = np.sum(x, axis=1, keepdims=True)
            x = x / sv
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite liquid composition computed")
        return x

    compute_liquid = computeLiquid

    def singlePhaseLabel(self, p, T, z):
        """Label stable single-phase cells using MRST's Li critical-temperature rule."""
        del p
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before phase labeling")
        T = np.asarray(value(T), dtype=float).reshape(-1)
        z = np.asarray(value(z), dtype=float)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if T.size == 1 and z.shape[0] != 1:
            T = np.full(z.shape[0], float(T[0]))
        Vc = np.asarray(self.CompositionalMixture.Vcrit, dtype=float)
        Tc = np.asarray(self.CompositionalMixture.Tcrit, dtype=float)
        Vz = z * Vc[None, :]
        Tc_est = np.sum(Vz * Tc[None, :], axis=1) / np.sum(Vz, axis=1)
        return (T < Tc_est).astype(float)

    single_phase_label = singlePhaseLabel

    def estimateSinglePhaseState(self, p, T, z, L, stable):
        """Estimate liquid/vapor labels for cells MRST marks stable."""
        L = np.asarray(value(L), dtype=float).reshape(-1).copy()
        stable = np.asarray(value(stable), dtype=bool).reshape(-1)
        if not np.any(stable):
            return L
        z_arr = np.asarray(value(z), dtype=float)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
        p_arr = np.asarray(value(p), dtype=float).reshape(-1)
        T_arr = np.asarray(value(T), dtype=float).reshape(-1)
        K = estimateEquilibriumWilson(self, p_arr[stable], T_arr[stable])
        L0 = np.full(np.count_nonzero(stable), 0.5)
        local = self.solveRachfordRice(L0, K, z_arr[stable, :])
        out = L[stable]
        out[local > 0.5] = 1.0
        out[local <= 0.5] = 0.0
        L[stable] = out
        return L

    estimate_single_phase_state = estimateSinglePhaseState

    def performPhaseStabilityTest(self, P, T, z, K=None):
        """Perform MRST-style stability testing and return ``stable, x, y``."""
        z_arr = np.asarray(value(z), dtype=float)
        if z_arr.size == 0:
            ncomp = z_arr.shape[1] if z_arr.ndim == 2 else self.getNumberOfComponents()
            empty = np.zeros((0, ncomp))
            return np.array([], dtype=bool), empty, empty.copy()
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)

        if not self.equilibriumConstantFunctions:
            return phaseStabilityTest(self, z_arr, P, T, K=K)

        K_eval = self.evaluateEquilibriumConstants(P, T, z_arr)
        check_stability = getattr(self.PropertyModel, "checkStabilityFunction", None)
        if callable(check_stability):
            stable, L = check_stability(P, T, [z_arr[:, i] for i in range(z_arr.shape[1])])
            stable = np.asarray(value(stable), dtype=bool).reshape(-1)
            L = np.asarray(value(L), dtype=float).reshape(-1)
        else:
            L = self.solveRachfordRice(None, K_eval, z_arr)
            stable = (np.abs(L - 1.0) <= 1.0e-10) | (L <= 1.0e-10)
        x = self.computeLiquid(L, K_eval, z_arr)
        y = self.computeVapor(L, K_eval, z_arr)
        return stable, x, y

    perform_phase_stability_test = performPhaseStabilityTest

    def setFlag(self, state, pureLiquid=None, pureVapor=None):
        """Set MRST phase flags: 0 two-phase, 1 pure liquid, 2 pure vapor."""
        if pureVapor is None:
            L = np.asarray(value(self._state_get(state, "L")), dtype=float)
            pureVapor = L == 0
            if pureLiquid is None:
                pureLiquid = L == 1
        pureLiquid = np.asarray(value(pureLiquid), dtype=bool)
        pureVapor = np.asarray(value(pureVapor), dtype=bool).copy()
        pureVapor[pureLiquid] = False
        flag = pureLiquid.astype(int) + 2 * pureVapor.astype(int)
        self._state_set(state, "flag", flag)
        return state

    set_flag = setFlag

    def getSinglePhase(self, state):
        """Return cells marked pure liquid or pure vapor."""
        if not self._state_has(state, "flag"):
            state = self.setFlag(state)
        return np.asarray(value(self._state_get(state, "flag"))) != 0

    get_single_phase = getSinglePhase

    def getFlag(self, state):
        """Return ``isLiquid, isVapor, is2ph`` boolean masks."""
        if not self._state_has(state, "flag"):
            state = self.setFlag(state)
        flag = np.asarray(value(self._state_get(state, "flag")))
        is_liquid = flag == 1
        is_vapor = flag == 2
        is_2ph = ~(is_liquid | is_vapor)
        return is_liquid, is_vapor, is_2ph

    get_flag = getFlag

    def getTwoPhaseFlag(self, state):
        """Return cells marked as two-phase."""
        if not self._state_has(state, "flag"):
            state = self.setFlag(state)
        return np.asarray(value(self._state_get(state, "flag"))) == 0

    get_two_phase_flag = getTwoPhaseFlag

    def updateAfterConvergence(self, state0, state, dt, drivingForces=None):
        """Finalize EOS phase labels/compositions after a converged step."""
        del state0, dt, drivingForces
        state = self.setFlag(state)
        changed = False
        if not self._state_has(state, "x"):
            self._state_set(state, "x", self.computeLiquid(state))
            changed = True
        if not self._state_has(state, "y"):
            self._state_set(state, "y", self.computeVapor(state))
            changed = True
        if changed:
            L = np.asarray(value(self._state_get(state, "L")), dtype=float).reshape(-1)
            single_phase = (L == 1.0) | (L == 0.0)
            if np.any(single_phase):
                components = np.asarray(value(self._state_get(state, "components")), dtype=float)
                x = np.asarray(value(self._state_get(state, "x")), dtype=float).copy()
                y = np.asarray(value(self._state_get(state, "y")), dtype=float).copy()
                x[single_phase, :] = components[single_phase, :]
                y[single_phase, :] = components[single_phase, :]
                self._state_set(state, "x", x)
                self._state_set(state, "y", y)
        return state, {}

    update_after_convergence = updateAfterConvergence

    def computeSaturations(self, p, T, rhoL, rhoV, x, y, L, Z_L, Z_V):
        """Convert molar phase split to liquid/vapor saturations."""
        del rhoL, rhoV
        volume_shift = getattr(self.PropertyModel, "volumeShift", None)
        return phase_saturations(
            self.CompositionalMixture,
            p,
            T,
            L,
            x,
            y,
            Z_L,
            Z_V,
            volume_shift=volume_shift,
        )

    compute_saturations = computeSaturations

    def getMoleFraction(self, massfraction):
        """Convert mass fractions to mole fractions using mixture molar masses."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before converting fractions")
        return mole_fraction(self.CompositionalMixture, massfraction)

    get_mole_fraction = getMoleFraction

    def getMassFraction(self, molfraction):
        """Convert mole fractions to mass fractions using mixture molar masses."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before converting fractions")
        return mass_fraction(self.CompositionalMixture, molfraction)

    get_mass_fraction = getMassFraction

    def computeFugacity(self, p, x, Z, A, B, Si, Bi):
        """Compute fugacity and log-fugacity coefficient from mixed EOS inputs."""
        m1, m2 = self.getEOSCoefficients()
        p = np.asarray(value(p), dtype=float).reshape(-1)
        x = self._as_matrix(x)
        Z = np.asarray(value(Z), dtype=float).reshape(-1)
        A = np.asarray(value(A), dtype=float).reshape(-1)
        B = np.asarray(value(B), dtype=float).reshape(-1)
        Si = self._as_matrix(Si)
        Bi = self._as_matrix(Bi)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if Si.ndim == 1:
            Si = Si.reshape(-1, 1)
        if Bi.ndim == 1:
            Bi = Bi.reshape(-1, 1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            a1 = -np.log(Z - B)
            b1 = np.log((Z + m2 * B) / (Z + m1 * B)) * (A / ((m1 - m2) * B))
            b2 = (Z - 1.0) / B
            phi = (
                a1[:, None]
                + b1[:, None] * (2.0 * Si / A[:, None] - Bi / B[:, None])
                + Bi * b2[:, None]
            )
            f = np.exp(phi) * (p[:, None] * x)
        return f, phi

    compute_fugacity = computeFugacity

    def computeCompressibilityZ(self, p, xy, A, B, Si, Bi, isLiquid):
        """Select phase compressibility factors following MRST's EOS model."""
        p = np.asarray(value(p), dtype=float).reshape(-1)
        xy = self._as_matrix(xy)
        A = np.asarray(value(A), dtype=float).reshape(-1)
        B = np.asarray(value(B), dtype=float).reshape(-1)
        Si = self._as_matrix(Si)
        Bi = self._as_matrix(Bi)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 1)

        if not self.selectGibbsMinimum:
            return self.computeLiquidZ(A, B) if bool(isLiquid) else self.computeVaporZ(A, B)

        roots = self.solveCubicEOS(A, B)
        roots[roots < B[:, None]] = np.nan
        candidates = np.isfinite(roots)
        num_roots = np.sum(candidates, axis=1)
        with np.errstate(all="ignore"):
            z = np.nanmax(roots, axis=1)
        multiple = num_roots > 1
        if np.any(multiple):
            z_max = np.nanmax(roots[multiple], axis=1)
            z_min = np.nanmin(roots[multiple], axis=1)
            _, phi_max = self.computeFugacity(
                p[multiple], xy[multiple], z_max, A[multiple], B[multiple], Si[multiple], Bi[multiple]
            )
            _, phi_min = self.computeFugacity(
                p[multiple], xy[multiple], z_min, A[multiple], B[multiple], Si[multiple], Bi[multiple]
            )
            g_max = np.sum(phi_max * xy[multiple], axis=1)
            g_min = np.sum(phi_min * xy[multiple], axis=1)
            z[multiple] = np.where(g_min < g_max, z_min, z_max)
        self.checkZ(z)
        return z

    compute_compressibility_z = computeCompressibilityZ

    def getReducedPT(self, P, T, useCell: bool = False):
        """Return reduced pressure and temperature for each component."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before querying reduced variables")
        P = np.asarray(value(P), dtype=float).reshape(-1)
        T = np.asarray(value(T), dtype=float).reshape(-1)
        if T.size == 1 and P.size != 1:
            T = np.full(P.size, float(T[0]))
        pr = P[:, None] / np.asarray(self.CompositionalMixture.Pcrit, dtype=float)[None, :]
        tr = T[:, None] / np.asarray(self.CompositionalMixture.Tcrit, dtype=float)[None, :]
        if useCell:
            return [pr[:, ix] for ix in range(pr.shape[1])], [tr[:, ix] for ix in range(tr.shape[1])]
        return pr, tr

    get_reduced_pt = getReducedPT

    def getMixingParameters(self, P, T, acf=None, useCell: bool = False):
        """Return pure-component EOS mixing parameters ``A_ij`` and ``Bi``."""
        if self.CompositionalMixture is None:
            raise ValueError("CompositionalMixture must be set before querying mixing parameters")
        if acf is None:
            acf = self.CompositionalMixture.acentricFactors
        acf = np.asarray(acf, dtype=float).reshape(-1)
        pr, tr = self.getReducedPT(P, T, useCell=False)
        if self.eosType in (1, 5):
            base = 0.37464 + 1.54226 * acf - 0.26992 * acf**2
            if self.eosType == 5:
                heavy = acf > 0.49
                corrected = 0.379642 + 1.48503 * acf - 0.164423 * acf**2 + 0.016666 * acf**3
                base = np.where(heavy, corrected, base)
            oa = self.omegaA * (1.0 + (1.0 - np.sqrt(tr)) * base[None, :]) ** 2
        elif self.eosType == 2:
            base = 0.48 + 1.574 * acf - 0.176 * acf**2
            oa = self.omegaA * (1.0 + (1.0 - np.sqrt(tr)) * base[None, :]) ** 2
        elif self.eosType == 3:
            raise NotImplementedError("Zudkevitch-Joffe EOS is not implemented")
        elif self.eosType == 4:
            oa = self.omegaA * tr ** -0.5
        else:
            raise ValueError(f"unknown EOS type {self.eosType!r}")

        ob = np.full_like(oa, float(self.omegaB))
        sai = np.sqrt(oa * pr) / tr
        bi = ob * pr / tr
        bic = self.CompositionalMixture.getBinaryInteraction()
        aij = [
            sai[:, :, None] * sai[:, None, :] * (1.0 - bic)[None, :, :]
        ][0]
        if useCell:
            aij_cells = [
                [aij[:, i, j] for j in range(aij.shape[2])]
                for i in range(aij.shape[1])
            ]
            return aij_cells, [bi[:, ix] for ix in range(bi.shape[1])]
        return aij, bi

    get_mixing_parameters = getMixingParameters

    def getPhaseMixCoefficients(self, x, A_ij, Bi):
        """Return phase mixture coefficients ``Si``, ``A`` and ``B``."""
        x = self._as_matrix(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if isinstance(A_ij, list):
            aij = np.stack([
                np.stack([np.asarray(value(A_ij[i][j]), dtype=float).reshape(-1)
                          for j in range(len(A_ij[i]))], axis=1)
                for i in range(len(A_ij))
            ], axis=1)
        else:
            aij = np.asarray(value(A_ij), dtype=float)
        bi = self._as_matrix(Bi)
        if bi.ndim == 1:
            bi = bi.reshape(-1, 1)
        b_mix = np.sum(x * bi, axis=1)
        si = np.einsum("cij,cj->ci", aij, x)
        a_mix = np.einsum("ci,cij,cj->c", x, aij, x)
        return si, a_mix, b_mix

    get_phase_mix_coefficients = getPhaseMixCoefficients

    def getMixtureFugacityCoefficients(self, P, T, x, y, acf=None):
        """Return MRST's phase fugacity coefficient ingredients."""
        aij, bi = self.getMixingParameters(P, T, acf, useCell=False)
        si_l, a_l, b_l = self.getPhaseMixCoefficients(x, aij, bi)
        si_v, a_v, b_v = self.getPhaseMixCoefficients(y, aij, bi)
        return si_l, si_v, a_l, a_v, b_l, b_v, bi

    get_mixture_fugacity_coefficients = getMixtureFugacityCoefficients

    def setZDerivatives(self, Z, A, B, *_, **__):
        """MRST hook for AD Z derivatives; value-only PyMRST returns ``Z``."""
        del A, B
        return Z

    set_z_derivatives = setZDerivatives

    def get_compressibility_and_fugacity(
        self,
        P,
        T,
        x,
        y,
        z=None,
        Z_L=None,
        Z_V=None,
        *args,
        **kwargs,
    ):
        """Return ``Z_L, Z_V, f_L, f_V`` like MRST's cubic EOS model."""
        del z
        acf = self.CompositionalMixture.acentricFactors
        si_l, si_v, a_l, a_v, b_l, b_v, bi = self.getMixtureFugacityCoefficients(P, T, x, y, acf)
        if Z_L is None or (np.asarray(Z_L).size == 0):
            Z_L = self.computeCompressibilityZ(P, x, a_l, b_l, si_l, bi, True)
        if Z_V is None or (np.asarray(Z_V).size == 0):
            Z_V = self.computeCompressibilityZ(P, y, a_v, b_v, si_v, bi, False)
        Z_L = self.setZDerivatives(Z_L, a_l, b_l, *args, **kwargs)
        Z_V = self.setZDerivatives(Z_V, a_v, b_v, *args, **kwargs)
        f_L, _ = self.computeFugacity(P, x, Z_L, a_l, b_l, si_l, bi)
        f_V, _ = self.computeFugacity(P, y, Z_V, a_v, b_v, si_v, bi)
        return Z_L, Z_V, f_L, f_V

    def getCompressibilityAndFugacity(
        self,
        P,
        T,
        x,
        y,
        z=None,
        Z_L=None,
        Z_V=None,
        *args,
        **kwargs,
    ):
        return self.get_compressibility_and_fugacity(P, T, x, y, z, Z_L, Z_V, *args, **kwargs)


__all__ = [
    "EquationOfStateModel",
    "phase_fraction_derivatives",
    "flash_ad",
    "equations_equilibrium",
    "phase_compressibility_ad",
]
