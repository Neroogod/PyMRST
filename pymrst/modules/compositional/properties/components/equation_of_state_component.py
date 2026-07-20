"""EOS component object (MRST `EquationOfStateComponent`)."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pymrst.modules.compositional.eos import Mixture


def _as_phase_matrix(values, nph: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("phase property matrix must be two-dimensional")
    if nph is not None and arr.shape[1] != nph:
        raise ValueError(f"expected {nph} phases, got {arr.shape[1]}")
    return arr


@dataclass
class EquationOfStateComponent:
    """Component-level property evaluator for an EOS component.

    MRST stores one object per component. The object pulls its row from
    `ComponentPhaseMassFractions` and combines it with phase densities.
    """

    name: str
    pressure: float
    T: float
    component_index: int
    surface_phase_mass_fractions: Sequence[float]
    surface_phase_density_pure: Sequence[float]
    molar_mass: float

    def __post_init__(self) -> None:
        self.surface_phase_mass_fractions = np.asarray(self.surface_phase_mass_fractions, dtype=float)
        self.surface_phase_density_pure = np.asarray(self.surface_phase_density_pure, dtype=float)
        if self.component_index < 0:
            raise ValueError("component_index is zero-based and must be non-negative")

    @property
    def componentIndex(self) -> int:
        """MRST-compatible alias for ``component_index``."""
        return self.component_index

    @componentIndex.setter
    def componentIndex(self, value: int):
        value = int(value)
        if value < 0:
            raise ValueError("componentIndex/component_index must be non-negative")
        self.component_index = value

    @property
    def surfacePhaseMassFractions(self) -> np.ndarray:
        """MRST-compatible alias for ``surface_phase_mass_fractions``."""
        return self.surface_phase_mass_fractions

    @surfacePhaseMassFractions.setter
    def surfacePhaseMassFractions(self, value):
        self.surface_phase_mass_fractions = np.asarray(value, dtype=float)

    @property
    def surfacePhaseDensityPure(self) -> np.ndarray:
        """MRST-compatible alias for ``surface_phase_density_pure``."""
        return self.surface_phase_density_pure

    @surfacePhaseDensityPure.setter
    def surfacePhaseDensityPure(self, value):
        self.surface_phase_density_pure = np.asarray(value, dtype=float)

    @property
    def molarMass(self) -> float:
        """MRST-compatible alias for ``molar_mass``."""
        return self.molar_mass

    @molarMass.setter
    def molarMass(self, value: float):
        self.molar_mass = float(value)

    def get_phase_composition(self, mass_fractions) -> list[np.ndarray]:
        """Return this component's mass fraction in each phase."""
        mf = _as_phase_matrix(mass_fractions)
        if self.component_index >= mf.shape[0]:
            raise IndexError("component_index exceeds component count")
        return [mf[self.component_index, ph] for ph in range(mf.shape[1])]

    def getPhaseComposition(self, model=None, state=None, mass_fractions=None):
        """MRST-compatible alias for phase mass fractions of this component."""
        if mass_fractions is None:
            if model is None:
                raise TypeError("mass_fractions or model must be provided")
            if hasattr(model, "getProps"):
                mass_fractions = model.getProps(state, "ComponentPhaseMassFractions")
            elif hasattr(model, "get_props"):
                mass_fractions = model.get_props(state, "ComponentPhaseMassFractions")
            else:
                raise TypeError("model must implement getProps/get_props for MRST-style calls")
        return self.get_phase_composition(mass_fractions)

    def get_component_density(self, density, mass_fractions) -> list[np.ndarray]:
        """Return phase density times this component's phase mass fraction."""
        rho = np.asarray(density, dtype=float)
        fractions = self.get_phase_composition(mass_fractions)
        if rho.size != len(fractions):
            raise ValueError("density length must match number of phases")
        return [rho[ph] * fractions[ph] for ph in range(len(fractions))]

    def getComponentDensity(self, model=None, state=None, extra=None, density=None, mass_fractions=None):
        """MRST-compatible alias for component phase density."""
        if mass_fractions is None:
            if model is None:
                raise TypeError("mass_fractions or model must be provided")
            if hasattr(model, "getProps"):
                mass_fractions = model.getProps(state, "ComponentPhaseMassFractions")
            elif hasattr(model, "get_props"):
                mass_fractions = model.get_props(state, "ComponentPhaseMassFractions")
            else:
                raise TypeError("model must implement getProps/get_props for MRST-style calls")
        if density is None:
            if extra is not None and hasattr(extra, "rho"):
                density = extra.rho
            elif isinstance(extra, dict) and "rho" in extra:
                density = extra["rho"]
            elif model is not None and hasattr(model, "getProps"):
                density = model.getProps(state, "Density")
            elif model is not None and hasattr(model, "get_props"):
                density = model.get_props(state, "Density")
            else:
                raise TypeError("density, extra.rho, or model Density prop must be provided")
        return self.get_component_density(density, mass_fractions)

    def get_phase_composition_surface(self, nph: int | None = None) -> list[float]:
        """Return stored surface mass fractions for each phase."""
        fractions = np.asarray(self.surface_phase_mass_fractions, dtype=float)
        if nph is not None and fractions.size != nph:
            raise ValueError(f"expected {nph} phases, got {fractions.size}")
        return [float(v) for v in fractions]

    def getPhaseCompositionSurface(self, model=None, state=None, pressure=None, temperature=None):
        """MRST-compatible camelCase alias."""
        nph = None
        if model is not None:
            if hasattr(model, "get_number_of_phases"):
                nph = model.get_number_of_phases()
            elif hasattr(model, "getNumberOfPhases"):
                nph = model.getNumberOfPhases()
            else:
                nph = getattr(model, "n_phases", None)
        return self.get_phase_composition_surface(nph)

    def get_pure_phase_density_surface(self, pressure, temperature) -> list[np.ndarray]:
        """Return pure-component surface phase densities with ideal-gas scaling."""
        pressure_arr = np.asarray(pressure, dtype=float)
        temperature_arr = np.asarray(temperature, dtype=float)
        scale = (pressure_arr / float(self.pressure)) * (float(self.T) / temperature_arr)
        rho = np.asarray(self.surface_phase_density_pure, dtype=float)
        return [rho_ph * scale for rho_ph in rho]

    def getPurePhaseDensitySurface(self, model=None, state=None, pressure=None, temperature=None):
        """MRST-compatible camelCase alias."""
        p = self.pressure if pressure is None else pressure
        t = self.T if temperature is None else temperature
        return self.get_pure_phase_density_surface(p, t)

    def get_phase_component_fraction_injection(
        self,
        mix: Mixture,
        components,
        nph: int = 2,
        liquid_index: int = 0,
        vapor_index: int = 1,
    ) -> list[np.ndarray | None]:
        """Return injected mass fraction for this component in liquid/vapour phases."""
        comp = np.atleast_2d(np.asarray(components, dtype=float))
        denom = comp @ mix.molarMass
        mass_fraction = comp * mix.molarMass[None, :] / denom[:, None]
        ci = mass_fraction[:, self.component_index]
        out: list[np.ndarray | None] = [None] * nph
        if np.any(ci != 0.0):
            out[liquid_index] = ci
            out[vapor_index] = ci
        return out

    def getPhaseComponentFractionInjection(self, model=None, state=None, force=None):
        """MRST-compatible alias for injected component mass fractions."""
        if force is None:
            raise TypeError("force with components must be provided")
        if hasattr(force, "components"):
            components = force.components
        elif isinstance(force, dict) and "components" in force:
            components = force["components"]
        else:
            components = [getattr(item, "components", item["components"]) for item in force]
        mix = None
        if model is not None:
            mix = getattr(model, "mix", None)
            if mix is None:
                mix = getattr(model, "mixture", None)
            eos = getattr(model, "EOSModel", None)
            if mix is None and eos is not None:
                mix = getattr(eos, "mix", None) or getattr(eos, "mixture", None)
        if mix is None:
            raise TypeError("model must expose mix/mixture/EOSModel.mix for injection mass fractions")
        nph = 2
        if model is not None:
            if hasattr(model, "getNumberOfPhases"):
                nph = int(model.getNumberOfPhases())
            elif hasattr(model, "get_number_of_phases"):
                nph = int(model.get_number_of_phases())
            else:
                nph = int(getattr(model, "n_phases", nph))
        liquid_index = 0
        vapor_index = 1
        if model is not None:
            if hasattr(model, "getLiquidIndex"):
                liquid_index = int(model.getLiquidIndex())
            elif hasattr(model, "get_liquid_index"):
                liquid_index = int(model.get_liquid_index())
            if hasattr(model, "getVaporIndex"):
                vapor_index = int(model.getVaporIndex())
            elif hasattr(model, "get_vapor_index"):
                vapor_index = int(model.get_vapor_index())
        return self.get_phase_component_fraction_injection(
            mix,
            components,
            nph=nph,
            liquid_index=liquid_index,
            vapor_index=vapor_index,
        )


if __name__ == "__main__":
    from pymrst.modules.compositional.fluids import get_benchmark_mixture

    mixture, _ = get_benchmark_mixture("simple")
    component = EquationOfStateComponent(
        "Methane",
        pressure=101325.0,
        T=288.15,
        component_index=0,
        surface_phase_mass_fractions=[0.1, 0.9],
        surface_phase_density_pure=[450.0, 0.7],
        molar_mass=mixture.molarMass[0],
    )
    mass_fractions = np.array([[0.2, 0.8], [0.8, 0.2]])
    density = np.array([500.0, 10.0])
    assert component.get_phase_composition(mass_fractions) == [0.2, 0.8]
    assert component.get_component_density(density, mass_fractions) == [100.0, 8.0]
    assert component.getPhaseComposition(mass_fractions=mass_fractions) == [0.2, 0.8]
    assert component.getComponentDensity(density=density, mass_fractions=mass_fractions) == [100.0, 8.0]
    injected = component.get_phase_component_fraction_injection(mixture, [[0.5, 0.3, 0.2]])
    assert injected[0].shape == (1,)
    assert injected[1].shape == (1,)
