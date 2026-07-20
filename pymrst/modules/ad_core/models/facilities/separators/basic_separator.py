"""Basic separator base class.

This mirrors MRST's `BasicSeparator`: it stores separator pressure and
temperature and defines the mass/mole stream API used by `SeparatorGroup`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeparatorMassResult:
    """Mass-mode split from a basic separator."""

    phase_mass_streams: list[np.ndarray]
    # MRST calls these massFractions, but BasicSeparator stores each component's
    # separated mass stream in each phase rather than normalized fractions.
    phase_mass_fractions: list[np.ndarray]
    densities: list[np.ndarray]

    def __iter__(self):
        """Unpack like MRST's three-output separateComponentMassStream."""
        yield self.phase_mass_streams
        yield self.phase_mass_fractions
        yield self.densities


@dataclass
class SeparatorMoleResult:
    """Mole-mode split from a basic separator."""

    phase_mole_streams: list[np.ndarray]
    phase_mole_fractions: list[np.ndarray]
    molar_densities: list[np.ndarray]
    mass_result: SeparatorMassResult

    def __iter__(self):
        """Unpack like MRST's three-output separateComponentMoleStream."""
        yield self.phase_mole_streams
        yield self.phase_mole_fractions
        yield self.molar_densities


@dataclass
class BasicSeparator:
    """Pressure/temperature separator base."""

    pressure: float = np.nan
    T: float = np.nan

    def separate_component_mass_stream(self, model, mass_stream):
        """Separate a component mass stream using component surface properties.

        This ports MRST ``BasicSeparator.separateComponentMassStream``: each
        component independently reports the fraction that belongs in each
        surface phase and its pure phase density at the separator pressure and
        temperature. The returned ``phase_mass_fractions`` follow MRST's
        ``massFractions`` output literally: they are per-component separated
        mass streams, not normalized mass fractions.
        """
        components = _components(model)
        ncomp = len(components)
        stream, _ = _as_mass_stream_array(mass_stream, ncomp)
        nph = _number_of_phases(model, components)

        component_phase_streams = [
            np.zeros((stream.shape[0], ncomp), dtype=float)
            for _ in range(nph)
        ]
        phase_mass_streams = [np.zeros(stream.shape[0], dtype=float) for _ in range(nph)]
        density_contrib = [np.zeros(stream.shape[0], dtype=float) for _ in range(nph)]
        density_terms = []

        for c, comp in enumerate(components):
            frac = _component_phase_composition(comp, model, self.pressure, self.T, nph)
            rho = _component_phase_density(comp, model, self.pressure, self.T, nph, stream.shape[0])
            density_terms = rho
            for ph in range(nph):
                result = stream[:, c] * frac[ph]
                component_phase_streams[ph][:, c] = result
                phase_mass_streams[ph] += result

        # Match MRST 2024a BasicSeparator literally: its density loop reuses
        # the final component object from the phase-split loop.
        for ph in range(nph):
            density_contrib[ph] = component_phase_streams[ph].sum(axis=1) * density_terms[ph]

        mass_fractions = []
        densities = []
        for ph in range(nph):
            mass_fractions.append(component_phase_streams[ph])
            densities.append(density_contrib[ph])

        return SeparatorMassResult(
            phase_mass_streams=phase_mass_streams,
            phase_mass_fractions=mass_fractions,
            densities=densities,
        )

    def separate_component_mole_stream(self, model, mole_stream):
        """Separate a component mole stream using surface component data.

        MRST's ``BasicSeparator`` is component-property based rather than EOS
        based.  The mole-stream variant is therefore the mass-stream split in a
        different basis: convert each component mole stream to mass, apply
        ``separate_component_mass_stream``, then convert the separated phase
        component streams back to moles and normalize phase mole fractions.
        """
        components = _components(model)
        ncomp = len(components)
        stream, _ = _as_mass_stream_array(mole_stream, ncomp)
        molar_mass = _component_molar_masses(model, components)
        mass_result = self.separate_component_mass_stream(model, stream * molar_mass[None, :])

        phase_mole_components: list[np.ndarray] = []
        phase_mole_streams: list[np.ndarray] = []
        phase_mole_fractions: list[np.ndarray] = []
        molar_densities: list[np.ndarray] = []
        for component_mass, density in zip(
            mass_result.phase_mass_fractions,
            mass_result.densities,
        ):
            component_moles = component_mass / molar_mass[None, :]
            total_moles = component_moles.sum(axis=1)
            fractions = np.zeros_like(component_moles)
            active = np.abs(total_moles) > 0.0
            fractions[active] = component_moles[active] / total_moles[active, None]
            phase_molar_mass = fractions @ molar_mass
            rho_molar = np.zeros_like(total_moles)
            valid = np.isfinite(density) & (phase_molar_mass > 0.0)
            rho_molar[valid] = density[valid] / phase_molar_mass[valid]
            phase_mole_components.append(component_moles)
            phase_mole_streams.append(total_moles)
            phase_mole_fractions.append(fractions)
            molar_densities.append(rho_molar)
        return SeparatorMoleResult(
            phase_mole_streams=phase_mole_streams,
            phase_mole_fractions=phase_mole_fractions,
            molar_densities=molar_densities,
            mass_result=mass_result,
        )

    # MRST-compatible aliases.
    separateComponentMassStream = separate_component_mass_stream
    separateComponentMoleStream = separate_component_mole_stream


def _components(model) -> list:
    components = getattr(model, "components", None)
    if components is None:
        components = getattr(model, "Components", None)
    if components is None:
        raise TypeError("BasicSeparator requires a model with components/Components")
    return list(components)


def _as_mass_stream_array(stream, ncomp: int) -> tuple[np.ndarray, bool]:
    if isinstance(stream, (list, tuple)):
        arr = np.column_stack([np.asarray(s, dtype=float).reshape(-1) for s in stream])
    else:
        arr = np.asarray(stream, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError("stream must be a vector, matrix, or component list")
    if arr.shape[1] != ncomp:
        raise ValueError(f"expected {ncomp} components, got {arr.shape[1]}")
    return arr, arr.shape[0] == 1


def _number_of_phases(model, components: list) -> int:
    if hasattr(model, "get_number_of_phases"):
        return int(model.get_number_of_phases())
    if hasattr(model, "getNumberOfPhases"):
        return int(model.getNumberOfPhases())
    nph = getattr(model, "n_phases", None)
    if nph is not None:
        return int(nph)
    nph = getattr(model, "nph", None)
    if nph is not None:
        return int(nph)
    first = components[0]
    if hasattr(first, "surface_phase_mass_fractions"):
        return int(np.asarray(first.surface_phase_mass_fractions).size)
    return 2


def _component_molar_masses(model, components: list) -> np.ndarray:
    for name in ("molarMass", "molar_mass", "molecular_weight"):
        values = getattr(model, name, None)
        if values is not None:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == len(components):
                return arr
    masses = []
    for comp in components:
        value = getattr(comp, "molar_mass", getattr(comp, "molarMass", None))
        if value is None:
            raise TypeError("component must expose molar_mass/molarMass for mole streams")
        masses.append(float(value))
    return np.asarray(masses, dtype=float)


def _component_phase_composition(comp, model, pressure, temperature, nph: int) -> list[np.ndarray]:
    if hasattr(comp, "get_phase_composition_surface"):
        out = comp.get_phase_composition_surface(nph)
    elif hasattr(comp, "getPhaseCompositionSurface"):
        out = comp.getPhaseCompositionSurface(model, None, pressure, temperature)
    else:
        raise TypeError("component must implement surface phase composition")
    vals = [np.asarray(v, dtype=float).reshape(-1) for v in out]
    if len(vals) != nph:
        raise ValueError(f"component returned {len(vals)} phase fractions, expected {nph}")
    return vals


def _component_phase_density(comp, model, pressure, temperature, nph: int, nstream: int) -> list[np.ndarray]:
    if hasattr(comp, "get_pure_phase_density_surface"):
        out = comp.get_pure_phase_density_surface(pressure, temperature)
    elif hasattr(comp, "getPurePhaseDensitySurface"):
        out = comp.getPurePhaseDensitySurface(model, None, pressure, temperature)
    else:
        raise TypeError("component must implement pure surface phase density")
    vals = []
    for v in out:
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(nstream, arr[0], dtype=float)
        elif arr.size != nstream:
            raise ValueError("component density must be scalar or one value per stream")
        vals.append(arr)
    if len(vals) != nph:
        raise ValueError(f"component returned {len(vals)} phase densities, expected {nph}")
    return vals


__all__ = ["BasicSeparator", "SeparatorMassResult", "SeparatorMoleResult"]


if __name__ == "__main__":
    from pymrst.modules.compositional.properties.components import EquationOfStateComponent
    from pymrst.utils.units import ATM

    class FakeModel:
        n_phases = 2

        def __init__(self):
            self.components = [
                EquationOfStateComponent(
                    "Light",
                    pressure=ATM,
                    T=300.0,
                    component_index=0,
                    surface_phase_mass_fractions=[0.2, 0.8],
                    surface_phase_density_pure=[700.0, 5.0],
                    molar_mass=0.016,
                ),
                EquationOfStateComponent(
                    "Heavy",
                    pressure=ATM,
                    T=300.0,
                    component_index=1,
                    surface_phase_mass_fractions=[0.9, 0.1],
                    surface_phase_density_pure=[850.0, 15.0],
                    molar_mass=0.100,
                ),
            ]

        def get_number_of_phases(self):
            return self.n_phases

    def test_basic_separator_mass_stream_split():
        sep = BasicSeparator(pressure=ATM, T=300.0)
        model = FakeModel()
        stream = np.array([2.0, 3.0])
        out = sep.separate_component_mass_stream(model, stream)
        streams, fractions, densities = out
        assert len(out.phase_mass_streams) == 2
        assert streams is out.phase_mass_streams
        assert fractions is out.phase_mass_fractions
        assert densities is out.densities
        assert np.allclose(out.phase_mass_streams[0], [3.1])
        assert np.allclose(out.phase_mass_streams[1], [1.9])
        assert np.allclose(out.phase_mass_fractions[0], [[0.4, 2.7]])
        assert np.allclose(out.densities[0], [2635.0])
        assert np.allclose(out.densities[1], [28.5])
        assert out.densities[0][0] > out.densities[1][0]
        print("ok test_basic_separator_mass_stream_split")

    def test_basic_separator_mole_stream_split():
        sep = BasicSeparator(pressure=ATM, T=300.0)
        model = FakeModel()
        stream = np.array([2.0 / 0.016, 3.0 / 0.100])
        out = sep.separate_component_mole_stream(model, stream)
        streams, fractions, densities = out
        assert len(out.phase_mole_streams) == 2
        assert streams is out.phase_mole_streams
        assert fractions is out.phase_mole_fractions
        assert densities is out.molar_densities
        assert np.allclose(out.mass_result.phase_mass_streams[0], [3.1])
        assert np.allclose(out.phase_mole_fractions[0].sum(axis=1), [1.0])
        assert np.allclose(out.phase_mole_fractions[1].sum(axis=1), [1.0])
        assert out.molar_densities[0][0] > out.molar_densities[1][0]
        print("ok test_basic_separator_mole_stream_split")

    test_basic_separator_mass_stream_split()
    test_basic_separator_mole_stream_split()
    print("\nAll tests passed!")
