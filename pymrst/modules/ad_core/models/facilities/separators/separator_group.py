"""Separator trees for compositional produced streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.modules.ad_core.models.facilities.separators.eos_separator import (
    EOSSeparator,
    _as_stream_array,
    _mixture,
)


@dataclass
class SeparatorSurfaceResult:
    """Surface-rate result from a separator group."""

    surface_rates: list[np.ndarray]
    phase_streams: list[np.ndarray]
    phase_mole_fractions: list[np.ndarray]
    densities: list[np.ndarray]
    surface_mole_streams: list[np.ndarray] | None = None
    molar_densities: list[np.ndarray] | None = None
    separator_streams: "SeparatorStreamResult | None" = None

    def __iter__(self):
        """Unpack as surface rates, phase streams, fractions and densities."""
        yield self.surface_rates
        yield self.phase_streams
        yield self.phase_mole_fractions
        yield self.densities


@dataclass
class SeparatorStreamResult:
    """MRST-style intermediate separator stream diagnostics.

    Mirrors the second output of ``SeparatorGroup.computeSurfaceMoleRates``:
    ``molestreams``, ``densities`` and ``molefractions`` are lists with one
    entry per intermediate separator. Inactive separators contain ``None``.
    """

    molestreams: list[list[np.ndarray] | None]
    densities: list[list[np.ndarray] | None]
    molefractions: list[list[np.ndarray] | None]

    def __iter__(self):
        """Unpack like MRST's separator-stream diagnostics."""
        yield self.molestreams
        yield self.densities
        yield self.molefractions


class SeparatorGroup:
    """MRST-style separator group.

    The group either flashes a produced stream directly in a single surface
    separator, or routes liquid/vapour outputs through a tree of intermediate
    separators before recomputing final surface volumes.
    """

    def __init__(
        self,
        separator,
        pressure=None,
        T=None,
        phase_destination=None,
        *,
        mode: str = "moles",
        phaseDestination=None,
        surfaceSeparator=None,
    ):
        if phaseDestination is not None:
            if phase_destination is not None:
                raise ValueError("use either phase_destination or phaseDestination, not both")
            phase_destination = phaseDestination
        self.surface_separator = separator if surfaceSeparator is None else surfaceSeparator
        self.mode = mode
        if pressure is None and T is None and phase_destination is None:
            self.separators: list[EOSSeparator] = []
            self.phase_destination = np.zeros((0, 2), dtype=int)
            self.topological_order = np.array([], dtype=int)
            return

        if pressure is None or T is None or phase_destination is None:
            raise ValueError("pressure, T and phase_destination must be provided together")
        pressure = np.asarray(pressure, dtype=float).reshape(-1)
        T = np.asarray(T, dtype=float).reshape(-1)
        if pressure.shape != T.shape:
            raise ValueError("pressure and T must have the same length")
        dest = np.asarray(phase_destination, dtype=int)
        if dest.shape != (pressure.size, 2):
            raise ValueError("phase_destination must have shape (nseparator, 2)")

        # MRST convention: a zero liquid destination means "to the next
        # separator", except at the last stage where zero means stock tank.
        dest = dest.copy()
        rows_to_next = np.flatnonzero(dest[:, 0] == 0)
        dest[rows_to_next, 0] = rows_to_next + 2
        dest[-1, 0] = 0
        dest[dest == -1] = 0

        self.separators = [
            _clone_separator(separator, pressure=float(p), T=float(t))
            for p, t in zip(pressure, T)
        ]
        self.phase_destination = dest
        self.topological_order = self._topological_sort(dest)

    @property
    def has_intermediate_separators(self) -> bool:
        """Whether this group contains separator stages before the stock tank."""
        return len(self.separators) > 0

    @staticmethod
    def _topological_sort(dest: np.ndarray) -> np.ndarray:
        """Topologically sort the 1-based separator graph."""
        nsep = dest.shape[0]
        incoming = np.zeros(nsep, dtype=int)
        outgoing: list[list[int]] = [[] for _ in range(nsep)]
        for i in range(nsep):
            for target in dest[i]:
                if 1 <= target <= nsep:
                    outgoing[i].append(target - 1)
                    incoming[target - 1] += 1
        ready = [i for i in range(nsep) if incoming[i] == 0]
        order: list[int] = []
        while ready:
            i = ready.pop(0)
            order.append(i)
            for j in outgoing[i]:
                incoming[j] -= 1
                if incoming[j] == 0:
                    ready.append(j)
        if len(order) != nsep:
            raise ValueError("separator phase_destination contains a cycle")
        return np.asarray(order, dtype=int)

    @property
    def phaseDestination(self) -> np.ndarray:
        """MRST-compatible alias for ``phase_destination``."""
        return self.phase_destination

    @phaseDestination.setter
    def phaseDestination(self, value):
        self.phase_destination = np.asarray(value, dtype=int)
        self.topological_order = self._topological_sort(self.phase_destination)

    @property
    def topologicalOrder(self) -> np.ndarray:
        """MRST-compatible alias for ``topological_order``."""
        return self.topological_order

    @topologicalOrder.setter
    def topologicalOrder(self, value):
        self.topological_order = np.asarray(value, dtype=int)

    @property
    def surfaceSeparator(self):
        """MRST-compatible alias for ``surface_separator``."""
        return self.surface_separator

    @surfaceSeparator.setter
    def surfaceSeparator(self, value):
        self.surface_separator = value

    def get_topological_sort(self) -> np.ndarray:
        """Return the MRST-style separator topological ordering."""
        if self.phase_destination.size == 0:
            return np.array([], dtype=int)
        return self._topological_sort(self.phase_destination)

    def compute_surface_mole_rates(
        self,
        model,
        stream_mole,
        start: int = 1,
        *,
        return_separator_streams: bool = False,
    ):
        """Route a mole stream through the separator tree.

        `start` follows MRST's 1-based separator numbering.
        """
        mix = _mixture(model)
        stream, _ = _as_stream_array(stream_mole, mix.n)
        if not self.has_intermediate_separators:
            surface = [np.zeros_like(stream), np.zeros_like(stream)]
            if return_separator_streams:
                return surface, SeparatorStreamResult([], [], [])
            return surface
        if not (1 <= start <= len(self.separators)):
            raise ValueError("start separator must be in 1..nseparator")

        separator_streams: list[np.ndarray | None] = [None] * len(self.separators)
        separator_streams[start - 1] = stream
        surface_streams = [np.zeros_like(stream), np.zeros_like(stream)]
        out_molestreams: list[list[np.ndarray] | None] = [None] * len(self.separators)
        out_fractions: list[list[np.ndarray] | None] = [None] * len(self.separators)
        out_densities: list[list[np.ndarray] | None] = [None] * len(self.separators)
        for sep_index in self.topological_order:
            in_stream = separator_streams[sep_index]
            if in_stream is None:
                continue
            split = self.separators[sep_index].separate_component_mole_stream(mix, in_stream)
            for phase_index in range(2):
                component_stream = (
                    split.phase_mole_fractions[phase_index]
                    * split.phase_mole_streams[phase_index][:, None]
                )
                target = int(self.phase_destination[sep_index, phase_index])
                if target <= 0 or target > len(self.separators):
                    surface_streams[phase_index] += component_stream
                else:
                    old = separator_streams[target - 1]
                    separator_streams[target - 1] = (
                        component_stream if old is None else old + component_stream
                    )
            out_molestreams[sep_index] = split.phase_mole_streams
            out_fractions[sep_index] = split.phase_mole_fractions
            out_densities[sep_index] = split.molar_densities
        if return_separator_streams:
            return surface_streams, SeparatorStreamResult(
                molestreams=out_molestreams,
                densities=out_densities,
                molefractions=out_fractions,
            )
        return surface_streams

    def _flash_surface_stream(
        self,
        mix,
        stream: np.ndarray,
        phase_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flash one final phase stream at stock-tank conditions."""
        total = stream.sum(axis=1)
        frac = np.zeros_like(stream)
        active = np.abs(total) > 0.0
        frac[active] = stream[active] / total[active, None]
        rates = np.zeros(stream.shape[0])
        density = np.full(stream.shape[0], np.nan)
        molar_density = np.full(stream.shape[0], np.nan)
        if active.any():
            split = self.surface_separator.separate_component_mole_stream(mix, stream[active])
            phase_volumes = [
                split.phase_mole_streams[i] / split.molar_densities[i]
                for i in range(2)
            ]
            rates[active] = phase_volumes[0] + phase_volumes[1]
            molar_density[active] = split.molar_densities[phase_index]
            density[active] = split.molar_densities[phase_index] * (
                split.phase_mole_fractions[phase_index] @ mix.molarMass
            )
        return rates, frac, density, molar_density

    def compute_surface_rates(self, model, stream, *, mode: str | None = None) -> SeparatorSurfaceResult:
        """Compute stock-tank phase volume rates from component streams."""
        mode = _normalize_separator_mode(self.mode if mode is None else mode)
        if mode == "mass":
            if self.has_intermediate_separators:
                mix = _mixture(model)
                mass_stream, _ = _as_stream_array(stream, mix.n)
                mole_stream = mass_stream / mix.molarMass[None, :]
                mole_result = self.compute_surface_rates(mix, mole_stream, mode="moles")
                surface_mole_streams = mole_result.surface_mole_streams or []
                phase_component_mass_streams = [
                    np.asarray(phase_stream, dtype=float) * mix.molarMass[None, :]
                    for phase_stream in surface_mole_streams
                ]
                phase_mass_streams = [phase.sum(axis=1) for phase in phase_component_mass_streams]
                return SeparatorSurfaceResult(
                    surface_rates=mole_result.surface_rates,
                    phase_streams=phase_mass_streams,
                    phase_mole_fractions=phase_component_mass_streams,
                    densities=mole_result.densities,
                    surface_mole_streams=surface_mole_streams,
                    molar_densities=mole_result.molar_densities,
                    separator_streams=mole_result.separator_streams,
                )
            result = self.surface_separator.separate_component_mass_stream(model, stream)
            rates = []
            for q_mass, rho in zip(result.phase_mass_streams, result.densities):
                q = np.zeros_like(q_mass)
                active = np.isfinite(rho) & (np.abs(rho) > 0.0)
                q[active] = q_mass[active] / rho[active]
                rates.append(q)
            return SeparatorSurfaceResult(
                surface_rates=rates,
                phase_streams=result.phase_mass_streams,
                phase_mole_fractions=result.phase_mass_fractions,
                densities=result.densities,
                surface_mole_streams=None,
                molar_densities=None,
            )
        if mode != "moles":
            raise NotImplementedError(f"unknown separator mode '{mode}'")
        mix = _mixture(model)
        stream_mole, _ = _as_stream_array(stream, mix.n)

        if not self.has_intermediate_separators:
            split = self.surface_separator.separate_component_mole_stream(mix, stream_mole)
            rates = [
                split.phase_mole_streams[i] / split.molar_densities[i]
                for i in range(2)
            ]
            mass_density = [
                split.molar_densities[i] * (split.phase_mole_fractions[i] @ mix.molarMass)
                for i in range(2)
            ]
            phase_streams = [
                split.phase_mole_fractions[i] * split.phase_mole_streams[i][:, None]
                for i in range(2)
            ]
            return SeparatorSurfaceResult(
                surface_rates=rates,
                phase_streams=phase_streams,
                phase_mole_fractions=split.phase_mole_fractions,
                densities=mass_density,
                surface_mole_streams=phase_streams,
                molar_densities=split.molar_densities,
            )

        surface_streams, separator_streams = self.compute_surface_mole_rates(
            mix,
            stream_mole,
            return_separator_streams=True,
        )
        rates: list[np.ndarray] = []
        fractions: list[np.ndarray] = []
        mass_densities: list[np.ndarray] = []
        molar_densities: list[np.ndarray] = []
        for phase_index, phase_stream in enumerate(surface_streams):
            q, frac, rho_mass, rho_molar = self._flash_surface_stream(mix, phase_stream, phase_index)
            rates.append(q)
            fractions.append(frac)
            mass_densities.append(rho_mass)
            molar_densities.append(rho_molar)
        return SeparatorSurfaceResult(
            surface_rates=rates,
            phase_streams=surface_streams,
            phase_mole_fractions=fractions,
            densities=mass_densities,
            surface_mole_streams=surface_streams,
            molar_densities=molar_densities,
            separator_streams=separator_streams,
        )

    def get_surface_rates(self, model, component_mass_rates) -> SeparatorSurfaceResult:
        """Compute surface rates from component mass rates.

        In ``mode='moles'`` the incoming mass stream is converted to component
        mole rates using the mixture molar masses, exactly as MRST's
        ``SeparatorGroup.getSurfaceRates`` does. In ``mode='mass'`` the stream
        is passed directly to ``BasicSeparator.separateComponentMassStream``.
        """
        if _normalize_separator_mode(self.mode) == "mass":
            return self.compute_surface_rates(model, component_mass_rates, mode="mass")
        mix = _mixture(model)
        mass_stream, _ = _as_stream_array(component_mass_rates, mix.n)
        mole_stream = mass_stream / mix.molarMass[None, :]
        return self.compute_surface_rates(mix, mole_stream, mode="moles")

    # MRST-compatible aliases.
    computeSurfaceMoleRates = compute_surface_mole_rates
    computeSurfaceRates = compute_surface_rates
    getSurfaceRates = get_surface_rates
    getTopologicalSort = get_topological_sort


__all__ = ["SeparatorGroup", "SeparatorStreamResult", "SeparatorSurfaceResult"]


def _clone_separator(separator, *, pressure: float, T: float):
    if hasattr(separator, "clone"):
        return separator.clone(pressure=pressure, T=T)
    return type(separator)(pressure=pressure, T=T)


def _normalize_separator_mode(mode: str) -> str:
    mode = str(mode).lower()
    aliases = {
        "mole": "moles",
        "moles": "moles",
        "mol": "moles",
        "molar": "moles",
        "mass": "mass",
        "masses": "mass",
        "kg": "mass",
    }
    return aliases.get(mode, mode)


if __name__ == "__main__":
    from pymrst.modules.ad_core.models.facilities.separators.basic_separator import BasicSeparator
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
                    surface_phase_mass_fractions=[0.25, 0.75],
                    surface_phase_density_pure=[700.0, 5.0],
                    molar_mass=0.016,
                ),
                EquationOfStateComponent(
                    "Heavy",
                    pressure=ATM,
                    T=300.0,
                    component_index=1,
                    surface_phase_mass_fractions=[0.85, 0.15],
                    surface_phase_density_pure=[850.0, 15.0],
                    molar_mass=0.100,
                ),
            ]

        def get_number_of_phases(self):
            return self.n_phases

    def test_separator_group_mass_mode_surface_rates():
        model = FakeModel()
        group = SeparatorGroup(BasicSeparator(pressure=ATM, T=300.0), mode="mass")
        stream = np.array([2.0, 3.0])
        out = group.get_surface_rates(model, stream)
        rates, streams, fractions, densities = out
        assert len(out.surface_rates) == 2
        assert rates is out.surface_rates
        assert streams is out.phase_streams
        assert fractions is out.phase_mole_fractions
        assert densities is out.densities
        assert np.allclose(out.phase_streams[0] + out.phase_streams[1], [5.0])
        assert out.surface_rates[0][0] > 0.0
        assert out.surface_rates[1][0] > 0.0
        assert out.molar_densities is None
        print("ok test_separator_group_mass_mode_surface_rates")

    test_separator_group_mass_mode_surface_rates()
    print("\nAll tests passed!")
