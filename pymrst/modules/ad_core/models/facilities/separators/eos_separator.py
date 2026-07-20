"""EOS separator for compositional component streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.modules.ad_core.models.facilities.separators.basic_separator import (
    BasicSeparator,
    SeparatorMassResult,
)
from pymrst.modules.compositional.eos import Mixture, flash
from pymrst.modules.compositional.properties import molar_density


@dataclass
class SeparatorFlashResult:
    """Phase split from a separator flash."""

    phase_mole_streams: list[np.ndarray]
    phase_mole_fractions: list[np.ndarray]
    molar_densities: list[np.ndarray]
    flash_result: Any

    def __iter__(self):
        """Unpack like MRST's three-output separateComponentMoleStream."""
        yield self.phase_mole_streams
        yield self.phase_mole_fractions
        yield self.molar_densities


def _mixture(model_or_mixture) -> Mixture:
    """Extract a compositional mixture from a model or accept one directly."""
    if isinstance(model_or_mixture, Mixture):
        return model_or_mixture
    for name in ("mix", "mixture", "fluid", "compositional_mixture"):
        candidate = getattr(model_or_mixture, name, None)
        if isinstance(candidate, Mixture):
            return candidate
    raise TypeError("expected a Mixture or a model carrying a Mixture")


def _as_stream_array(stream, ncomp: int | None = None) -> tuple[np.ndarray, bool]:
    """Return stream as `(nstream, ncomp)` and whether input was scalar-row."""
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
    if ncomp is not None and arr.shape[1] != ncomp:
        raise ValueError(f"expected {ncomp} components, got {arr.shape[1]}")
    return arr, arr.shape[0] == 1


class EOSSeparator(BasicSeparator):
    """Peng-Robinson separator equivalent to MRST `EOSSeparator`."""

    def __init__(
        self,
        pressure: float = np.nan,
        T: float = np.nan,
        EOSModel=None,
        *,
        eos_model=None,
    ):
        if EOSModel is not None and eos_model is not None:
            raise ValueError("use either EOSModel or eos_model, not both")
        super().__init__(pressure=float(pressure), T=float(T))
        self.EOSModel = EOSModel if eos_model is None else eos_model

    @property
    def eos_model(self):
        """Pythonic alias for MRST's ``EOSModel`` property."""
        return self.EOSModel

    @eos_model.setter
    def eos_model(self, value):
        self.EOSModel = value

    def clone(self, *, pressure: float | None = None, T: float | None = None) -> "EOSSeparator":
        """Return the same EOS separator at a new pressure/temperature."""
        return type(self)(
            pressure=self.pressure if pressure is None else float(pressure),
            T=self.T if T is None else float(T),
            EOSModel=self.EOSModel,
        )

    def separate_component_mass_stream(self, model, mass_stream) -> SeparatorMassResult:
        """Flash component mass streams at separator conditions.

        MRST facility equations often carry produced component streams in mass
        units before separator allocation.  The EOS separator can still perform
        the thermodynamic split by converting each component mass rate to a mole
        rate, applying ``separate_component_mole_stream``, and converting the
        phase component mole streams back to mass.  The return type mirrors
        ``BasicSeparator``: total phase mass streams, per-phase component mass
        streams, and phase mass densities.
        """
        mix = _mixture(self.EOSModel if self.EOSModel is not None else model)
        stream_mass, _ = _as_stream_array(mass_stream, mix.n)
        stream_mole = stream_mass / mix.molarMass[None, :]
        split = self.separate_component_mole_stream(mix, stream_mole)

        phase_component_mass_streams: list[np.ndarray] = []
        phase_mass_streams: list[np.ndarray] = []
        densities: list[np.ndarray] = []
        for phase_index in range(2):
            component_moles = (
                split.phase_mole_fractions[phase_index]
                * split.phase_mole_streams[phase_index][:, None]
            )
            component_mass = component_moles * mix.molarMass[None, :]
            total_mass = component_mass.sum(axis=1)
            phase_component_mass_streams.append(component_mass)
            phase_mass_streams.append(total_mass)
            phase_molar_mass = split.phase_mole_fractions[phase_index] @ mix.molarMass
            densities.append(split.molar_densities[phase_index] * phase_molar_mass)

        return SeparatorMassResult(
            phase_mass_streams=phase_mass_streams,
            phase_mass_fractions=phase_component_mass_streams,
            densities=densities,
        )

    def separate_component_mole_stream(self, model, mole_stream) -> SeparatorFlashResult:
        """Flash component mole streams at the separator conditions.

        This follows MRST's `EOSSeparator.separateComponentMoleStream`: normalize
        the incoming component mole rates to an overall composition, flash it,
        and split the total mole rate into liquid and vapour streams.
        """
        mix = _mixture(self.EOSModel if self.EOSModel is not None else model)
        stream, _ = _as_stream_array(mole_stream, mix.n)
        total = stream.sum(axis=1)
        safe_total = np.where(np.abs(total) > 0.0, total, 1.0)
        z = stream / safe_total[:, None]
        z = np.where(np.isfinite(z), z, 0.0)
        z = np.maximum(z, 1.0e-8)
        z = z / z.sum(axis=1, keepdims=True)

        p = np.full(stream.shape[0], self.pressure, dtype=float)
        T = np.full(stream.shape[0], self.T, dtype=float)
        res = flash(mix, p, T, z, tol=1.0e-12, max_it=500)
        vapour = res.V
        liquid = 1.0 - vapour

        mole_l = liquid * total
        mole_v = vapour * total
        rho_l = molar_density(mix, p, T, res.x, res.Z_L)
        rho_v = molar_density(mix, p, T, res.y, res.Z_V)
        return SeparatorFlashResult(
            phase_mole_streams=[mole_l, mole_v],
            phase_mole_fractions=[res.x, res.y],
            molar_densities=[rho_l, rho_v],
            flash_result=res,
        )

    # MRST-compatible aliases.
    separateComponentMassStream = separate_component_mass_stream
    separateComponentMoleStream = separate_component_mole_stream


__all__ = ["EOSSeparator", "SeparatorFlashResult", "SeparatorMassResult", "_as_stream_array", "_mixture"]
