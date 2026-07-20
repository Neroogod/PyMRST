"""Phase 119: BasicSeparator mass mode vs MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core import BasicSeparator, SeparatorGroup
from pymrst.modules.compositional.properties.components import EquationOfStateComponent

HERE = Path(__file__).parent
REF = HERE / "ref_p119_basic_separator_mass.mat"


class FakeModel:
    n_phases = 2

    def __init__(self, pressure: float, temperature: float):
        self.components = [
            EquationOfStateComponent(
                "Light",
                pressure=pressure,
                T=temperature,
                component_index=0,
                surface_phase_mass_fractions=[0.20, 0.80],
                surface_phase_density_pure=[700.0, 5.0],
                molar_mass=0.016,
            ),
            EquationOfStateComponent(
                "Heavy",
                pressure=pressure,
                T=temperature,
                component_index=1,
                surface_phase_mass_fractions=[0.90, 0.10],
                surface_phase_density_pure=[850.0, 15.0],
                molar_mass=0.100,
            ),
        ]

    def get_number_of_phases(self) -> int:
        return self.n_phases


def _phase_rows(values):
    return np.asarray([np.asarray(v, dtype=float).reshape(-1)[0] for v in values])


def _phase_matrix(values):
    return np.vstack([np.asarray(v, dtype=float).reshape(1, -1) for v in values])


def compare(name, py, ref, rtol=0.0, atol=0.0) -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if py.shape != ref.shape:
        diff = np.inf
        ok = False
    else:
        diff = float(np.nanmax(np.abs(py - ref))) if py.size else 0.0
        ref_scale = max(float(np.nanmax(np.abs(ref))), 1.0)
        ok = diff <= atol + rtol * ref_scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
        f"(rtol {rtol:.1e}, atol {atol:.1e})"
    )
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p119_basic_separator_mass.mat missing (run dump_phase119_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    pressure = float(np.asarray(ref["pressure"]).reshape(-1)[0])
    temperature = float(np.asarray(ref["temperature"]).reshape(-1)[0])
    stream = np.asarray(ref["component_stream"], dtype=float).reshape(-1)
    model = FakeModel(pressure, temperature)
    sep = BasicSeparator(pressure=pressure, T=temperature)

    split = sep.separate_component_mass_stream(model, stream)
    group = SeparatorGroup(sep, mode="mass")
    group_rates = group.get_surface_rates(model, stream)

    print("\nPhase 119: BasicSeparator mass mode vs MATLAB")
    failures = 0
    failures += compare("phase mass streams", _phase_rows(split.phase_mass_streams),
                        ref["phase_mass_streams"], rtol=1.0e-12, atol=1.0e-14)
    failures += compare("phase component mass streams", _phase_matrix(split.phase_mass_fractions),
                        ref["phase_mass_fractions"], rtol=1.0e-12, atol=1.0e-14)
    failures += compare("separator densities", _phase_rows(split.densities),
                        ref["densities"], rtol=1.0e-12, atol=1.0e-14)
    failures += compare("separator surface rates", _phase_rows(group_rates.surface_rates),
                        ref["surface_rates"], rtol=1.0e-12, atol=1.0e-14)

    if failures:
        print(f"\nPhase 119 validation failed ({failures} checks).")
    else:
        print("\nBasicSeparator mass-mode split matches MATLAB.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
