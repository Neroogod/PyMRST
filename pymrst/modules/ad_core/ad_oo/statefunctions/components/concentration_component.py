"""MRST ``ad-core/statefunctions/components/ConcentrationComponent.m``: a
component carried as a concentration in the phases `phaseIndices`. Its
density depends on the phase densities; when injected, each phase holding it
receives the well's or boundary's phase fraction (`compi`, else `sat`) times
the injected mass fraction, which subclasses provide
(`getInjectionMassFraction`)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.components.generic_component import (
    GenericComponent,
)


class ConcentrationComponent(GenericComponent):
    """MRST's `ConcentrationComponent`; `phaseIndices` are 0-based."""

    def __init__(self, name: str, phases: Sequence[int]):
        super().__init__(name)
        self.phaseIndices = np.atleast_1d(np.asarray(phases, dtype=np.int64))
        self.function_depends_on("getComponentDensity", "Density", "PVTPropertyFunctions")

    def get_phase_component_fraction_injection(self, model: Any, state: Any,
                                               force: Any) -> List[Optional[np.ndarray]]:
        c: List[Optional[np.ndarray]] = [None]*model.get_number_of_phases()
        forces = force if isinstance(force, (list, tuple)) else [force]
        if all(_get(f, "compi") is not None for f in forces):
            comp_i = np.vstack([np.atleast_2d(_get(f, "compi")) for f in forces])
        else:
            comp_i = np.vstack([np.atleast_2d(_get(f, "sat")) for f in forces])
        c_inj = self.get_injection_mass_fraction(model, force)
        if c_inj is None or np.size(c_inj) == 0:
            return c
        c_inj = np.atleast_1d(c_inj)
        for i, index in enumerate(self.phaseIndices):
            ci = comp_i[:, index]*c_inj[i]
            if np.any(ci != 0):
                c[index] = ci
        return c

    def get_injection_mass_fraction(self, model: Any, force: Any) -> Any:
        raise NotImplementedError("Not supported for base class!")

    getPhaseComponentFractionInjection = get_phase_component_fraction_injection
    getInjectionMassFraction = get_injection_mass_fraction


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


__all__ = ["ConcentrationComponent"]


if __name__ == "__main__":
    from types import SimpleNamespace

    class _Tracer(ConcentrationComponent):
        def get_injection_mass_fraction(self, model, force):
            return np.array([0.25])

    MODEL = SimpleNamespace(get_number_of_phases=lambda: 2)

    def test_injection_into_the_carrying_phase_only():
        comp = _Tracer("tracer", [0])
        wells = [{"compi": np.array([1.0, 0.0])}, {"compi": np.array([0.0, 1.0])}]
        c = comp.get_phase_component_fraction_injection(MODEL, None, wells)
        assert np.allclose(c[0], [0.25, 0.0]) and c[1] is None
        print("ok test_injection_into_the_carrying_phase_only")

    def test_sat_when_no_compi_and_base_refuses():
        comp = _Tracer("tracer", [1])
        c = comp.get_phase_component_fraction_injection(MODEL, None, [{"sat": np.array([0.4, 0.6])}])
        assert c[0] is None and np.allclose(c[1], [0.15])
        try:
            ConcentrationComponent("x", [0]).get_injection_mass_fraction(MODEL, None)
        except NotImplementedError:
            print("ok test_sat_when_no_compi_and_base_refuses")
            return
        raise AssertionError("expected NotImplementedError")

    test_injection_into_the_carrying_phase_only()
    test_sat_when_no_compi_and_base_refuses()
    print("\nAll tests passed!")
