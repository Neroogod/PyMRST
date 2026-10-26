"""MRST ``ad-core/statefunctions/facility/WellComponentPhaseFlux.m``.

Mass flux of each component in each phase through every perforation.
Producing perforations carry the reservoir's component density; injecting
perforations carry the well's surface composition (crossflow-corrected) at
the perforated cell's phase density.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.utils.cross_flow_mixture import cross_flow_mixture
from pymrst.utils.adi import value

COMPOSITION_FLOOR: float = 1e-10


class WellComponentPhaseFlux(StateFunction):
    """`Q_{i,alpha}`, `ncomp x nphase`."""

    def __init__(self, model: Any):
        super().__init__(model, label="Q_{i,\\alpha}")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on(["FacilityWellMapping", "PhaseFlux", "ComponentPhaseDensity"])
        self.depends_on(["Density"], "PVTPropertyFunctions")

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[List[Any]]:
        model = facility.ReservoirModel
        ncomp = model.get_number_of_components()
        nph = model.get_number_of_phases()
        mapping, phase_flux, component_density = self.get_evaluated_dependencies(
            state, "FacilityWellMapping", "PhaseFlux", "ComponentPhaseDensity")
        wc = mapping.cells
        W = mapping.W
        nw = len(W)
        flux: List[List[Any]] = [[None]*nph for _ in range(ncomp)]
        if nw == 0:
            return flux
        nperf = np.size(wc)
        perf_total = np.sum(np.column_stack([np.asarray(value(q), dtype=float)
                                             for q in phase_flux]), axis=1)
        perf_is_injector = np.asarray(mapping.isInjector, dtype=bool)[mapping.perf2well]
        perf_injecting = perf_total > 0
        surface_composition: List[List[Any]] = []
        for c in range(ncomp):
            surface_composition.append(list(
                model.Components[c].get_phase_component_fraction_injection(
                    model, state, W)))
            for ph in range(nph):
                rhoc = component_density[c][ph]
                if rhoc is not None:
                    flux[c][ph] = rhoc*phase_flux[ph]
        if np.any(perf_is_injector):
            mass_density = model.PVTPropertyFunctions.get(model, state, "Density")
            # MRST's `x(wc(perfInjecting))`: the injected fluid's density keeps
            # its derivatives in the injector cell's pressure.
            inj_cells = np.asarray(wc)[perf_injecting]
            inj_density = [rho[inj_cells] if hasattr(rho, "jac")
                           else np.asarray(rho)[inj_cells] for rho in mass_density]
            from pymrst.modules.ad_core.statefunctions.components.components import (
                ConcentrationComponent,
            )
            is_mass = np.array([not isinstance(c, ConcentrationComponent)
                                for c in model.Components])
            for c in range(ncomp):
                for ph in range(nph):
                    if surface_composition[c][ph] is None:
                        surface_composition[c][ph] = np.zeros(nw)
            for ph in range(nph):
                q = phase_flux[ph]
                compi = np.column_stack([np.asarray(surface_composition[c][ph],
                                                    dtype=float).ravel()
                                         for c in range(ncomp)])
                cflux = np.zeros((nperf, ncomp))
                for c in range(ncomp):
                    if flux[c][ph] is not None:
                        cflux[:, c] = np.asarray(value(flux[c][ph]), dtype=float)
                masses = compi[:, is_mass]
                masses = masses/np.maximum(masses.sum(axis=1, keepdims=True),
                                           COMPOSITION_FLOOR)
                compi[:, is_mass] = cross_flow_mixture(cflux[:, is_mass], masses,
                                                       mapping)
                rows = np.flatnonzero(perf_injecting)
                for c in range(ncomp):
                    if flux[c][ph] is not None:
                        perf_density = compi[mapping.perf2well[rows], c]*inj_density[ph]
                        updated = flux[c][ph] + 0.0
                        updated[rows] = q[rows]*perf_density
                        flux[c][ph] = updated
        return flux

    evaluateOnDomain = evaluate_on_domain


__all__ = ["WellComponentPhaseFlux"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_no_wells_gives_empty_component_fluxes():
        facility = SimpleNamespace(ReservoirModel=SimpleNamespace(
            get_number_of_components=lambda: 2, get_number_of_phases=lambda: 2))
        prop = WellComponentPhaseFlux(SimpleNamespace())
        prop.get_evaluated_dependencies = lambda st, *n: [
            SimpleNamespace(cells=np.zeros(0), W=[]), [], []]
        assert prop.evaluate_on_domain(facility, {}) == [[None, None], [None, None]]
        print("ok test_no_wells_gives_empty_component_fluxes")

    test_no_wells_gives_empty_component_fluxes()
    print("\nAll tests passed!")
