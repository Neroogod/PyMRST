"""MRST ``ad-core/statefunctions/facility/WellComponentTotalVolumeBalanceCrossflow.m``.

The component fluxes between wells and cells, as `ComponentTotalFlux` sums
them, except at the perforations of a cross-flowing well that take fluid
from the wellbore. There, the mass a well receives - the backflow from its
other perforations plus, for an injector, the injected surface stream -
is mixed, and each such perforation gets a share of the mixture that
balances its reservoir volume flux: a producer by phase mass fractions of
the wellbore mixture (`'phasemassfractions'`), an injector by the backflow
density of the well's injecting volume plus the injected stream weighted
by the perforation's share (`'specialsauce'`). `computeCrossFlux` also
carries MRST's `'exactmassfractions'` and `'exactmass'` rules.

With `onlyLocalDerivatives` (the default), the well totals enter as values;
only the perforation's own flux carries derivatives.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import ad_max, ad_min, value

MIN_FRACTION: float = 1e-12


class WellComponentTotalVolumeBalanceCrossflow(StateFunction):
    """Component fluxes with volume-balanced mixing at cross-flowing perforations."""

    def __init__(self, model: Any):
        super().__init__(model, label="Q_i")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.onlyLocalDerivatives = True
        self.depends_on("FacilityWellMapping")
        self.depends_on(["ComponentPhaseFlux", "InjectionSurfaceDensity"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        ncomp = model.get_number_of_components()
        nph = model.get_number_of_phases()
        phase_flux = self.get_evaluated_dependencies(state, "ComponentPhaseFlux")
        v: List[Any] = [None]*ncomp
        for c in range(ncomp):
            for ph in range(nph):
                m = phase_flux[c][ph]
                if m is not None:
                    v[c] = m if v[c] is None else v[c] + m
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        W = mapping.W
        mass_flux = np.column_stack([np.asarray(value(x), dtype=float).ravel() for x in v])
        mass_flux_total = np.sum(mass_flux, axis=1)
        psum = mapping.perforationSum
        perf2well = np.asarray(mapping.perf2well, dtype=np.int64)
        is_injector = np.asarray(mapping.isInjector, dtype=bool)
        injection = mass_flux_total > 0
        production = mass_flux_total < 0
        crossflow = (injection & ~is_injector[perf2well]) | (production & is_injector[perf2well])
        well_has_xflow = psum @ crossflow.astype(float)
        replace = injection & (well_has_xflow[perf2well] != 0)
        if not np.any(replace):
            return v

        rhoS = self.get_evaluated_dependencies(state, "InjectionSurfaceDensity")
        ws = [state["wellSol"][i] for i in np.asarray(mapping.active, dtype=np.int64)]
        target_type = [w.type for w in ws]
        nw = len(W)
        resmodel = model.ReservoirModel
        surface_composition = [[None]*nph for _ in range(ncomp)]
        for c in range(ncomp):
            surface_composition[c] = list(
                resmodel.Components[c].get_phase_component_fraction_injection(resmodel, state, W))
        for c in range(ncomp):
            for ph in range(nph):
                if surface_composition[c][ph] is None:
                    surface_composition[c][ph] = np.zeros(nw)
        injection_mass: List[List[Any]] = [[0]*nph for _ in range(ncomp)]
        phase_compi = np.vstack([np.asarray(w.compi, dtype=float).ravel() for w in W])
        is_rate = ~np.array([t.lower() in ("resv", "bhp") for t in target_type])
        is_rate_injector = is_rate & is_injector
        qs = state["FacilityState"]["surfacePhaseRates"]
        target = np.array([float(np.ravel(w.val)[0]) for w in ws])
        for ph in range(nph):
            q_surf = (~is_rate_injector*np.asarray(value(qs[ph]), dtype=float)
                      + is_rate_injector*target)
            qs_ph = np.asarray(rhoS[ph], dtype=float)*phase_compi[:, ph]*q_surf
            for c in range(ncomp):
                if np.any(surface_composition[c][ph]):
                    injection_mass[c][ph] = qs_ph*surface_composition[c][ph]

        vol_flux = model.get_prop(state, "PhaseFlux")
        resdens = resmodel.get_prop(state, "Density")
        cells = np.asarray(mapping.cells, dtype=np.int64)
        resdens = [r[cells] for r in resdens]

        well_component_total_mass: List[Any] = [0]*ncomp
        well_component_total_mass_xflow: List[Any] = [0]*ncomp
        well_phase_total_mass: List[Any] = [0]*nph
        phase_mass_composition: List[List[Any]] = [[None]*nph for _ in range(ncomp)]
        for ph in range(nph):
            for c in range(ncomp):
                mi = phase_flux[c][ph]
                if mi is None:
                    continue
                from_surface = -np.maximum(injection_mass[c][ph], 0)
                qm = np.asarray(value(ad_min(mi, 0)), dtype=float)
                qm = self.reduce(qm)
                xflow = psum @ qm
                added_mass = xflow + from_surface
                well_phase_total_mass[ph] = well_phase_total_mass[ph] + added_mass
                well_component_total_mass_xflow[c] = well_component_total_mass_xflow[c] + xflow
                well_component_total_mass[c] = well_component_total_mass[c] + added_mass
                phase_mass_composition[c][ph] = added_mass
            tot = 0
            for c in range(ncomp):
                x = phase_mass_composition[c][ph]
                if x is not None:
                    x = np.maximum(np.abs(x), MIN_FRACTION)
                    tot = tot + x
                    phase_mass_composition[c][ph] = x
            for c in range(ncomp):
                x = phase_mass_composition[c][ph]
                if x is not None:
                    phase_mass_composition[c][ph] = x/tot

        vol_flux_total = 0
        for i in range(nph):
            vol_flux_total = vol_flux_total + vol_flux[i]
        active = perf2well[replace]
        tot = 0
        phase_volume_into_wellbore: List[Any] = [None]*nph
        for ph in range(nph):
            r = _rows(resdens[ph], replace)
            x = np.asarray(well_phase_total_mass[ph])[active]/r
            x = ad_max(_ad_abs(x), MIN_FRACTION)
            tot = tot + x
            phase_volume_into_wellbore[ph] = x
        phase_volume_fractions = [phase_volume_into_wellbore[ph]/tot for ph in range(nph)]
        for c in range(ncomp):
            v[c] = _assign(v[c], replace, 0)
        vft = vol_flux_total[replace]
        replace_mass_flux = [phase_volume_fractions[ph]*vft*_rows(resdens[ph], replace)
                             for ph in range(nph)]
        args = (mapping, replace, ncomp, nph, resdens, phase_flux, replace_mass_flux,
                well_component_total_mass, well_component_total_mass_xflow,
                phase_mass_composition, vol_flux_total, rhoS, phase_compi,
                surface_composition, qs, injection_mass)
        v_r = self.compute_cross_flux(model, state, "specialsauce", *args)
        v_f = self.compute_cross_flux(model, state, "phasemassfractions", *args)
        ii = is_injector[active]
        for c in range(ncomp):
            new = v_f[c]*~ii
            if np.any(ii):
                new = new + v_r[c]*ii
            v[c] = _assign(v[c], replace, new)
        return v

    def compute_cross_flux(self, model, state, masstype, mapping, replace, ncomp, nph,
                           resdens, phase_flux, replace_mass_flux,
                           well_component_total_mass, well_component_total_mass_xflow,
                           phase_mass_composition, vol_flux_total, rhoS, phase_compi,
                           surface_composition, qs, injection_mass) -> List[Any]:
        """MRST `computeCrossFlux`: the replaced perforations' component fluxes."""
        psum = mapping.perforationSum
        perf2well = np.asarray(mapping.perf2well, dtype=np.int64)
        active = perf2well[replace]
        vft = vol_flux_total[replace]
        inj_vol = self.reduce(vol_flux_total)
        inj_vol = ad_max(inj_vol, 0)
        inj_vol_tot = psum @ np.asarray(value(inj_vol), dtype=float) \
            if not hasattr(inj_vol, "jac") else _matmul(psum, inj_vol)
        v: List[Any] = [None]*ncomp
        kind = masstype.lower()
        if kind == "exactmassfractions":
            replace_total = 0
            for ph in range(nph):
                replace_total = replace_total + replace_mass_flux[ph]
            xi = np.column_stack([np.abs(np.asarray(value(m), dtype=float)) + MIN_FRACTION
                                  for m in well_component_total_mass])
            xi = xi/np.sum(xi, axis=1)[:, None]
            for c in range(ncomp):
                v[c] = xi[active, c]*replace_total
        elif kind == "phasemassfractions":
            for c in range(ncomp):
                qi = 0
                for ph in range(nph):
                    x = phase_mass_composition[c][ph]
                    if x is not None:
                        qi = qi + replace_mass_flux[ph]*x[active]
                v[c] = qi
        elif kind == "exactmass":
            for c in range(ncomp):
                mass = -well_component_total_mass[c]
                rho = mass/inj_vol_tot
                rp = np.asarray(value(rho[active] if hasattr(rho, "jac") else
                                      np.asarray(rho)[active]), dtype=float)
                v[c] = rp*vft
        elif kind == "specialsauce":
            tmp = self.reduce(vol_flux_total)
            delta = (-np.asarray(value(vol_flux_total), dtype=float) + vol_flux_total
                     if self.onlyLocalDerivatives else 0)
            inj_vol_tot_perf = _rows(inj_vol_tot, perf2well) + delta
            tmp = _matmul(psum, inj_vol)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = vol_flux_total/_rows(tmp, perf2well)
            w = _assign(w, ~np.isfinite(np.asarray(value(w), dtype=float)), 1)
            for c in range(ncomp):
                mass = -np.asarray(well_component_total_mass_xflow[c])[active]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho = mass/inj_vol_tot_perf[replace]
                qi = rho*vft
                extra = 0
                added = False
                for ph in range(nph):
                    add = injection_mass[c][ph]
                    if np.size(add) == 0 or np.all(np.asarray(value(add)) == 0):
                        continue
                    add = np.maximum(add, 0)
                    extra = extra + np.asarray(add)[perf2well]*w
                    added = True
                v[c] = qi + extra[replace] if added else qi
        else:
            raise ValueError(f"{masstype} is not supported")
        return v

    def reduce(self, x: Any) -> Any:
        return value(x) if self.onlyLocalDerivatives else x

    evaluateOnDomain = evaluate_on_domain
    computeCrossFlux = compute_cross_flux


def _ad_abs(x: Any) -> Any:
    from pymrst.utils.adi import ad_abs
    return ad_abs(x)


def _matmul(mat: Any, x: Any) -> Any:
    from pymrst.utils.adi import ad_matmul
    return ad_matmul(mat, x)


def _rows(x: Any, idx: np.ndarray) -> Any:
    return x[idx] if hasattr(x, "jac") else np.asarray(x)[idx]


def _assign(x: Any, mask: np.ndarray, new: Any) -> Any:
    """MATLAB `x(mask) = new`, on a copy."""
    if not np.any(mask):
        return x
    if hasattr(x, "jac"):
        x = x.copy()
    else:
        x = np.array(x, dtype=float)
        if hasattr(new, "jac"):
            raise TypeError("cannot write an AD value into a plain array")
    x[mask] = new
    return x


__all__ = ["WellComponentTotalVolumeBalanceCrossflow"]
