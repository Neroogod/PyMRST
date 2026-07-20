"""Validate compositional well forces (MRST ``validateCompositionalForces.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.utils.standalone_flash import standaloneFlash


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _call(obj: Any, camel: str, snake: str | None = None, default=None):
    if obj is None:
        return default
    for name in (camel, snake):
        if not name:
            continue
        value = _get(obj, name, None)
        if value is not None:
            return value() if callable(value) else value
    return default


def _wells(forces: Any) -> list[Any]:
    wells = _get(forces, "W", [])
    if wells is None:
        return []
    if isinstance(wells, np.ndarray):
        wells = wells.tolist()
    if isinstance(wells, (list, tuple)):
        return list(wells)
    return [wells]


def _facility(model: Any):
    facility = _get(model, "FacilityModel", None)
    if facility is None:
        facility = _get(model, "facility_model", None)
    if facility is None:
        facility = _get(model, "facility", None)
    return facility


def _eos(model: Any):
    eos = _get(model, "EOSModel", None)
    if eos is None:
        eos = _get(model, "eos_model", None)
    if eos is None:
        raise AssertionError("model must expose EOSModel before validating compositional forces")
    return eos


def _molar_mass(eos) -> np.ndarray:
    mixture = _get(eos, "CompositionalMixture", None)
    if mixture is None:
        mixture = _get(eos, "mixture", None)
    if mixture is None:
        mixture = _get(eos, "fluid", None)
    if mixture is None:
        raise ValueError("EOSModel must expose a compositional mixture")
    return np.asarray(mixture.molarMass, dtype=float)


def _mass_fraction(eos, z) -> np.ndarray:
    z = np.atleast_2d(np.asarray(z, dtype=float))
    mw = _molar_mass(eos)
    mass = z * mw[None, :]
    denom = mass.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(mass, denom, out=np.zeros_like(mass), where=denom != 0.0)
    return out


def _surface_densities(model: Any, nph: int, liquid_index: int, vapor_index: int, rho_l: float, rho_v: float) -> np.ndarray:
    rho = _call(model, "getSurfaceDensities", "get_surface_densities", default=None)
    if rho is None:
        rho = np.zeros(nph, dtype=float)
    else:
        rho = np.asarray(rho, dtype=float).copy()
        if rho.ndim > 1:
            rho = rho.reshape(-1, rho.shape[-1])[0].copy()
        if rho.size < nph:
            rho = np.pad(rho, (0, nph - rho.size), constant_values=0.0)
    rho[liquid_index] = rho_l
    rho[vapor_index] = rho_v
    return rho


def _phase_indices(model: Any) -> tuple[int, int, int]:
    liquid_index = int(_call(model, "getLiquidIndex", "get_liquid_index", default=0))
    vapor_index = int(_call(model, "getVaporIndex", "get_vapor_index", default=1))
    nph = _call(model, "getNumberOfPhases", "get_number_of_phases", default=None)
    if nph is None:
        nph = _get(model, "n_phases", max(liquid_index, vapor_index) + 1)
    return int(nph), liquid_index, vapor_index


def _has_rhos(well: Any) -> bool:
    return _get(well, "rhoS", None) is not None or _get(well, "rho_s", None) is not None


def _active(well: Any) -> bool:
    return bool(_get(well, "status", True))


def _component_matrix(wells: list[Any]) -> np.ndarray:
    rows = []
    for well in wells:
        comp = _get(well, "components", None)
        if comp is None:
            raise AssertionError("Wells must have field .components for a compositional model.")
        rows.append(np.asarray(comp, dtype=float).reshape(-1))
    return np.vstack(rows) if rows else np.zeros((0, 0))


def _component_surface_split(model: Any, mass_fraction: np.ndarray, nph: int, liquid_index: int, vapor_index: int) -> np.ndarray:
    components = _get(model, "Components", None)
    if components is None:
        components = _get(model, "components", None)
    if not components:
        comp = np.zeros(nph, dtype=float)
        comp[liquid_index] = 1.0
        comp[vapor_index] = 0.0
        return comp
    eos_components = []
    for component in components:
        if hasattr(component, "surfacePhaseMassFractions") or hasattr(component, "surface_phase_mass_fractions"):
            eos_components.append(component)
    if not eos_components:
        comp = np.zeros(nph, dtype=float)
        comp[liquid_index] = 1.0
        return comp
    table = np.vstack([
        np.asarray(_get(component, "surface_phase_mass_fractions", _get(component, "surfacePhaseMassFractions")), dtype=float).reshape(-1)
        for component in eos_components[: mass_fraction.size]
    ])
    if table.shape[1] < nph:
        table = np.pad(table, ((0, 0), (0, nph - table.shape[1])), constant_values=0.0)
    return mass_fraction @ table[:, :nph]


def validateCompositionalForces(model, forces, it: int | None = None):
    """Validate compositional driving forces and fill missing well ``rhoS``.

    This ports the well-facing behavior of MRST ``validateCompositionalForces``:
    wells must carry ``components`` and active wells without ``rhoS`` are flashed
    at facility separator pressure/temperature to get surface phase densities.
    """
    wells = _wells(forces)
    if not wells:
        return forces

    facility = _facility(model)
    if facility is None:
        raise AssertionError(
            "FacilityModel must be set up before validating driving forces for a compositional problem with wells!"
        )
    _component_matrix(wells)  # MRST asserts this before checking active status.
    missing_active = [well for well in wells if _active(well) and not _has_rhos(well)]
    if not missing_active:
        return forces

    eos = _eos(model)
    p = float(_get(facility, "pressure", _get(facility, "p", np.nan)))
    T = float(_get(facility, "T", _get(facility, "temperature", np.nan)))
    z = _component_matrix(missing_active)
    flash = standaloneFlash(np.full(z.shape[0], p), np.full(z.shape[0], T), z, eos)
    mass_z = _mass_fraction(eos, z)
    nph, liquid_index, vapor_index = _phase_indices(model)

    rho_l = np.asarray(eos.PropertyModel.computeDensity(eos, np.full(z.shape[0], p), flash.x, flash.Z_L, np.full(z.shape[0], T), isLiquid=True), dtype=float).reshape(-1)
    rho_v = np.asarray(eos.PropertyModel.computeDensity(eos, np.full(z.shape[0], p), flash.y, flash.Z_V, np.full(z.shape[0], T), isLiquid=False), dtype=float).reshape(-1)

    for i, well in enumerate(missing_active):
        rho = _surface_densities(model, nph, liquid_index, vapor_index, float(rho_l[i]), float(rho_v[i]))
        _set(well, "rhoS", rho)
        if _get(well, "compi", None) is None:
            compi = _component_surface_split(model, mass_z[i], nph, liquid_index, vapor_index)
            _set(well, "compi", compi)
    return forces


def validate_compositional_forces(model, forces, it: int | None = None):
    """Pythonic alias for :func:`validateCompositionalForces`."""
    return validateCompositionalForces(model, forces, it=it)


__all__ = ["validateCompositionalForces", "validate_compositional_forces"]
