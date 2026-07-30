"""Mixture density for cross-flowing well perforations."""

from __future__ import annotations

import numpy as np


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _sum_perforations(values, perforation_to_well, number_of_wells):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    output = np.zeros((number_of_wells, values.shape[1]), dtype=float)
    for phase in range(values.shape[1]):
        output[:, phase] = np.bincount(
            perforation_to_well,
            weights=values[:, phase],
            minlength=number_of_wells,
        )
    return output


def cross_flow_mixture_density(
    mass_flux, total_volume_flux, surface_mass_flux, mapping
):
    """Compute well mixture densities using MRST's cross-flow convention."""
    perforation_to_well = np.asarray(
        _get(mapping, "perf2well"), dtype=np.int64
    ).ravel()
    number_of_wells = len(_get(mapping, "W"))
    mass_in = _sum_perforations(
        np.maximum(-np.asarray(mass_flux), 0.0),
        perforation_to_well,
        number_of_wells,
    )
    volume_flux = np.asarray(total_volume_flux, dtype=float).ravel()
    volume_in = _sum_perforations(
        np.maximum(-volume_flux, 0.0),
        perforation_to_well,
        number_of_wells,
    ).ravel()
    volume_out = _sum_perforations(
        np.maximum(volume_flux, 0.0),
        perforation_to_well,
        number_of_wells,
    ).ravel()
    surface_mass = np.asarray(surface_mass_flux, dtype=float)
    if surface_mass.ndim == 1:
        surface_mass = surface_mass[:, None]
    total_mass_in = mass_in + np.maximum(surface_mass, 0.0)
    surface_volume = volume_in - volume_out
    denominator = volume_out + np.maximum(surface_volume, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mixture = total_mass_in / denominator[:, None]
    bad = (np.sum(total_mass_in, axis=1) == 0.0) | (volume_out == 0.0)
    mixture[bad, :] = 1.0
    return mixture


crossFlowMixtureDensity = cross_flow_mixture_density

__all__ = ["cross_flow_mixture_density", "crossFlowMixtureDensity"]
