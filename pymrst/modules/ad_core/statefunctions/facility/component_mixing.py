"""Shared component-composition helpers for facility crossflow."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def well_component_composition(
    reservoir: Any, wells: Sequence[Any], phase: int | None = None
) -> np.ndarray:
    """Infer topside component fractions from component or phase controls."""
    ncomp = len(reservoir.components)
    result = np.zeros((len(wells), ncomp), dtype=float)
    for well_index, well in enumerate(wells):
        explicit = getattr(well, "components", None)
        if explicit is not None and phase is None:
            values = np.asarray(explicit, dtype=float).ravel()
            result[well_index, :min(ncomp, values.size)] = values[:ncomp]
        else:
            phase_compi = np.asarray(well.compi, dtype=float).ravel()
            for component_index, component in enumerate(reservoir.components):
                component_phase = getattr(
                    component, "phase_index",
                    getattr(component, "gas_index",
                            getattr(component, "oil_index", component_index)),
                )
                if phase is not None and component_phase != phase:
                    continue
                if component_phase < phase_compi.size:
                    result[well_index, component_index] = phase_compi[
                        component_phase
                    ]
        total = result[well_index].sum()
        if total > 0.0:
            result[well_index] /= total
    return result


def crossflow_composition(
    flux: np.ndarray, topside: np.ndarray, mapping: Any,
    conserve_component_amount: bool = False,
) -> np.ndarray:
    """MRST ``crossFlowMixture`` on a numeric perforation/component matrix."""
    incoming = -np.minimum(np.asarray(flux, dtype=float), 0.0)
    if not np.any(incoming):
        return topside
    sum_in = np.asarray(mapping.perforation_sum @ incoming)
    net = np.asarray(mapping.perforation_sum @ np.sum(flux, axis=1)).ravel()
    net_injection = np.maximum(net, 0.0)
    top_in = net_injection[:, None] * topside
    combined = sum_in + top_in
    total = combined.sum(axis=1)
    output = topside.copy()
    active = total > 0.0
    output[active] = combined[active] / total[active, None]
    if conserve_component_amount:
        active = net_injection > 0.0
        output[active] = (
            combined[active] / net_injection[active, None]
        )
    return output


__all__ = ["crossflow_composition", "well_component_composition"]
