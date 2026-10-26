"""Compositional equilibrium-region construction.

Port of MRST ``getInitializationRegionsCompositional.m``.  The returned
density closures retain the pressure/depth signature used by equilibrium
initialisation, so temperature and phase compositions may vary with depth.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.initialization.regions import (
    get_initialization_regions_base_ad,
    get_min_max_phase_saturations_ad,
)


def _function(value, name: str) -> Callable:
    if callable(value):
        return value
    if value is None:
        raise ValueError(f"{name} must be supplied for an active hydrocarbon phase")
    constant = np.asarray(value, dtype=float)
    if constant.ndim == 0:
        return lambda p, z: np.zeros_like(np.asarray(z, dtype=float)) + constant
    return lambda p, z: np.broadcast_to(
        constant, (np.atleast_1d(np.asarray(z)).size, constant.size)
    )


def _phase_names(model: Any) -> tuple[str, ...]:
    if hasattr(model, "getPhaseNames"):
        return tuple(str(x).upper() for x in model.getPhaseNames())
    return tuple(str(x)[0].upper() for x in model.fluid.phases)


def _phase_density(model: Any, p, T, composition, is_liquid: bool):
    eos = model.EOSModel
    p = np.atleast_1d(np.asarray(p, dtype=float))
    T = np.broadcast_to(np.asarray(T, dtype=float), p.shape)
    x = np.asarray(composition, dtype=float)
    if x.ndim == 1:
        x = np.broadcast_to(x, (p.size, x.size))
    if eos.__class__.__name__ == "EquilibriumConstantModel":
        Z = np.full(p.shape, np.nan)
    else:
        aij, bi = eos.getMixingParameters(
            p, T, eos.CompositionalMixture.acentricFactors, False
        )
        si, a, b = eos.getPhaseMixCoefficients(x, aij, bi)
        Z = eos.computeCompressibilityZ(p, x, a, b, si, bi, is_liquid)
    return np.asarray(
        eos.PropertyModel.computeDensity(eos, p, x, Z, T, is_liquid),
        dtype=float,
    )


def get_initialization_regions_compositional(
    model: Any,
    contacts: Sequence[float],
    *,
    cells: Optional[Sequence[int]] = None,
    T: Any = 303.15,
    x: Any = None,
    y: Any = None,
    datum_depth: float = 0.0,
    datum_pressure: float = 101325.0,
    contacts_pc: Optional[Sequence[float]] = None,
    pc_functions=None,
    pc_sign=None,
    pc_scale=None,
    s_min=None,
    s_max=None,
):
    """Build one MRST-style compositional equilibration region."""
    nc = int(model.G.cells.num)
    cells = np.arange(nc, dtype=np.int64) if cells is None else np.asarray(cells, dtype=np.int64).ravel()
    if cells.size == 0:
        raise ValueError("cells must contain at least one cell")
    names = _phase_names(model)
    temperature = _function(T, "T")
    liquid = _function(x, "x") if "O" in names else None
    vapor = _function(y, "y") if "G" in names else None
    rho = [None] * len(names)

    if "W" in names:
        ix = names.index("W")
        fluid = model.fluid
        b_w = getattr(fluid, "bW", getattr(fluid, "b_w", None))
        rho_ws = getattr(fluid, "rhoWS", getattr(fluid, "rho_ws", None))
        if b_w is None or rho_ws is None:
            raise AttributeError("water-active compositional model requires fluid.bW and fluid.rhoWS")
        rho_w = rho_ws() if callable(rho_ws) else rho_ws
        rho[ix] = lambda p, z: np.asarray(b_w(p), dtype=float) * float(np.asarray(rho_w).reshape(-1)[0])
    if "O" in names:
        ix = names.index("O")
        rho[ix] = lambda p, z: _phase_density(
            model, p, temperature(p, z), liquid(p, z), True
        )
    if "G" in names:
        ix = names.index("G")
        rho[ix] = lambda p, z: _phase_density(
            model, p, temperature(p, z), vapor(p, z), False
        )

    if s_min is None or s_max is None:
        dmin, dmax = get_min_max_phase_saturations_ad(model)
        s_min = dmin if s_min is None else s_min
        s_max = dmax if s_max is None else s_max
    reference = names.index("O") if "O" in names else 0
    return get_initialization_regions_base_ad(
        model, rho, contacts, datum_depth=datum_depth,
        datum_pressure=datum_pressure, cells=cells,
        pc_functions=pc_functions, pc_sign=pc_sign,
        contacts_pc=contacts_pc, reference_index=reference,
        pc_scale=pc_scale, s_min=s_min, s_max=s_max,
    )


getInitializationRegionsCompositional = get_initialization_regions_compositional

__all__ = [
    "get_initialization_regions_compositional",
    "getInitializationRegionsCompositional",
]
