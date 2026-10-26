"""MRST ``ad-core/utils/addFluxesFromSourcesAndBC.m``: boundary and source
fluxes subtracted from the phase equations (surface rates; `b` from the
densities over the surface densities). Warns that dissolved gas or vaporised
oil is not fully supported. Returns `(eqs, qBC, qRes, BCTocellMap, qSRC,
srcCells, bcCells)`."""

from __future__ import annotations

import warnings
from typing import Any, List, Sequence

import numpy as np


def add_fluxes_from_sources_and_bc(model: Any, eqs: List[Any], pressure: Sequence[Any],
                                   rho: Sequence[Any], mob: Sequence[Any], s: Sequence[Any],
                                   forces: Any):
    """MRST `[eqs, qBC, qRes, BCTocellMap, qSRC, srcCells, bcCells] = addFluxesFromSourcesAndBC(...)`."""
    from pymrst.modules.ad_core.utils.get_boundary_condition_fluxes_ad import (
        get_boundary_condition_fluxes_ad,
    )
    from pymrst.modules.ad_core.utils.get_source_fluxes_ad import get_source_fluxes_ad
    n = len(mob)
    qBC: List[Any] = [None]*n
    qSRC: List[Any] = [None]*n
    qRes: List[Any] = [None]*n
    BCTocellMap = srcCells = bcCells = None
    bc, src = _get(forces, "bc"), _get(forces, "src")
    if bc is None and src is None:
        return eqs, qBC, qRes, BCTocellMap, qSRC, srcCells, bcCells
    if getattr(model, "disgas", False) or getattr(model, "vapoil", False):
        warnings.warn("Boundary conditions and source terms do not fully support problems with "
                      "vapoil or disgas active!")
    eqs = list(eqs)
    if bc is not None:
        rhoS = np.asarray(model.get_surface_densities(), dtype=float).ravel()
        b = [rho[i]/rhoS[i] for i in range(n)]
        qBC, BCTocellMap, bcCells, qRes = get_boundary_condition_fluxes_ad(
            model, pressure, s, mob, rho, b, bc)
        for i in range(len(qBC)):
            if eqs[i] is None:
                continue
            from pymrst.utils.adi import ad_matmul
            eqs[i] = eqs[i] - ad_matmul(BCTocellMap, qBC[i])
    if src is not None:
        qSRC, _, srcCells = get_source_fluxes_ad(model, mob, s, src)
        for i in range(len(qSRC)):
            if eqs[i] is None:
                continue
            eqs[i] = eqs[i] + 0
            eqs[i][srcCells] = eqs[i][srcCells] - qSRC[i]
    return eqs, qBC, qRes, BCTocellMap, qSRC, srcCells, bcCells


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


addFluxesFromSourcesAndBC = add_fluxes_from_sources_and_bc
__all__ = ["addFluxesFromSourcesAndBC", "add_fluxes_from_sources_and_bc"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.params.wells_and_bc import add_source

    MODEL = SimpleNamespace(get_active_phases=lambda: np.array([True, True]),
                            G=SimpleNamespace(cells=SimpleNamespace(num=3)))

    def test_sources_are_subtracted_from_their_cells():
        src = add_source(None, np.array([2]), np.array([5.0]), sat=np.array([0.4, 0.6]))
        eqs, *_ = add_fluxes_from_sources_and_bc(MODEL, [np.ones(3), np.ones(3)], [None]*2,
                                                 [None]*2, [None]*2, [None]*2, {"src": src})
        assert np.allclose(eqs[0], [1, 1, -1]) and np.allclose(eqs[1], [1, 1, -2])
        print("ok test_sources_are_subtracted_from_their_cells")

    def test_no_forces_returns_the_equations_unchanged():
        eqs = [np.ones(2)]
        out = add_fluxes_from_sources_and_bc(MODEL, eqs, [None], [None], [None], [None], {})
        assert out[0] is eqs and out[3] is None
        print("ok test_no_forces_returns_the_equations_unchanged")

    test_sources_are_subtracted_from_their_cells()
    test_no_forces_returns_the_equations_unchanged()
    print("\nAll tests passed!")
