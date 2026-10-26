"""MRST ``ad-core/utils/computeSourcesAndBoundaryConditionsAD.m``: the source
and boundary-condition contributions of a step, each as MRST's struct of
`phaseMass` (surface rates times surface density, dissolved components
added), `phaseVolume` (reservoir rates), `components` (empty), `mapping` and
`sourceCells`.

The `b` factors come from the densities (`phaseDensitiesTobfactor`). Where a
force injects (total surface rate positive) dissolution uses
`force.dissolution` (0 if absent), elsewhere the cell's ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class ForceContributions:
    """MRST's contribution struct."""

    phaseMass: List[Any]
    phaseVolume: List[Any]
    components: List[Any]
    mapping: Any
    sourceCells: np.ndarray


def compute_sources_and_boundary_conditions_ad(model: Any, pressure: Sequence[Any], s: Sequence[Any],
                                               mob: Sequence[Any], rho: Sequence[Any],
                                               dissolved: Any, forces: Any
                                               ) -> Tuple[ForceContributions, ForceContributions]:
    """MRST `[src, bc] = computeSourcesAndBoundaryConditionsAD(model, pressure, s, mob, rho, dissolved, forces)`."""
    from pymrst.modules.ad_core.utils.get_boundary_condition_fluxes_ad import (
        get_boundary_condition_fluxes_ad,
    )
    from pymrst.modules.ad_core.utils.get_source_fluxes_ad import get_source_fluxes_ad
    from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import phase_densities_to_bfactor
    bc_force, src_force = _get(forces, "bc"), _get(forces, "src")
    has_bc = not _is_empty(bc_force, "face")
    has_src = not _is_empty(src_force, "cell")
    rhoS = np.asarray(model.get_surface_densities(), dtype=float).ravel()
    n = len(mob)
    qVolBC: List[Any] = [None]*n
    qResBC: List[Any] = [None]*n
    qVolSRC: List[Any] = [None]*n
    b: List[Any] = [None]*n
    bcCells = srcCells = np.zeros(0, dtype=np.int64)
    BCTocellMap = BCToSourceMap = None
    if has_bc or has_src:
        b = phase_densities_to_bfactor(rho, rhoS, dissolved)
        if has_bc:
            qVolBC, BCTocellMap, bcCells, qResBC = get_boundary_condition_fluxes_ad(
                model, pressure, s, mob, rho, b, bc_force)
        if has_src:
            qVolSRC, BCToSourceMap, srcCells = get_source_fluxes_ad(model, mob, s, src_force)
    src = _contributions(src_force if has_src else None, qVolSRC, b, rhoS, srcCells, dissolved,
                         BCToSourceMap)
    bc = _contributions(bc_force if has_bc else None, qVolBC, b, rhoS, bcCells, dissolved,
                        BCTocellMap, qResBC)
    return src, bc


def _contributions(force: Any, q_s: List[Any], b: List[Any], rhoS: np.ndarray, cells: np.ndarray,
                   dissolved: Any, mapping: Any, q_r: Optional[List[Any]] = None) -> ForceContributions:
    """MRST's `getContributionsStruct`."""
    from pymrst.utils.adi import value
    n_ph = len(q_s)
    if q_r is None:
        q_r = [None if q_s[i] is None else q_s[i]/b[i][cells] for i in range(n_ph)]
    q_s = list(q_s)
    if dissolved and force is not None:
        q_s0 = list(q_s)
        q_t = 0
        for q in q_s:
            q_t = q_t + np.asarray(value(q), dtype=float)
        is_inj = np.asarray(q_t) > 0
        ds_mat = _get(force, "dissolution")
        for i in range(n_ph):
            for j in range(n_ph):
                r_cell = dissolved[i][j]
                if r_cell is None or (not hasattr(r_cell, "jac") and np.size(r_cell) == 0):
                    continue
                if ds_mat is not None:
                    ds = np.asarray(ds_mat)
                    r_inj = ds[:, i, j] if ds.ndim == 3 else ds[i, j]
                else:
                    r_inj = 0
                r = is_inj*r_inj + (~is_inj)*r_cell[cells]
                q_s[i] = q_s[i] + r*q_s0[j]
    mm = mapping[cells, :] if mapping is not None else None
    mass = [None if q is None else q*rhoS[i] for i, q in enumerate(q_s)]
    return ForceContributions(phaseMass=mass, phaseVolume=list(q_r), components=[], mapping=mm,
                              sourceCells=cells)


def _is_empty(force: Any, field: str) -> bool:
    return force is None or np.size(_get(force, field)) == 0


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


computeSourcesAndBoundaryConditionsAD = compute_sources_and_boundary_conditions_ad
__all__ = ["ForceContributions", "computeSourcesAndBoundaryConditionsAD",
           "compute_sources_and_boundary_conditions_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.params.wells_and_bc import add_source

    MODEL = SimpleNamespace(get_active_phases=lambda: np.array([True, True]),
                            get_surface_densities=lambda: np.array([1000.0, 800.0]),
                            G=SimpleNamespace(cells=SimpleNamespace(num=3)))

    def test_source_mass_and_volume():
        src = add_source(None, np.array([1]), np.array([2.0]), sat=np.array([1.0, 0.0]))
        rho = [np.full(3, 500.0), np.full(3, 800.0)]
        cs, cb = compute_sources_and_boundary_conditions_ad(
            MODEL, [None]*2, [None]*2, [np.ones(3)]*2, rho, [], {"src": src, "bc": None})
        assert np.allclose(cs.phaseMass[0], [2000.0]) and np.allclose(cs.phaseVolume[0], [4.0])
        assert cs.sourceCells.tolist() == [1] and cs.mapping.shape == (1, 1)
        assert cb.sourceCells.size == 0 and cb.mapping is None
        print("ok test_source_mass_and_volume")

    def test_nothing_to_do_without_forces():
        cs, cb = compute_sources_and_boundary_conditions_ad(MODEL, [None]*2, [None]*2, [None]*2,
                                                            [None]*2, [], {})
        assert cs.phaseMass == [None, None] and cb.sourceCells.size == 0
        print("ok test_nothing_to_do_without_forces")

    test_source_mass_and_volume()
    test_nothing_to_do_without_forces()
    print("\nAll tests passed!")
