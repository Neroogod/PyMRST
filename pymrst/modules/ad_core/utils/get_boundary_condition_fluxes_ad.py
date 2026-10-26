"""MRST ``ad-core/utils/getBoundaryConditionFluxesAD.m``: phase fluxes across
boundary faces.

For each face of `bc` (on the boundary; `bc.sat` one column per phase):
- **pressure**: Darcy flow through the face transmissibility `T_all` with
  `dP = p_bc - p_cell - rho_avg g dz` (`rho_avg` the mean of inside and outside
  densities; for a compositional model's hydrocarbon phases saturation
  weighted). Outflow carries the cell's mobility and `b`; inflow the total
  mobility times the boundary saturation (or `bc.mob`) and the outside `b`
  (`bc.rho/rhoS`, or the cell's);
- **flux**: a surface rate; injection arrives with `bc.sat`, production
  leaves with the cell's fractional flow;
- any other type: a reservoir rate split between the phases by
  potential-upwinded sequential fluxes, inside and outside mobilities
  stacked.
A face with an all-zero `bc.sat` takes the cell's saturation. Each case is
assigned to its own faces only, as in MRST. Returns `(qSurf, BCTocellMap,
BCcells, qRes)` with `BCTocellMap` the cells-by-faces map. Indices 0-based.

MRST indexes a per-phase pressure value as `p_bnd(isP, i)`, which works for
the first phase only; the port uses each phase's own column.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def get_boundary_condition_fluxes_ad(model: Any, pressure: Sequence[Any], s: Sequence[Any],
                                     mob: Sequence[Any], rho: Sequence[Any], b: Sequence[Any],
                                     bc: Any) -> Tuple[List[Any], sp.csr_array, np.ndarray, List[Any]]:
    """MRST `[qSurf, BCTocellMap, BCcells, qRes] = getBoundaryConditionFluxesAD(...)`."""
    from pymrst.utils.adi import value
    face = np.asarray(_get(bc, "face"), dtype=np.int64).ravel()
    T = np.asarray(model.operators.T_all, dtype=float)[face]
    G = model.G
    nph = int(np.sum(model.get_active_phases()))
    N = np.asarray(G.faces.neighbors, dtype=np.int64)[face]
    bc_sat = np.atleast_2d(np.asarray(_get(bc, "sat"), dtype=float))
    assert bc_sat.shape[1] == nph, (
        f"Wrong number of columns in BC sat field: Expected {nph} columns, but input had "
        f"{bc_sat.shape[1]} columns.")
    assert not np.any(np.all(N >= 0, axis=1)), "bc on internal boundary"
    outside_mob = _get(bc, "mob")
    outside_rho = _get(bc, "rho")
    has_outside_mob = outside_mob is not None
    has_outside_rho = outside_rho is not None
    if not has_outside_mob:
        ss = bc_sat.sum(axis=1)
        assert np.all((ss - 1 < np.sqrt(np.finfo(float).eps)) | (ss < np.sqrt(np.finfo(float).eps))), (
            "Boundary conditions should have 'sat' field which has row-sum 1.")
    BCcells = N.max(axis=1)
    nbc = face.size
    cell_to_bc = sp.csr_array((np.ones(nbc), (np.arange(nbc), BCcells)), shape=(nbc, G.cells.num))
    BCTocellMap = cell_to_bc.T.tocsr()
    if any(str(t).lower() == "topsurfacegrid" for t in np.atleast_1d(getattr(G, "type", []))):
        dzbc = -np.linalg.norm(model.gravity)*(np.asarray(G.cells.z)[BCcells]
                                               - np.asarray(G.faces.z)[face])
    else:
        g = np.asarray(model.get_gravity_vector(), dtype=float)
        dz = np.asarray(G.cells.centroids)[BCcells] - np.asarray(G.faces.centroids)[face]
        dzbc = -(dz @ g)
    types = np.resize(np.array([str(t).lower() for t in np.atleast_1d(_get(bc, "type"))]), nbc)
    isP = types == "pressure"
    isSF = types == "flux"
    isRF = ~(isP | isSF)
    values = np.asarray(_get(bc, "value"), dtype=float)
    values = values.reshape(nbc, -1)
    sat = bc_sat.copy()
    no_sat = np.all(sat == 0, axis=1)
    has_no_sat = bool(no_sat.any())
    mro = {c.__name__ for c in type(model).__mro__}
    is_compositional = "ThreePhaseCompositionalModel" in mro
    is_transport = "TransportNaturalVariablesModel" in mro
    rhoS = np.asarray(model.get_surface_densities(), dtype=float).ravel()
    pressureF = [[None, None] for _ in range(nph)]
    rhoF = [[None, None] for _ in range(nph)]
    bF = [[None, None] for _ in range(nph)]
    mobF = [[None, None] for _ in range(nph)]
    sF = [[None, None] for _ in range(nph)]
    totMob = np.zeros(nbc)
    for i in range(nph):
        p_inside = pressure[i][BCcells]
        p_bnd = p_inside + 0
        col = values[:, i] if values.shape[1] > 1 else values[:, 0]
        p_bnd[isP] = col[isP]
        pressureF[i] = [p_bnd, p_inside]
        b_inside = b[i][BCcells]
        rho_inside = rho[i][BCcells]
        if has_outside_rho:
            rho_bnd = np.asarray(outside_rho, dtype=float)[:, i]
            b_bnd = rho_bnd/rhoS[i]
        else:
            rho_bnd, b_bnd = rho_inside, b_inside
        rhoF[i] = [rho_bnd, rho_inside]
        bF[i] = [b_bnd, b_inside]
        mob_inside = mob[i][BCcells]
        mob_bnd = np.asarray(outside_mob, dtype=float)[:, i] if has_outside_mob else mob_inside
        mobF[i] = [mob_bnd, mob_inside]
        totMob = totMob + mob_inside
        s_inside = s[i][BCcells]
        if has_no_sat:
            s_inside = np.asarray(value(s_inside), dtype=float)
            sat[no_sat, i] = s_inside[no_sat]
        sF[i] = [sat[:, i], s_inside]
    if not has_outside_mob:
        for i in range(nph):
            mobF[i][0] = totMob*sF[i][0]
    if is_transport and is_compositional:
        sT = [sat.sum(axis=1), np.zeros(T.size)]
        for i in range(nph):
            sT[1] = sT[1] + sF[i][1]
    rhoAvgF = []
    for i in range(nph):
        if is_compositional and i > _water_index(model):
            if is_transport:
                sL, sR = sF[i][0]/sT[0], sF[i][1]/sT[1]
            else:
                sL, sR = sF[i][0], sF[i][1]
            sTf = sL + sR
            zero = np.asarray(value(sTf)) == 0
            if np.any(zero):
                sTf = sTf + 0
                sTf[zero] = 1e-8
            rhoAvgF.append((sL*rhoF[i][0] + sR*rhoF[i][1])/sTf)
        else:
            rhoAvgF.append((rhoF[i][0] + rhoF[i][1])/2)
    if np.any(isRF):
        q_ph = _reservoir_rate_fluxes(values, dzbc, rhoAvgF, mobF, T, nph)
    qSurf, qRes = [], []
    for i in range(nph):
        if values.shape[1] == 1:
            bc_v = values[:, 0]
        else:
            assert values.shape[1] == nph, ("Boundary conditions should have either one value "
                                            "or one value per phase, for each face")
            bc_v = values[:, i]
        if hasattr(totMob, "jac"):
            sample = totMob
        else:
            from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad
            sample = get_sample_ad(pressure[i], mobF[i][0], mobF[i][1], T, rhoAvgF[i],
                                   bF[i][0], bF[i][1], dzbc)[0]
        q_s = model.AutoDiffBackend.convert_to_ad(np.zeros(nbc), sample)
        q_r = model.AutoDiffBackend.convert_to_ad(np.zeros(nbc), sample)
        if np.any(isP):
            rho_p = rhoAvgF[i][isP]
            dP = pressureF[i][0][isP] - pressureF[i][1][isP] - rho_p*dzbc[isP]
            inj_dir = np.asarray(value(dP)) > 0
            injP = isP.copy()
            injP[isP] = inj_dir
            if np.any(~inj_dir):
                subs = isP & ~injP
                q_res = mobF[i][1][subs]*T[subs]*dP[~inj_dir]
                q_s[subs] = bF[i][1][subs]*q_res
                q_r[subs] = q_res
            if np.any(inj_dir):
                subs = isP & injP
                if has_outside_mob:
                    q_res = mobF[i][0][subs]*T[subs]*dP[inj_dir]
                else:
                    q_res = totMob[subs]*T[subs]*dP[inj_dir]*sat[subs, i]
                q_s[subs] = bF[i][0][subs]*q_res
                q_r[subs] = q_res
        inj_neu = bc_v > 0
        if np.any(isSF):
            subs = isSF & inj_neu
            if np.any(subs):
                q_s[subs] = bc_v[subs]*sat[subs, i]
                q_r[subs] = bc_v[subs]*sat[subs, i]/bF[i][0][subs]
            subs = isSF & ~inj_neu
            if np.any(subs):
                f = mobF[i][1][subs]/totMob[subs]
                tmp = f*bc_v[subs]
                q_s[subs] = tmp
                q_r[subs] = tmp/bF[i][1][subs]
        if np.any(isRF):
            inj_neu_r = np.asarray(value(q_ph[i])) > 0
            subs = isRF & inj_neu_r
            if np.any(subs):
                tmp = q_ph[i][subs]
                q_s[subs] = tmp*bF[i][0][subs]
                q_r[subs] = tmp
            subs = isRF & ~inj_neu_r
            if np.any(subs):
                tmp = q_ph[i][subs]
                q_s[subs] = tmp*bF[i][1][subs]
                q_r[subs] = tmp
        qSurf.append(q_s)
        qRes.append(q_r)
    return qSurf, BCTocellMap, BCcells, qRes


def _reservoir_rate_fluxes(values, dzbc, rhoAvgF, mobF, T, nph) -> List[Any]:
    """MRST's sequential (potential-upwinded) split of a total reservoir rate."""
    from pymrst.modules.sequential.utils.compute_sequential_fluxes import (
        compute_sequential_fluxes,
    )
    from pymrst.utils.adi import vertcat_adi
    Gp = [dzbc*rhoAvgF[i] for i in range(nph)]
    mobC = [vertcat_adi(mobF[i][1], mobF[i][0]) for i in range(nph)]
    vT = values.sum(axis=1)
    nf = vT.size

    def upstr(flag, v):
        return flag*v[:nf] + (~flag)*v[nf:]

    q_ph = compute_sequential_fluxes(None, Gp, -vT, T, mobC, {}, [], upstr, "potential")
    return [-q for q in q_ph]


def _water_index(model: Any) -> int:
    """MRST's `i > model.water`: 1-based phase index against the water flag."""
    return int(bool(getattr(model, "water", False))) - 1


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


getBoundaryConditionFluxesAD = get_boundary_condition_fluxes_ad
__all__ = ["getBoundaryConditionFluxesAD", "get_boundary_condition_fluxes_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
    from pymrst.params.wells_and_bc import add_bc
    from pymrst.utils.adi import init_variables_adi, value

    G = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    MODEL = SimpleNamespace(G=G, operators=SimpleNamespace(T_all=np.full(G.faces.num, 2.0)),
                            get_active_phases=lambda: np.array([True, True]),
                            get_gravity_vector=lambda: np.zeros(3),
                            get_surface_densities=lambda: np.array([1000.0, 800.0]),
                            AutoDiffBackend=AutoDiffBackend())
    WEST = int(np.flatnonzero(np.isclose(G.faces.centroids[:, 0], 0.0))[0])

    def _inputs():
        p, sw = init_variables_adi(np.array([10.0, 8.0]), np.array([0.5, 0.5]))
        return [p, p], [sw, 1 - sw], [sw*0 + 1.0, sw*0 + 3.0], [sw*0 + 1.0, sw*0 + 1.0]

    def test_pressure_inflow_and_outflow():
        pressure, s, mob, b = _inputs()
        bc = add_bc(None, np.array([WEST]), "pressure", np.array([12.0]), sat=np.array([1.0, 0.0]))
        qS, bmap, cells, qR = get_boundary_condition_fluxes_ad(MODEL, pressure, s, mob, b, b, bc)
        # inflow: total mobility 4 times sat, T 2, dP 2
        assert np.allclose(value(qS[0]), [16.0]) and np.allclose(value(qS[1]), [0.0])
        assert cells.tolist() == [0] and bmap.shape == (2, 1)
        bc.value = np.array([9.0])
        qS, _, _, _ = get_boundary_condition_fluxes_ad(MODEL, pressure, s, mob, b, b, bc)
        assert np.allclose(value(qS[0]), [-2.0]) and np.allclose(value(qS[1]), [-6.0])
        print("ok test_pressure_inflow_and_outflow")

    def test_flux_faces_and_zero_saturation():
        pressure, s, mob, b = _inputs()
        bc = add_bc(None, np.array([WEST]), "flux", np.array([-4.0]), sat=np.array([0.0, 0.0]))
        qS, _, _, _ = get_boundary_condition_fluxes_ad(MODEL, pressure, s, mob, b, b, bc)
        assert np.allclose(value(qS[0]), [-1.0]) and np.allclose(value(qS[1]), [-3.0])
        print("ok test_flux_faces_and_zero_saturation")

    def test_internal_faces_are_refused():
        pressure, s, mob, b = _inputs()
        inner = int(np.flatnonzero(np.all(G.faces.neighbors >= 0, axis=1))[0])
        bc = add_bc(None, np.array([inner]), "flux", np.array([1.0]), sat=np.array([1.0, 0.0]))
        try:
            get_boundary_condition_fluxes_ad(MODEL, pressure, s, mob, b, b, bc)
        except AssertionError:
            print("ok test_internal_faces_are_refused")
            return
        raise AssertionError("expected an AssertionError")

    test_pressure_inflow_and_outflow()
    test_flux_faces_and_zero_saturation()
    test_internal_faces_are_refused()
    print("\nAll tests passed!")
