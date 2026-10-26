"""MRST ``ad-core/utils/getConvergenceValuesCNV.m``.

The reservoir convergence measures MRST's black-oil models use: per phase,
CNV (the largest cell residual relative to its pore volume) and MB (the
material balance over the domain), each converted back to reservoir volume
by the phase's average formation volume factor and multiplied by `dt`.

    CNV_i = B_avg_i * dt * max|r_i / pv|
    MB_i  = |B_avg_i * dt * sum r_i| / sum pv

Returns `(values, names, tolerances, evaluated)`: CNV for every phase with an
equation, then MB for the same phases, and a mask of the equations measured.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from pymrst.utils.adi import value


def get_convergence_values_cnv(model: Any, problem: Any
                               ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    state = problem.state
    tol_mb = model.toleranceMB
    tol_cnv = model.toleranceCNV
    evaluated = np.zeros(problem.num_eq(), dtype=bool)
    pvt = model.PVTPropertyFunctions
    b = pvt.get(model, state, "ShrinkageFactors")
    pv = np.asarray(value(pvt.get(model, state, "PoreVolume")), dtype=float)
    pvtot = pv.sum()
    if "cnvscale" in state:
        pv = pv*state["cnvscale"]
    nph = len(b)
    cnv = np.zeros(nph)
    mb = np.zeros(nph)
    short = model.get_phase_names()
    long_names = model.get_long_phase_names()
    dt = problem.dt
    is_mass = model.isGenericReservoirModel
    rho_s = pvt.get(model, state, "SurfaceDensity") if is_mass else None
    active = np.ones(nph, dtype=bool)
    for ph in range(nph):
        eq_ix = problem.index_of_equation_name(long_names[ph])
        if isinstance(eq_ix, np.ndarray):
            eq_ix = int(np.flatnonzero(eq_ix)[0]) if eq_ix.any() else None
        active[ph] = eq_ix is not None
        if not active[ph]:
            continue
        eq = np.asarray(value(problem.equations[eq_ix]), dtype=float)
        if is_mass:
            eq = eq/np.asarray(value(rho_s[ph]), dtype=float)
        B = 1.0 if b[ph] is None else 1.0/np.asarray(value(b[ph]), dtype=float)
        b_avg = float(np.mean(B))
        worst = np.abs(eq)/pv
        m_ix = int(np.argmax(worst))
        cnv[ph] = b_avg*dt*worst[m_ix]
        mb[ph] = abs(b_avg*dt*eq.sum())/pvtot
        evaluated[eq_ix] = True
        if model.verbose and model.verbose > 1:
            print(f"CNV_{short[ph]}: {m_ix + 1} is the worst cell at {cnv[ph]:g}")
    n_active = int(active.sum())
    tolerances = np.concatenate([np.full(n_active, tol_cnv), np.full(n_active, tol_mb)])
    values = np.concatenate([cnv[active], mb[active]])
    names = ([f"CNV_{short[i]}" for i in range(nph) if active[i]]
             + [f"MB_{short[i]}" for i in range(nph) if active[i]])
    return values, names, tolerances, evaluated


getConvergenceValuesCNV = get_convergence_values_cnv

__all__ = ["get_convergence_values_cnv", "getConvergenceValuesCNV"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _problem(r_w, r_o, dt=2.0):
        names = ["water", "oil", "closureWells"]
        return SimpleNamespace(
            state={}, dt=dt, equations=[np.asarray(r_w), np.asarray(r_o), np.zeros(1)],
            num_eq=lambda: 3,
            index_of_equation_name=lambda n: names.index(n) if n in names else None)

    def _model():
        pvt = SimpleNamespace(get=lambda m, s, n: {
            "ShrinkageFactors": [np.array([1.0, 1.0]), np.array([0.5, 0.5])],
            "PoreVolume": np.array([1.0, 2.0])}[n])
        return SimpleNamespace(toleranceMB=1e-7, toleranceCNV=1e-3, verbose=False,
                               PVTPropertyFunctions=pvt, isGenericReservoirModel=False,
                               get_phase_names=lambda: "WO",
                               get_long_phase_names=lambda: ["water", "oil"])

    def test_cnv_is_the_worst_scaled_cell_and_mb_the_domain_sum():
        values, names, tol, evaluated = get_convergence_values_cnv(
            _model(), _problem([0.1, -0.4], [0.3, 0.3]))
        assert names == ["CNV_W", "CNV_O", "MB_W", "MB_O"]
        # Water: B = 1, worst |r|/pv = max(0.1, 0.2); oil: B = 2.
        np.testing.assert_allclose(values, [2*0.2, 2*2*0.3, 2*0.3/3, 2*2*0.6/3])
        np.testing.assert_allclose(tol, [1e-3, 1e-3, 1e-7, 1e-7])
        np.testing.assert_array_equal(evaluated, [True, True, False])
        print("ok test_cnv_is_the_worst_scaled_cell_and_mb_the_domain_sum")

    test_cnv_is_the_worst_scaled_cell_and_mb_the_domain_sum()
    print("\nAll tests passed!")
