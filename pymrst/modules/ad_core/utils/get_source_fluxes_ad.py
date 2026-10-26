"""MRST ``ad-core/utils/getSourceFluxesAD.m``: phase surface rates of source
terms. An injecting source (`rate > 0`) delivers its rate split by `src.sat`;
a producing one takes the cell's fractional flow. The rates become AD only
when some source produces, as in MRST. Returns `(qSurf, BCTocellMap, cells)`,
the map cells-by-sources. Indices 0-based."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


def get_source_fluxes_ad(model: Any, mob: Sequence[Any], s: Sequence[Any],
                         src: Any) -> Tuple[List[Any], sp.csr_array, np.ndarray]:
    """MRST `[qSurf, BCTocellMap, cells] = getSourceFluxesAD(model, mob, s, src)`."""
    from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
    n_ph = int(np.sum(model.get_active_phases()))
    sat = np.atleast_2d(np.asarray(_get(src, "sat"), dtype=float))
    assert sat.shape[1] == n_ph
    cells = np.asarray(_get(src, "cell"), dtype=np.int64).ravel()
    rate = np.asarray(_get(src, "rate"), dtype=float).ravel()
    nsrc = cells.size
    inj = rate > 0
    prod = ~inj
    tot_mob = 0
    if np.any(prod):
        for i in range(n_ph):
            tot_mob = tot_mob + mob[i]
    q_surf = []
    for i in range(n_ph):
        q = np.zeros(nsrc)
        if np.any(prod):
            q = double_to_adi(q, tot_mob)
        if np.any(inj):
            q[inj] = rate[inj]*sat[inj, i]
        if np.any(~inj):
            c = cells[~inj]
            f = mob[i][c]/tot_mob[c]
            q[~inj] = q[~inj] + f*rate[~inj]
        q_surf.append(q)
    cell_to_bc = sp.csr_array((np.ones(nsrc), (np.arange(nsrc), cells)), shape=(nsrc, model.G.cells.num))
    return q_surf, cell_to_bc.T.tocsr(), cells


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


getSourceFluxesAD = get_source_fluxes_ad
__all__ = ["getSourceFluxesAD", "get_source_fluxes_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.params.wells_and_bc import add_source
    from pymrst.utils.adi import init_variables_adi, value

    MODEL = SimpleNamespace(get_active_phases=lambda: np.array([True, True]),
                            G=SimpleNamespace(cells=SimpleNamespace(num=3)))

    def test_injection_split_and_production_fractional_flow():
        (sw,) = init_variables_adi(np.array([0.2, 0.5, 0.8]))
        mob = [sw*1.0, (1 - sw)*1.0]
        src = add_source(None, np.array([0, 2]), np.array([2.0, -1.0]),
                         sat=np.array([[0.25, 0.75], [0.0, 0.0]]))
        q, smap, cells = get_source_fluxes_ad(MODEL, mob, [sw, 1 - sw], src)
        assert np.allclose(value(q[0]), [0.5, -0.8]) and np.allclose(value(q[1]), [1.5, -0.2])
        assert smap.shape == (3, 2) and cells.tolist() == [0, 2]
        print("ok test_injection_split_and_production_fractional_flow")

    def test_injection_only_stays_numeric():
        src = add_source(None, np.array([1]), np.array([3.0]), sat=np.array([1.0, 0.0]))
        q, _, _ = get_source_fluxes_ad(MODEL, [None, None], [None, None], src)
        assert not hasattr(q[0], "jac") and np.allclose(q[0], [3.0])
        print("ok test_injection_only_stays_numeric")

    test_injection_split_and_production_fractional_flow()
    test_injection_only_stays_numeric()
    print("\nAll tests passed!")
