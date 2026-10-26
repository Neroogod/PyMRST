"""MRST ``sequential/utils/computeSequentialFluxes.m``.

Phase (and component) fluxes for transport at a given total flux `vT`: the
mobilities (and component fractions) are upwinded with the flags of
`getSaturationUpwind`, and each phase gets its fractional flow of
`vT + T sum_{j != i} mob_j (G_i - G_j)`. With potential upwinding (or equal
viscous and gravity flags) that is one evaluation; with hybrid upwinding the
viscous and gravity parts are upwinded and added separately ('hybrid_combined'
forms the component fluxes from the summed phase fluxes). Mobilities may
keep their derivatives.

The legacy oil-water sequential solver (`seq_model.py`) builds its fluxes
itself.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence

import numpy as np

from pymrst.utils.adi import value
from pymrst.utils.numel_value import numel_value


def compute_sequential_fluxes(state: Any, G: Sequence[Any], vT: Any, T: Any,
                              mob: Sequence[Any], rho: Any, components: Sequence[Any],
                              upstr: Callable, upwindType: str, nargout: int = 1):
    """MRST `[q_phase, q_components] = computeSequentialFluxes(...)`; the
    component fluxes are returned too with `nargout=2`."""
    from pymrst.modules.sequential.utils.get_saturation_upwind import get_saturation_upwind
    nph = len(mob)
    ncomp = len(components)
    flag_v, flag_g = get_saturation_upwind(upwindType, state, G, vT, T, mob, upstr)
    is_equal = upwindType.lower() == "potential" or np.array_equal(flag_v, flag_g)
    components_f = _upstream_weight_compositions(components, upstr, flag_v, nph, ncomp)
    mob_f = _upstream_weight_phases(mob, upstr, flag_v, nph)
    q_components = None
    if is_equal:
        q_phase = _compute_phase_volumetric_fluxes(vT, T, mob_f, G)
        if nargout > 1:
            q_components = _compute_component_fluxes(q_phase, components_f)
    else:
        q_visc = _compute_phase_volumetric_fluxes(vT, T, mob_f)
        mob_g = _upstream_weight_phases(mob, upstr, flag_g, nph)
        q_grav = _compute_phase_volumetric_fluxes(0, T, mob_g, G)
        q_phase = [q_visc[i] + q_grav[i] for i in range(nph)]
        if nargout > 1:
            if upwindType.lower() == "hybrid_combined":
                q_components = _compute_component_fluxes(q_phase, components_f)
            else:
                q_v = _compute_component_fluxes(q_visc, components_f)
                q_g = _compute_component_fluxes(q_grav, components_f)
                q_components = [q_v[i] + q_g[i] for i in range(ncomp)]
    return (q_phase, q_components) if nargout > 1 else q_phase


def _upstream_weight_phases(v: Sequence[Any], upstr: Callable, flag: np.ndarray,
                            nph: int) -> List[Any]:
    flag = np.asarray(flag)
    return [x if numel_value(x) == flag.shape[0] else upstr(flag[:, i], x)
            for i, x in enumerate(v[:nph])]


def _upstream_weight_compositions(components: Sequence[Any], upstr: Callable, flag: np.ndarray,
                                  nph: int, ncomp: int) -> List[List[Any]]:
    flag = np.asarray(flag)
    out = [[None]*nph for _ in range(ncomp)]
    for i in range(nph):
        for j in range(ncomp):
            c = components[j][i]
            if c is not None and np.size(value(c)):
                out[j][i] = upstr(flag[:, i], c)
    return out


def _compute_component_fluxes(q: Sequence[Any], components: Sequence[Any]) -> List[Any]:
    out = []
    for comp in components:
        qc = np.zeros(np.shape(value(q[0])))
        for p, c in enumerate(comp):
            if c is not None:
                qc = qc + q[p]*c
        out.append(qc)
    return out


def _compute_phase_volumetric_fluxes(vT: Any, T: Any, mob_f: Sequence[Any],
                                     G: Any = None) -> List[Any]:
    nph = len(mob_f)
    mob_t = np.zeros(np.shape(vT))
    for m in mob_f:
        mob_t = mob_t + m
    q = []
    for i in range(nph):
        f = mob_f[i]/mob_t
        if G is None or len(G) == 0:
            q.append(f*vT)
        else:
            g = np.zeros(np.shape(vT))
            for j in range(nph):
                if i != j:
                    g = g + mob_f[j]*(G[i] - G[j])
            q.append(f*(vT + T*g))
    return q


computeSequentialFluxes = compute_sequential_fluxes

__all__ = ["computeSequentialFluxes", "compute_sequential_fluxes"]


if __name__ == "__main__":
    def _upstr(nf):
        return lambda flag, v: flag*v[:nf] + (~flag)*v[nf:]

    def test_without_gravity_the_fractional_flow_splits_vt():
        """Mobilities stacked [inside; outside]; positive vT takes the inside
        ones: q_i = mob_i/sum(mob) * vT."""
        mob = [np.array([1.0, 9.0]), np.array([3.0, 1.0])]
        q = compute_sequential_fluxes(None, [np.zeros(1), np.zeros(1)], np.array([2.0]),
                                      np.array([1.0]), mob, None, [], _upstr(1), "potential")
        np.testing.assert_allclose([q[0][0], q[1][0]], [0.5, 1.5])
        print("ok test_without_gravity_the_fractional_flow_splits_vt")

    def test_hybrid_adds_a_viscous_and_a_gravity_part():
        mob = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        q = compute_sequential_fluxes(None, [np.array([1.0]), np.array([-1.0])], np.array([0.0]),
                                      np.array([1.0]), mob, None, [], _upstr(1), "hybrid")
        np.testing.assert_allclose(q[0] + q[1], 0.0, atol=1e-15)
        assert q[0][0] > 0 > q[1][0]
        print("ok test_hybrid_adds_a_viscous_and_a_gravity_part")

    test_without_gravity_the_fractional_flow_splits_vt()
    test_hybrid_adds_a_viscous_and_a_gravity_part()
    print("\nAll tests passed!")
