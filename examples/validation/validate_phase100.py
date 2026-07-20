"""Validate small-strain von Mises plasticity helpers."""

import numpy as np

from pymrst.modules.ad_mechanics import (
    empty_plasticity_state,
    isotropic_elastic_stress,
    von_mises_equivalent,
    von_mises_return_map,
)


def main():
    print("Phase 100: ad_mechanics plasticity material update")
    young = 2.0e9
    nu = 0.25
    sig_y = 1.0e6
    hardening = 2.0e8

    elastic_strain = np.array([1.0e-5, -0.2e-5, 0.0, 0.0, 0.0, 0.0])
    stress_el, state_el, dg_el = von_mises_return_map(
        elastic_strain, young, nu, sig_y, hardening
    )
    assert not state_el.yielded[0]
    assert np.allclose(dg_el, 0.0)
    assert np.allclose(stress_el, isotropic_elastic_stress(elastic_strain, young, nu))

    strains = np.array([
        [2.0e-3, -1.0e-3, -1.0e-3, 0.0, 0.0, 0.0],
        [1.5e-3, 0.0, -1.5e-3, 0.0, 0.0, 0.4e-3],
    ])
    state0 = empty_plasticity_state(strains.shape[0])
    stress, state, dg = von_mises_return_map(
        strains, young, nu, sig_y, hardening, state0
    )
    seq = von_mises_equivalent(stress)
    surface = sig_y + hardening * state.eq_plastic_strain
    assert np.all(state.yielded)
    assert np.all(dg > 0.0)
    assert np.allclose(seq, surface, rtol=1e-10, atol=1e-6)

    stress2, state2, dg2 = von_mises_return_map(
        1.15 * strains, young, nu, sig_y, hardening, state
    )
    assert np.all(state2.eq_plastic_strain >= state.eq_plastic_strain)
    assert np.all(dg2 >= 0.0)
    assert np.all(von_mises_equivalent(stress2) <= (
        sig_y + hardening * state2.eq_plastic_strain + 1e-6
    ))

    plane = np.array([2.0e-3, -1.0e-3, 0.3e-3])
    stress_plane, state_plane, dg_plane = von_mises_return_map(
        plane, young, nu, sig_y
    )
    assert stress_plane.shape == (6,)
    assert state_plane.plastic_strain.shape == (1, 6)
    assert dg_plane.shape == (1,)
    print(
        "  ok plastic return: "
        f"max_eqp={np.max(state2.eq_plastic_strain):.3e}, "
        f"plane_seq={von_mises_equivalent(stress_plane)[0]:.3e}"
    )
    print("Phase 100 validation passed: von Mises plasticity is stable.")


if __name__ == "__main__":
    main()
