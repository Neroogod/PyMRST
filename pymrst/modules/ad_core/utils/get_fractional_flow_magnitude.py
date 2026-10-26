"""MRST ``ad-core/utils/getFractionalFlowMagnitude.m``.

The magnitude of the fractional-flow derivative in each cell, the wave
speed of a saturation front. With one phase it is 1. Otherwise all
saturations but the last are made AD variables (the last is one minus the
rest), the phase mobilities are evaluated, and the derivatives of the
fractional flows `mob_i/mobT` (i < nph) with respect to those saturations
form an (nph-1)x(nph-1) Jacobian per cell; its largest eigenvalue magnitude
(for two phases, `|df/ds|`) is `F`. MRST evaluates the capillary pressure
but returns an empty capillary term `Q`; so does this port (`Q = None`).

`fractional_flow_magnitude` (in `cfl.py`) is PyMRST's earlier version for
the legacy models.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from pymrst.utils.adi import init_variables_adi


def get_fractional_flow_magnitude(model: Any, state: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """MRST `[F, Q] = getFractionalFlowMagnitude(model, state)`."""
    nph = int(model.water) + int(model.oil) + int(model.gas)
    nc = model.G.cells.num
    Q = None
    if nph == 1:
        F = np.ones((nc, 1))
    else:
        F, Q = _get_fractional_flow_magnitude_internal(model, state)
    F = find_largest_eigenvalue(F)
    Q = None if Q is None else find_largest_eigenvalue(Q)
    return F, Q


def _get_fractional_flow_magnitude_internal(model: Any, state: Any):
    if getattr(model, "FlowPropertyFunctions", None) is None:
        model = model.validate_model()
    state = model.reduce_state(state, True)
    s = np.asarray(model.get_prop(state, "saturation"), dtype=float)
    nph = s.shape[1]
    sat = list(init_variables_adi(*[s[:, i] for i in range(nph - 1)]))
    last = 1
    for i in range(nph - 1):
        last = last - sat[i]
    sat.append(last)
    state = model.set_prop(state, "saturation", sat)
    mob = model.get_props(state, "Mobility")
    mobT = 0
    for i in range(nph):
        mobT = mobT + mob[i]
    nc = model.G.cells.num
    F = np.zeros((nc, (nph - 1)*(nph - 1)))
    for i in range(nph - 1):
        frac = mob[i]/mobT
        offset = (nph - 1)*i
        d = np.column_stack([_diagonal(j, nc) for j in frac.jac])
        F[:, offset:offset + nph - 1] = d
    model.get_prop(state, "CapillaryPressure")      # evaluated, unused, as in MRST
    return F, None


def _diagonal(jac: Any, nc: int) -> np.ndarray:
    if jac is None or np.size(jac) == 0:
        return np.zeros(nc)
    diagonal = getattr(jac, "diagonal", None)
    return np.asarray(diagonal() if callable(diagonal) else np.diag(jac), dtype=float).ravel()


def find_largest_eigenvalue(F: np.ndarray) -> np.ndarray:
    """Per cell, the largest |eigenvalue| of the square matrix in each row of
    `F`; a single column is just its magnitude."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 2 and F.shape[1] > 1:
        nph = int(round(np.sqrt(F.shape[1])))
        # `reshape(F(i, :), nph, nph)` is column-major.
        M = F.reshape(F.shape[0], nph, nph).transpose(0, 2, 1)
        return np.abs(np.linalg.eigvals(M)).max(axis=1)
    return np.abs(F).ravel()


getFractionalFlowMagnitude = get_fractional_flow_magnitude

__all__ = ["find_largest_eigenvalue", "getFractionalFlowMagnitude",
           "get_fractional_flow_magnitude"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.solvers.init_res_sol import init_res_sol

    def test_linear_two_phase_wave_speed():
        """kr = s, equal viscosities: f_w = s_w, so df/ds = 1 everywhere."""
        G = compute_geometry(cart_grid([4], [4.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.3), fluid, gas=False)
        model = model.validate_model()
        state = model.validate_state(init_res_sol(G, 1e7, np.array([0.3, 0.7])))
        F, Q = get_fractional_flow_magnitude(model, state)
        np.testing.assert_allclose(F, 1.0)
        assert Q is None
        print("ok test_linear_two_phase_wave_speed")

    def test_quadratic_two_phase_wave_speed():
        """kr = s^2, equal viscosities: f = s^2/(s^2+(1-s)^2)."""
        G = compute_geometry(cart_grid([3], [3.0]))
        fluid = init_simple_adi_fluid(phases="WO", mu=[1e-3, 1e-3], rho=[1000, 800], n=[2, 2])
        model = GenericBlackOilModel(G, make_rock(G, 1e-12, 0.3), fluid, gas=False)
        model = model.validate_model()
        sw = np.array([0.2, 0.5, 0.8])
        state = model.validate_state(init_res_sol(G, 1e7, np.column_stack([sw, 1 - sw])))
        F, _ = get_fractional_flow_magnitude(model, state)
        d = sw**2 + (1 - sw)**2
        np.testing.assert_allclose(F, (2*sw*d - sw**2*(4*sw - 2))/d**2, rtol=1e-12)
        print("ok test_quadratic_two_phase_wave_speed")

    def test_largest_eigenvalue_of_a_column_major_block():
        F = np.array([[1.0, 0.0, 3.0, -4.0]])          # [[1 3]; [0 -4]]
        np.testing.assert_allclose(find_largest_eigenvalue(F), [4.0])
        np.testing.assert_allclose(find_largest_eigenvalue(np.array([[-2.0], [3.0]])), [2.0, 3.0])
        print("ok test_largest_eigenvalue_of_a_column_major_block")

    test_linear_two_phase_wave_speed()
    test_quadratic_two_phase_wave_speed()
    test_largest_eigenvalue_of_a_column_major_block()
    print("\nAll tests passed!")
