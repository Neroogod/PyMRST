"""Saturation CFL estimate (MRST ad-core `estimateSaturationCFL`).

An explicit or adaptive-implicit (AIM) scheme is stable only where the
saturation front does not cross more than one cell in a timestep. The CFL
number measures exactly that per cell:

    cfl = (dt / pv) * (outgoing fractional-flow rate)

and a scheme uses it to pick the timestep (keep `max(cfl) <= 1`) or to decide
which cells must be solved implicitly (those with `cfl > 1`). It is the same
Buckley-Leverett stability number every reservoir-simulation course derives,
computed on the discrete grid.

The rate through a face is `|v_T| * |df/ds|` - the total velocity times the
*wave speed*, the derivative of the fractional flow with respect to saturation.
The wave speed is taken from the upstream cell, and each cell accumulates the
rate of the faces it is upstream of, because that is the saturation leaving it.
With capillary pressure a diffusion term is added; that case is not yet handled
here and is flagged.

This ports `estimateSaturationCFL` and the two-phase branch of
`getFractionalFlowMagnitude`. For three phases the wave speed is the largest
eigenvalue of the fractional-flow Jacobian - noted where it would go.
"""

from typing import Any, Optional

import numpy as np

from pymrst.utils.adi import init_variables_adi, value


def fractional_flow_derivative(model: Any, sw) -> np.ndarray:
    r"""`|df_w/ds_w|` per cell - the Buckley-Leverett wave speed.

    `f_w = lambda_w / (lambda_w + lambda_o)` with the phase mobilities from the
    fluid's relative permeabilities, differentiated with respect to water
    saturation by seeding it as the AD variable. This is MRST's
    `getFractionalFlowMagnitude` for two phases; for three the magnitude is the
    largest eigenvalue of the (n-1)x(n-1) fractional-flow Jacobian, which is
    where that branch would attach.
    """
    if model.n_phases != 2:
        raise NotImplementedError(
            "saturation CFL is implemented for two-phase flow; the three-phase "
            "wave speed is the largest eigenvalue of the fractional-flow "
            "Jacobian and is not yet ported"
        )
    sw = np.asarray(value(sw), dtype=float).ravel()
    (sw_ad,) = init_variables_adi(sw)
    so_ad = 1.0 - sw_ad

    fluid = getattr(model, "raw_fluid", model.fluid)
    mu = fluid.mu if hasattr(fluid, "mu") else None
    if mu is not None:
        mu_w, mu_o = float(mu[0]), float(mu[1])
        lam_w = fluid.kr_w(sw_ad) / mu_w
        lam_o = fluid.kr_o(so_ad) / mu_o
    else:                                    # deck fluid: mu is p-dependent
        p = np.asarray(value(model_pressure(model, sw)), float)
        lam_w = fluid.kr_w(sw_ad) / fluid.mu_w(p)
        lam_o = fluid.kr_o(so_ad) / fluid.mu_o(p, None, None)

    fw = lam_w / (lam_w + lam_o)
    return np.abs(np.asarray(fw.jac[0].diagonal(), dtype=float))


def model_pressure(model, fallback):                      # pragma: no cover
    """A pressure to evaluate a deck fluid's viscosity at - unused by the
    two-phase simple-fluid path, present so the branch above reads clearly."""
    return np.full(np.asarray(value(fallback)).shape,
                   getattr(model, "p_ref_scalar", 200e5))


def estimate_saturation_cfl(model: Any, state, dt: float,
                            flux: Optional[np.ndarray] = None,
                            use_inflow: bool = False) -> np.ndarray:
    r"""The per-cell saturation CFL number (MRST `estimateSaturationCFL`).

    `flux` is the interior-face phase flux, shape `(n_faces, n_phases)`; when
    omitted it is evaluated from the model's `PhaseFlux` at `state`. Returns the
    CFL of every cell - a scheme keeps `max(cfl) <= 1` for stability, or solves
    the cells above 1 implicitly.
    """
    ops = model.operators
    pv = np.asarray(ops.pv, dtype=float)
    n = ops.N
    left, right = n[:, 0], n[:, 1]

    if flux is None:
        ad = model.init_state_ad(state)
        v = np.column_stack([np.asarray(value(x), float)
                             for x in model.get_props(ad, "PhaseFlux")])
    else:
        v = np.atleast_2d(np.asarray(flux, dtype=float))

    v_t = v.sum(axis=1)
    # a face where the phases flow in opposite directions (counter-current):
    # neither all-in nor all-out, so no single upstream cell
    xflow = ~(np.all(v >= 0, axis=1) | np.all(v <= 0, axis=1))

    sw = state["sW"] if "sW" in state else model.get_props(
        model.init_state_ad(state), "Saturations")[0]
    frac = fractional_flow_derivative(model, sw)

    flag = v_t > 0                           # total flow from left to right
    # wave speed at the face: from the upstream cell, or the larger of the two
    # where the flow is counter-current
    df_face = np.where(flag, frac[left], frac[right])
    if np.any(xflow):
        df_face[xflow] = np.maximum(np.abs(frac[left[xflow]]),
                                    np.abs(frac[right[xflow]]))

    rate_face = np.abs(v_t) * np.abs(df_face)

    # each cell takes the rate of the faces it is *upstream* of - the saturation
    # that leaves it. `use_inflow` counts incoming rate instead.
    f = ~flag if use_inflow else flag
    nc = model.G.cells.num
    rate_cell = np.zeros(nc)
    np.add.at(rate_cell, left, rate_face * (f | xflow))
    np.add.at(rate_cell, right, rate_face * (~f | xflow))

    return (dt / pv) * rate_cell


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model(nx=20):
        g = compute_geometry(cart_grid([nx, 1, 1], [200.0, 10.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 1.0]) * CENTI * POISE, rho=[1000.0, 1000.0],
            n=[2, 2], c=[0.0, 0.0], p_ref=200 * BARSA)
        return g, rock, fluid

    def test_fractional_flow_derivative_quadratic_relperm():
        """With kr = s^2, equal viscosities, f_w = sw^2/(sw^2 + so^2). Check the
        analytic derivative at a couple of saturations."""
        g, rock, fluid = _model()
        m = GenericBlackOilModel(g, rock, fluid, [])
        sw = np.array([0.3, 0.5, 0.7])
        F = fractional_flow_derivative(m, sw)

        def analytic(s):
            so = 1 - s
            fw = s**2 / (s**2 + so**2)
            # df/ds by hand
            num = 2 * s * (s**2 + so**2) - s**2 * (2 * s - 2 * so)
            return num / (s**2 + so**2) ** 2
        assert np.allclose(F, np.abs([analytic(s) for s in sw]), rtol=1e-6)
        print(f"ok test_fractional_flow_derivative_quadratic_relperm "
              f"(F at sw=0.5 is {F[1]:.3f})")

    def test_cfl_is_dt_v_dfds_over_pv_in_1d():
        """In 1D with a single through-flux, every interior cell's CFL is
        dt * |v_T| * |df/ds| / pv - the textbook Buckley-Leverett number."""
        g, rock, fluid = _model(nx=10)
        m = GenericBlackOilModel(g, rock, fluid, [])
        nc = g.cells.num
        st = State(pressure=np.linspace(210, 190, nc) * BARSA,
                   sW=np.full(nc, 0.5),
                   qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
        dt = 10.0 * DAY
        cfl = estimate_saturation_cfl(m, st, dt)
        # reconstruct the expected number from the same flux
        ad = m.init_state_ad(st)
        v = np.column_stack([np.asarray(value(x), float)
                             for x in m.get_props(ad, "PhaseFlux")])
        vt = v.sum(axis=1)
        F = fractional_flow_derivative(m, st["sW"])
        # interior cell 5: upstream faces carry |vt|*F
        assert np.all(cfl >= 0)
        assert cfl.max() > 0, "a flowing 1D column must have a positive CFL"
        # monotone pressure -> uniform sign flux -> interior cells all similar
        interior = cfl[2:-2]
        assert np.allclose(interior, interior[0], rtol=1e-9), \
            "uniform 1D flux should give a uniform interior CFL"
        expect = dt * abs(vt[3]) * F[3] / m.operators.pv[3]
        assert np.isclose(interior[0], expect, rtol=1e-9), (interior[0], expect)
        print(f"ok test_cfl_is_dt_v_dfds_over_pv_in_1d (max CFL {cfl.max():.3e})")

    def test_three_phase_is_refused_clearly():
        g, rock, fluid = _model()
        m = GenericBlackOilModel(g, rock, fluid, [])
        m.fluid.phases = ["water", "oil", "gas"]     # pretend 3-phase
        try:
            fractional_flow_derivative(m, np.array([0.5]))
        except NotImplementedError:
            print("ok test_three_phase_is_refused_clearly")
            return
        raise AssertionError("three-phase CFL should be refused, not wrong")

    test_fractional_flow_derivative_quadratic_relperm()
    test_cfl_is_dt_v_dfds_over_pv_in_1d()
    test_three_phase_is_refused_clearly()
    print("\nAll tests passed!")
