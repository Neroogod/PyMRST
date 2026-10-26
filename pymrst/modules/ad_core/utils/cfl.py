"""CFL estimates (MRST ``estimateSaturationCFL``/``estimateCompositionCFL``).

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


def estimate_flow_cfl(model: Any, state, dt: float, forces=None):
    """Return saturation and composition CFL arrays for FIM/AIM/IMPES.

    The composition estimate is valid for arbitrary component counts and is
    the stability controller for three-phase black oil.  The specialized
    Buckley--Leverett saturation wave-speed estimate is additionally used for
    two-phase models.
    """
    ad_state = model.init_state_ad(state)
    phase_flux = np.column_stack([
        np.asarray(value(v), dtype=float)
        for v in model.get_props(ad_state, "PhaseFlux")
    ])
    component_flux = np.column_stack([
        np.asarray(value(v), dtype=float)
        for v in model.get_props(ad_state, "ComponentTotalFlux")
    ])
    component_mass = np.column_stack([
        np.asarray(value(v), dtype=float)
        for v in model.get_props(ad_state, "ComponentTotalMass")
    ])
    composition = estimate_composition_cfl_ad(
        model, state, dt, phase_flux=phase_flux,
        component_flux=component_flux, component_mass=component_mass,
    )
    # Both estimates now apply at any phase count: the saturation one takes the
    # largest eigenvalue of the fractional-flow Jacobian, which for two phases
    # is the familiar `|df_w/ds_w|`. Returning zeros for three phases - as this
    # did while that branch was unported - made the saturation controller
    # silently neutral on exactly the models where a front moves fastest.
    saturation = estimate_saturation_cfl_ad(model, state, dt, flux=phase_flux)
    return saturation, composition


def fractional_flow_derivative(model: Any, sw) -> np.ndarray:
    r"""`|df_w/ds_w|` per cell - the Buckley-Leverett wave speed.

    `f_w = lambda_w / (lambda_w + lambda_o)` with the phase mobilities from the
    fluid's relative permeabilities, differentiated with respect to water
    saturation by seeding it as the AD variable. This is MRST's
    `getFractionalFlowMagnitude` for two phases; `fractional_flow_magnitude`
    below is the general form, and this stays because it reaches the fluid's
    curves directly and so works on a model with no state-function graph.
    """
    if model.n_phases != 2:
        # A bare water saturation cannot determine a multi-phase wave speed:
        # with three phases the Jacobian's off-diagonal entries depend on the
        # gas saturation too, and `1 - s_w` splits between oil and gas in a
        # way one number does not fix. Refused rather than guessed - the
        # guess would return a plausible speed for a saturation state the
        # model is not in.
        if isinstance(sw, (list, tuple)) or np.ndim(value(sw)) == 2:
            return fractional_flow_magnitude(model, sw)
        raise NotImplementedError(
            f"fractional_flow_derivative takes a single water saturation and "
            f"this model has {model.n_phases} phases; one saturation does not "
            f"determine the others. Use fractional_flow_magnitude(model, "
            f"[s_w, s_o, s_g]) - or estimate_saturation_cfl_ad, which builds the "
            f"full set from the state.")
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


def fractional_flow_magnitude(model: Any, state) -> np.ndarray:
    r"""The wave speed for any phase count (MRST `getFractionalFlowMagnitude`).

    With `n` phases there are `n-1` independent saturations, so the fractional
    flow is a map from `R^(n-1)` to itself and its derivative is an
    `(n-1)x(n-1)` matrix per cell:

        J_ij = d(f_i)/d(s_j),   f_i = lambda_i / sum_k lambda_k

    The wave speed is the **largest absolute eigenvalue** of that matrix, not
    the largest entry. The distinction is the whole reason this is not a
    two-phase formula applied three times: the characteristic speeds of a
    three-phase system are the eigenvalues of the coupled Jacobian, and a
    single row of it describes no wave at all. Taking a row maximum instead
    under-estimates the fastest wave wherever the phases are strongly coupled -
    which is exactly where the front is - and an under-estimated CFL is a
    timestep chosen too long.

    For two phases the matrix is 1x1 and this reduces to `|df_w/ds_w|`, which
    is what makes it a generalisation rather than a second implementation.

    The last saturation is *not* an independent variable: it is
    `1 - sum(others)`, and seeding it as well would give a singular system
    describing a state the model cannot occupy.
    """
    saturations = _saturations(model, state)
    n_phase = len(saturations)
    n_cells = np.asarray(saturations[0]).size
    if n_phase < 2:
        return np.ones(n_cells)

    seeded = list(init_variables_adi(
        *[np.asarray(value(s), dtype=float).ravel()
          for s in saturations[:n_phase - 1]]))
    last = 1.0
    for s in seeded:
        last = last - s
    seeded.append(last)

    mobility = _mobilities(model, seeded)
    total = mobility[0]
    for m in mobility[1:]:
        total = total + m

    # J[:, i, j] = d(f_i)/d(s_j), one (n-1)x(n-1) matrix per cell.
    jacobian = np.zeros((n_cells, n_phase - 1, n_phase - 1))
    for i in range(n_phase - 1):
        frac = mobility[i]/total
        for j in range(n_phase - 1):
            jacobian[:, i, j] = np.asarray(frac.jac[j].diagonal(), dtype=float)

    if n_phase == 2:
        return np.abs(jacobian[:, 0, 0])
    return np.max(np.abs(np.linalg.eigvals(jacobian)), axis=1)


def _as_ad_state(model, state):
    """`model.init_state_ad(state)` where that is a usable single-argument
    call, otherwise the state unchanged.

    `hasattr(model, "init_state_ad")` is not the question. `PhysicalModel`
    declares `init_state_ad(state, variables, names, origin=None)` - three
    required arguments - so any model that inherits it answers the probe
    `True` and then raises `TypeError` on the one-argument call. Models that
    build their own AD state (`GenericBlackOilModelAD`) take a single argument
    and are the ones this is for.

    So the test is whether the method is the base's, not whether it exists.
    This is the third instance of that trap in the port; each time it appeared
    as a model joining the hierarchy and breaking an unrelated caller.
    """
    from pymrst.modules.ad_core.models.physical_model import PhysicalModel

    own = getattr(type(model), "init_state_ad", None)
    if own is None or own is PhysicalModel.init_state_ad:
        return state
    return model.init_state_ad(state)


def _state_saturations(model: Any, state) -> list:
    """Every phase saturation on a reservoir state, in the model's order."""
    if hasattr(state, "get") and state.get("s") is not None:
        s = state["s"]
        if isinstance(s, (list, tuple)):
            return [np.asarray(value(x), dtype=float) for x in s]
        arr = np.atleast_2d(np.asarray(value(s), dtype=float))
        return [arr[:, i] for i in range(arr.shape[1])]
    ad = model.init_state_ad(state)
    return [np.asarray(value(x), dtype=float)
            for x in model.get_props(ad, "Saturations")]


def _saturations(model: Any, state) -> list:
    """Every phase saturation at `state`, in the model's phase order."""
    if isinstance(state, (list, tuple)):
        return list(state)
    arr = np.asarray(value(state))
    if arr.ndim == 2:
        return [arr[:, i] for i in range(arr.shape[1])]
    if arr.ndim == 1 and getattr(model, "n_phases", 2) == 2:
        # A bare water saturation, which is how the two-phase caller spells it.
        return [arr, 1.0 - arr]
    ad = _as_ad_state(model, state)
    return [np.asarray(value(s), dtype=float)
            for s in model.get_props(ad, "Saturations")]


def _mobilities(model: Any, saturations: list) -> list:
    """Phase mobilities at AD-seeded saturations.

    Reaches the fluid's relative permeabilities and viscosities directly rather
    than the `Mobility` state function, because this is called on models that
    do not carry the property graph. Where the graph exists the two agree - the
    state function computes `kr/mu` from the same curves.
    """
    fluid = getattr(model, "raw_fluid", None) or model.fluid
    names = [str(n).upper() for n in model.get_phase_names()]
    n_phase = len(saturations)

    rule = getattr(model, "rel_perm_wog", None)
    three_phase = (n_phase == 3 and {"W", "O", "G"} <= set(names))
    if three_phase and rule is not None:
        # The three-phase oil curve depends on *both* the water and gas
        # saturations - it is interpolated between the two-phase curves and
        # has no form of its own. Using a single `kr_o(s_o)` here would give a
        # smooth, plausible wave speed computed from a relative permeability
        # the model never uses.
        w, o, g = (names.index("W"), names.index("O"), names.index("G"))
        kr = rule(saturations[w], saturations[o], saturations[g], fluid)
    elif three_phase and not hasattr(fluid, "kr_ow"):
        # No interpolation rule and no two-phase oil curves to build one from:
        # the fluid genuinely carries one curve per phase.
        kr = [_phase_kr(fluid, name, s) for name, s in zip(names, saturations)]
    elif three_phase:
        raise NotImplementedError(
            "a three-phase fluid with kr_ow/kr_og curves needs the model's "
            "rel_perm_wog interpolation to build the oil curve; this model "
            "does not provide one, and a single kr_o would be a different "
            "relative permeability from the one the model solves with")
    else:
        kr = [_phase_kr(fluid, name, s) for name, s in zip(names, saturations)]

    viscosity = _viscosities(model, fluid, n_phase)
    return [kr[i]/viscosity[i] for i in range(n_phase)]


def _phase_kr(fluid: Any, name: str, s):
    curve = getattr(fluid, f"kr_{name.lower()}", None) \
        or getattr(fluid, f"kr{name.upper()}", None)
    if curve is None:
        raise AttributeError(
            f"fluid has no relative-permeability curve for phase {name!r}; "
            f"the saturation CFL needs one per phase")
    return curve(s)


def _viscosities(model: Any, fluid: Any, n_phase: int) -> list:
    """Constant phase viscosities, or the fluid's at a reference pressure.

    The wave speed is a *saturation* derivative, so the viscosities are held
    fixed here - differentiating them would add a pressure wave to a
    saturation CFL, which is a different stability number.
    """
    p = np.full(1, float(getattr(model, "p_ref_scalar", 200e5)))
    mu = getattr(fluid, "mu", None)

    if mu is not None and not callable(mu):
        # A sequence: either constants (simple fluid) or one callable per
        # phase (deck fluid). A deck fluid's oil viscosity additionally takes
        # `rs` and a saturated flag, which is why the call is attempted with
        # a widening argument list rather than assumed - guessing the arity
        # from the phase name would break on the next fluid that differs.
        return [_scalar_viscosity(mu[i], p) for i in range(n_phase)]

    if callable(mu):
        return [_as_scalar(x) for x in mu(p)][:n_phase]

    names = [str(n).lower() for n in model.get_phase_names()]
    return [_scalar_viscosity(getattr(fluid, f"mu_{name}", None), p)
            for name in names[:n_phase]]


def _scalar_viscosity(curve, p) -> float:
    """One phase's viscosity at `p`, whatever arity its curve has."""
    if curve is None:
        return 1.0
    if not callable(curve):
        return float(curve)
    zeros = np.zeros_like(p)
    flag = np.zeros(p.size, dtype=bool)
    for args in ((p,), (p, zeros), (p, zeros, flag)):
        try:
            return _as_scalar(curve(*args))
        except TypeError:
            continue
    raise TypeError(
        f"could not evaluate viscosity curve {curve!r} at a single pressure; "
        f"tried (p), (p, rs) and (p, rs, flag)")


def _as_scalar(x) -> float:
    return float(np.ravel(value(x))[0])


def model_pressure(model, fallback):                      # pragma: no cover
    """A pressure to evaluate a deck fluid's viscosity at - unused by the
    two-phase simple-fluid path, present so the branch above reads clearly."""
    return np.full(np.asarray(value(fallback)).shape,
                   getattr(model, "p_ref_scalar", 200e5))


def estimate_saturation_cfl_ad(model: Any, state, dt: float,
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

    if model.n_phases == 2:
        # The two-phase path takes the water saturation alone and reaches the
        # fluid's curves directly, so it works on models with no property
        # graph. Unchanged, and validated against MATLAB.
        sw = state["sW"] if "sW" in state else model.get_props(
            model.init_state_ad(state), "Saturations")[0]
        frac = fractional_flow_derivative(model, sw)
    else:
        # Three or more phases need *every* saturation: the wave speed is an
        # eigenvalue of the coupled Jacobian, and one saturation determines
        # none of its off-diagonal entries.
        frac = fractional_flow_magnitude(
            model, _state_saturations(model, state))

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


def estimate_composition_cfl_ad(
    model: Any,
    state,
    dt: float,
    *,
    use_inflow: bool = False,
    component_flux: Optional[np.ndarray] = None,
    component_mass: Optional[np.ndarray] = None,
    phase_flux: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return MRST's component-throughput CFL for every cell and component.

    Optional arrays expose the numerical kernel for testing and specialized
    callers. Their shapes are respectively ``(nface, ncomp)``,
    ``(ncell, ncomp)``, and ``(nface, nphase)``. If omitted, the corresponding
    state functions are evaluated through ``model.get_props``.
    """
    n = np.asarray(model.operators.N, dtype=np.int64)
    left, right = n[:, 0], n[:, 1]

    if phase_flux is None:
        phase_flux = state["flux"]
    v = np.asarray(value(phase_flux), dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    v_t = v.sum(axis=1)
    flag = v_t > 0
    xflow = ~(np.all(v >= 0, axis=1) | np.all(v <= 0, axis=1))

    if component_flux is None or component_mass is None:
        ad_state = _as_ad_state(model, state)
        if component_flux is None:
            component_flux = model.get_props(ad_state, "ComponentTotalFlux")
        if component_mass is None:
            component_mass = model.get_props(ad_state, "ComponentTotalMass")
    q = np.asarray(value(component_flux), dtype=float)
    mass = np.asarray(value(component_mass), dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    if mass.ndim == 1:
        mass = mass[:, None]

    total_mass = mass.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = mass / total_mass
    safe_mass = mass.copy()
    safe_mass[fractions < 1e-7] = 1.0

    mass_face = np.where(flag[:, None], safe_mass[left], safe_mass[right])
    if np.any(xflow):
        mass_face[xflow] = np.maximum(
            np.abs(safe_mass[left[xflow]]),
            np.abs(safe_mass[right[xflow]]),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_face = np.abs(q) / mass_face

    nc, ncomp = mass.shape
    cfl = np.zeros((nc, ncomp), dtype=float)
    f = ~flag if use_inflow else flag
    for component in range(ncomp):
        np.add.at(
            cfl[:, component],
            left,
            rate_face[:, component] * (f | xflow),
        )
        np.add.at(
            cfl[:, component],
            right,
            rate_face[:, component] * (~f | xflow),
        )
    cfl *= float(dt)
    cfl[~np.isfinite(cfl)] = 0.0
    return cfl


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.

#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
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
        m = GenericBlackOilModelAD(g, rock, fluid, [])
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
        m = GenericBlackOilModelAD(g, rock, fluid, [])
        nc = g.cells.num
        st = State(pressure=np.linspace(210, 190, nc) * BARSA,
                   sW=np.full(nc, 0.5),
                   qWs=np.zeros(0), qOs=np.zeros(0), bhp=np.zeros(0))
        dt = 10.0 * DAY
        cfl = estimate_saturation_cfl_ad(m, st, dt)
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

    def test_three_phase_needs_every_saturation():
        """Three-phase wave speeds are ported, but they need the whole
        saturation set.

        This test used to assert that three phases were *refused outright*.
        They are not any more - `fractional_flow_magnitude` takes the largest
        eigenvalue of the fractional-flow Jacobian. What is still refused is
        being handed one saturation and asked to infer the rest, because the
        inference has no unique answer and the wrong one looks plausible.
        """
        g, rock, fluid = _model()
        m = GenericBlackOilModelAD(g, rock, fluid, [])
        m.fluid.phases = ["water", "oil", "gas"]     # pretend 3-phase
        try:
            fractional_flow_derivative(m, np.array([0.5]))
        except NotImplementedError as refused:
            assert "does not determine" in str(refused), str(refused)
        else:
            raise AssertionError("one saturation cannot fix a 3-phase state")

        # The *working* three-phase computation is not exercised here: this
        # model only pretends to have three phases, and its fluid carries no
        # gas relative permeability, so there is genuinely nothing to compute
        # from. tests/ad_core/utils/test_three_phase_cfl.py covers it against
        # a fluid that has all three curves.
        print("ok test_three_phase_needs_every_saturation")

    test_fractional_flow_derivative_quadratic_relperm()
    test_cfl_is_dt_v_dfds_over_pv_in_1d()
    test_three_phase_needs_every_saturation()
    print("\nAll tests passed!")
