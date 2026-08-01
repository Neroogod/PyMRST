"""Partial-volume weights (MRST ``getPartialVolumes.m`` compatibility)."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.compositional.backend import backend_flow_props
from pymrst.modules.compositional.eos_ad import phase_compressibility_ad
from pymrst.modules.compositional.properties import molar_density
from pymrst.utils.adi import combine_equations, init_variables_adi, value
from pymrst.utils.units import BARSA


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _option(kwargs: dict[str, Any], camel: str, snake: str, default):
    return kwargs.pop(camel, kwargs.pop(snake, default))


def _state_field(state: Any, *names: str):
    for name in names:
        value = _get(state, name, None)
        if value is not None:
            return value
    raise KeyError(f"state is missing all of {names!r}")


def _molar_mass(model: Any, ncomp: int) -> np.ndarray:
    eos = _get(model, "EOSModel", model)
    mixture = _get(eos, "CompositionalMixture", None)
    if mixture is None:
        mixture = _get(eos, "mixture", None)
    if mixture is None:
        mixture = _get(eos, "fluid", None)
    mw = _get(mixture, "molarMass", None)
    if mw is None:
        return np.ones(ncomp, dtype=float)
    mw = np.asarray(mw, dtype=float).reshape(-1)
    if mw.size < ncomp:
        mw = np.pad(mw, (0, ncomp - mw.size), constant_values=1.0)
    return mw[:ncomp]


def _num_cells(state: Any) -> int:
    return np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1).size


def _num_components(model: Any, state: Any) -> int:
    comps = _get(state, "components", None)
    if comps is not None:
        ncomp = np.asarray(comps).shape[1]
    else:
        eos = _get(model, "EOSModel", model)
        if hasattr(eos, "getNumberOfComponents"):
            ncomp = int(eos.getNumberOfComponents())
        else:
            mix = _get(eos, "CompositionalMixture", _get(eos, "mixture", None))
            ncomp = int(_get(mix, "n", len(_get(mix, "names", ()))) or 1)
    return ncomp + int(bool(_get(model, "water", False)))


def _numeric_accumulation_weights(
    model: Any,
    state: Any,
    acc,
    subs,
    *,
    reorder=None,
    well_var_indices=None,
) -> np.ndarray:
    nc = int(np.count_nonzero(subs))
    ncomp = _num_components(model, state)
    if acc is None:
        return np.ones((nc, ncomp), dtype=float)
    if hasattr(model, "partial_volume_weights") and reorder is None and well_var_indices is None:
        try:
            weights = model.partial_volume_weights(acc)
            return np.column_stack([np.asarray(w, dtype=float).reshape(-1)[subs] for w in weights])
        except Exception:
            if any(hasattr(a, "jac") for a in (acc if isinstance(acc, (list, tuple)) else [acc])):
                raise
    terms = list(acc) if isinstance(acc, (list, tuple)) else [acc]
    if not terms or not any(hasattr(term, "jac") for term in terms):
        return np.ones((nc, ncomp), dtype=float)
    ncell = np.asarray(getattr(terms[0], "val", terms[0])).size
    J = combine_equations(*terms).jac[0].tocsc()
    if reorder is not None:
        order = np.asarray(reorder, dtype=np.int64).reshape(-1)
        if order.size != J.shape[1] or set(order.tolist()) != set(range(J.shape[1])):
            raise ValueError("reorder must be a zero-based permutation of all Jacobian columns")
        J = J[:, order]
    if well_var_indices is not None:
        remove = np.asarray(well_var_indices, dtype=np.int64).reshape(-1)
        if remove.size and (remove.min() < 0 or remove.max() >= J.shape[1]):
            raise IndexError("wellVarIndices contains a Jacobian column outside the reordered system")
        keep = np.ones(J.shape[1], dtype=bool)
        keep[remove] = False
        J = J[:, keep]
    ndof = ncell * ncomp
    if J.shape[0] < ndof or J.shape[1] < ndof:
        raise ValueError(
            f"accumulation Jacobian must contain a {ndof}-DOF cell block after filtering"
        )
    B = J[:ndof, :ndof]
    C = J[:ndof, ndof:]
    D = J[ndof:, :ndof]
    E = J[ndof:, ndof:]
    if E.shape == (0, 0):
        A = B
    else:
        if E.shape[0] != E.shape[1]:
            raise ValueError("non-cell Jacobian block must be square after well-variable filtering")
        A = B - C @ spla.spsolve(E, D.toarray())
        A = sp.csc_array(A)
    rhs = np.zeros(ndof, dtype=float)
    rhs[:ncell] = 1.0 / BARSA
    weights = np.asarray(spla.spsolve(A.T, rhs), dtype=float).reshape(ncomp, ncell).T
    return weights[subs]


def _mass_densities(mix: Any, eos_model: Any, p: np.ndarray, T: np.ndarray, fp):
    rho_l_molar, rho_v_molar = _molar_densities(mix, eos_model, p, T, fp)
    rho_l = rho_l_molar * (np.asarray(fp.x, dtype=float) @ np.asarray(mix.molarMass, dtype=float))
    rho_v = rho_v_molar * (np.asarray(fp.y, dtype=float) @ np.asarray(mix.molarMass, dtype=float))
    return np.asarray(rho_l, dtype=float).reshape(-1), np.asarray(rho_v, dtype=float).reshape(-1)


def _analytical_density_from_eos(model: Any, state: Any, subs) -> np.ndarray:
    mix, eos_model = _mixture_and_backend(model)
    rows = np.flatnonzero(subs)
    p = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1)[rows]
    T = _temperature(model, state, rows)
    z = _normalize_rows(np.asarray(_state_field(state, "components", "z"), dtype=float)[rows])
    fp = backend_flow_props(mix, eos_model, p, T, z)
    rho_l, rho_v = _mass_densities(mix, eos_model, p, T, fp)
    L = _get(state, "L", None)
    if L is None:
        liquid = np.asarray(fp.L, dtype=float).reshape(-1) >= 0.5
    else:
        liquid = np.asarray(L, dtype=float).reshape(-1)[rows] >= 0.5
    return np.where(liquid, rho_l, rho_v)


def _water_partial_volume(model: Any, state: Any, subs, units: str) -> np.ndarray:
    """Return water-first volume weights from state density or model fallback."""
    rows = np.flatnonzero(subs)
    rho = _get(state, "rho", None)
    if rho is not None:
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 2 and rho.shape[1] >= 1:
            density = rho[rows, 0]
        else:
            density = np.full(rows.size, float(_get(model, "water_rho_s", 1000.0)))
    else:
        density = np.full(rows.size, float(_get(model, "water_rho_s", 1000.0)))
    volume = 1.0 / np.maximum(density, 1.0e-300)
    if units.lower() == "moles":
        volume = volume * float(_get(model, "water_molar_mass", 0.01801528))
    elif units.lower() != "mass":
        raise ValueError(f"unknown units {units!r}")
    return volume[:, None]


def _analytical_single_phase(model: Any, state: Any, subs, units: str) -> np.ndarray:
    rho = _get(state, "rho", None)
    if rho is None:
        values, _ = _analytical_single_phase_ad(model, state, subs, units)
        return values
    else:
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 1:
            phase_rho = rho[subs]
        else:
            L = np.asarray(_get(state, "L", np.ones(rho.shape[0])), dtype=float).reshape(-1)
            liquid = L > 0.5
            phase_idx = np.where(liquid, int(bool(_get(model, "water", False))), int(bool(_get(model, "water", False))) + 1)
            rows = np.flatnonzero(subs)
            phase_rho = np.array([rho[row, phase_idx[row]] for row in rows], dtype=float)
    components = _get(state, "components", _get(state, "z", None))
    hydrocarbon_components = (
        np.asarray(components).shape[1]
        if components is not None
        else _num_components(model, state) - int(bool(_get(model, "water", False)))
    )
    V = np.repeat(
        (1.0 / np.maximum(phase_rho, 1.0e-300))[:, None],
        hydrocarbon_components,
        axis=1,
    )
    if units.lower() == "mass":
        V = V / _molar_mass(model, hydrocarbon_components)[None, :]
    elif units.lower() != "moles":
        raise ValueError(f"unknown units {units!r}")
    if bool(_get(model, "water", False)):
        V = np.column_stack([_water_partial_volume(model, state, subs, units), V])
    return V


def _analytical_single_phase_ad(
    model: Any, state: Any, subs: np.ndarray, units: str
) -> tuple[np.ndarray, np.ndarray]:
    """MRST single-phase partial volumes with native cubic-EOS AD.

    Pressure and each overall composition are independent AD blocks.  The
    compressibility root receives its implicit cubic derivative through
    :func:`phase_compressibility_ad`, after which the density Jacobian gives
    MRST's ``1/rho - drho/dz_i/rho**2`` partial molar-volume expression.
    """
    mix, eos_backend = _mixture_and_backend(model)
    rows = np.flatnonzero(subs)
    pressure = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1)[rows]
    temperature = _temperature(model, state, rows)
    composition = _normalize_rows(
        np.asarray(_state_field(state, "components", "z"), dtype=float)[rows]
    )
    primary = init_variables_adi(pressure, *[composition[:, i] for i in range(composition.shape[1])])
    p_ad, z_ad = primary[0], list(primary[1:])

    fp = backend_flow_props(mix, eos_backend, pressure, temperature, composition)
    z_l_ad, z_v_ad = phase_compressibility_ad(
        mix, p_ad, temperature, z_ad, z_ad,
        np.asarray(fp.Z_L, dtype=float).reshape(-1),
        np.asarray(fp.Z_V, dtype=float).reshape(-1),
    )
    liquid = np.asarray(_get(state, "L", fp.L), dtype=float).reshape(-1)
    if liquid.size != rows.size:
        liquid = liquid[rows]
    liquid = liquid >= 0.5

    volume_shift = None
    eos = _get(model, "EOSModel", None)
    prop = _get(eos, "PropertyModel", _get(eos, "property_model", None))
    if prop is not None:
        volume_shift = _get(prop, "volumeShift", _get(prop, "volume_shift", None))
    rho_l = molar_density(mix, p_ad, temperature, z_ad, z_l_ad, volume_shift)
    rho_v = molar_density(mix, p_ad, temperature, z_ad, z_v_ad, volume_shift)
    rho = rho_l.copy()
    rho[~liquid] = rho_v[~liquid]

    rho_value = np.asarray(value(rho), dtype=float).reshape(-1)
    inv_rho = 1.0 / np.maximum(rho_value, 1.0e-300)
    values = np.empty((rows.size, composition.shape[1]), dtype=float)
    derivatives = np.empty_like(values)
    pressure_jac = rho.jac[0].diagonal()
    for component in range(composition.shape[1]):
        drho_dzi = rho.jac[component + 1].diagonal()
        values[:, component] = inv_rho - drho_dzi * inv_rho**2
        # This follows MRST getSinglePhaseVolumes: drho/dz is extracted as a
        # value before differentiating v, so no mixed second derivative enters.
        derivatives[:, component] = (
            -pressure_jac * inv_rho**2
            + 2.0 * drho_dzi * pressure_jac * inv_rho**3
        )

    if units.lower() == "mass":
        values = values / _molar_mass(model, composition.shape[1])[None, :]
        derivatives = derivatives / _molar_mass(model, composition.shape[1])[None, :]
    elif units.lower() != "moles":
        raise ValueError(f"unknown units {units!r}")
    if bool(_get(model, "water", False)):
        water = _water_partial_volume(model, state, subs, units)
        values = np.column_stack([water, values])
        derivatives = np.column_stack([np.zeros_like(water), derivatives])
    return values, derivatives


def _mixture_and_backend(model: Any):
    mix = _get(model, "mix", None)
    if mix is None:
        mix = _get(model, "mixture", None)
    eos_model = _get(model, "eos_model", None)
    if mix is None:
        eos = _get(model, "EOSModel", model)
        mix = _get(eos, "CompositionalMixture", None)
        if mix is None:
            mix = _get(eos, "mixture", None)
        if mix is None:
            mix = _get(eos, "fluid", None)
        if eos_model is None and eos is not model and hasattr(eos, "flash"):
            eos_model = eos
    if mix is None:
        raise NotImplementedError("EOS getPartialVolumes requires model.mix/mixture or model.EOSModel.fluid")
    return mix, eos_model


def _temperature(model: Any, state: Any, rows: np.ndarray) -> np.ndarray:
    T = _get(state, "T", _get(state, "temperature", _get(model, "T", _get(model, "temperature", None))))
    if T is None:
        raise KeyError("EOS getPartialVolumes requires state/model temperature")
    T = np.asarray(T, dtype=float).reshape(-1)
    if T.size == 1:
        return np.full(rows.size, float(T[0]), dtype=float)
    return T[rows]


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=float), 1.0e-300)
    return a / a.sum(axis=1, keepdims=True)


def _molar_densities(mix: Any, eos_model: Any, p: np.ndarray, T: np.ndarray, fp):
    if eos_model is not None and hasattr(eos_model, "property_model"):
        prop = eos_model.property_model
        rho_l = prop.compute_molar_density(p, fp.x, is_liquid=True)
        rho_v = prop.compute_molar_density(p, fp.y, is_liquid=False)
    else:
        rho_l = molar_density(mix, p, T, fp.x, fp.Z_L)
        rho_v = molar_density(mix, p, T, fp.y, fp.Z_V)
    return np.asarray(rho_l, dtype=float).reshape(-1), np.asarray(rho_v, dtype=float).reshape(-1)


def _total_molar_volume(mix: Any, eos_model: Any, p: np.ndarray, T: np.ndarray, z: np.ndarray, total_moles: np.ndarray) -> np.ndarray:
    fp = backend_flow_props(mix, eos_model, p, T, z)
    rho_l, rho_v = _molar_densities(mix, eos_model, p, T, fp)
    L = np.asarray(fp.L, dtype=float).reshape(-1)
    specific = L / np.maximum(rho_l, 1.0e-300) + (1.0 - L) / np.maximum(rho_v, 1.0e-300)
    return total_moles * specific


def _eos_partial_volumes(model: Any, state: Any, subs: np.ndarray, units: str) -> np.ndarray:
    mix, eos_model = _mixture_and_backend(model)
    rows = np.flatnonzero(subs)
    p = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1)[rows]
    T = _temperature(model, state, rows)
    z = _normalize_rows(np.asarray(_state_field(state, "components", "z"), dtype=float)[rows])
    ncomp = z.shape[1]
    # The total volume is homogeneous in component mole amounts.  One mole per
    # cell is enough to recover the partial molar derivatives dV/dN_i.
    N = z.copy()
    total = N.sum(axis=1)
    base = _total_molar_volume(mix, eos_model, p, T, _normalize_rows(N), total)
    eps = np.maximum(1.0e-8, 1.0e-6 * np.maximum(N, 1.0).max(axis=1))
    V = np.empty((rows.size, ncomp), dtype=float)
    for i in range(ncomp):
        pert = N.copy()
        pert[:, i] += eps
        total_pert = pert.sum(axis=1)
        vp = _total_molar_volume(mix, eos_model, p, T, _normalize_rows(pert), total_pert)
        V[:, i] = (vp - base) / eps
    if units.lower() == "mass":
        V = V / _molar_mass(model, ncomp)[None, :]
    elif units.lower() != "moles":
        raise ValueError(f"unknown units {units!r}")
    if bool(_get(model, "water", False)):
        water_col = _water_partial_volume(model, state, subs, units)
        V = np.column_stack([water_col, V])
    return V


def _pressure_derivative_eos(model: Any, state: Any, subs: np.ndarray, units: str, pressure_epsilon: float) -> np.ndarray:
    V = _eos_partial_volumes(model, state, subs, units)
    class Perturbed:
        pass
    pert = Perturbed()
    if isinstance(state, dict):
        pert.__dict__.update(state)
    else:
        pert.__dict__.update(getattr(state, "__dict__", {}))
    p = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1).copy()
    p -= float(pressure_epsilon)
    pert.pressure = p
    V_lower = _eos_partial_volumes(model, pert, subs, units)
    return (V - V_lower) / float(pressure_epsilon)


def _state_with_pressure(state: Any, pressure: np.ndarray):
    """Return a shallow state copy with a replaced pressure vector."""
    if isinstance(state, dict):
        perturbed = dict(state)
        perturbed["pressure"] = pressure
        return perturbed
    class Perturbed:
        pass
    perturbed = Perturbed()
    perturbed.__dict__.update(getattr(state, "__dict__", {}))
    perturbed.pressure = pressure
    return perturbed


def _pressure_derivative_analytical(
    model: Any,
    state: Any,
    subs: np.ndarray,
    units: str,
    pressure_epsilon: float,
) -> np.ndarray:
    """Differentiate single-phase weights using native EOS AD when available."""
    current = _analytical_single_phase(model, state, subs, units)
    if _get(state, "rho", None) is not None:
        # A stored density has no associated pressure law. Do not fabricate one.
        return np.zeros_like(current)
    del pressure_epsilon
    _, derivative = _analytical_single_phase_ad(model, state, subs, units)
    return derivative


def _partial_volume_group(
    model: Any,
    state: Any,
    acc,
    subs: np.ndarray,
    strategy: str,
    diff: str,
    units: str,
    pressure_epsilon: float,
    reorder=None,
    well_var_indices=None,
    iteration: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one uniform phase/strategy subset like MRST's wrapper."""
    strategy = str(strategy).lower()
    diff = str(diff).lower()
    nc = _num_cells(state)
    if strategy == "numerical":
        V = _numeric_accumulation_weights(
            model,
            state,
            acc,
            subs,
            reorder=reorder,
            well_var_indices=well_var_indices,
        )
    elif strategy == "analytical":
        V = _analytical_single_phase(model, state, subs, units)
    elif strategy == "eos":
        V = _eos_partial_volumes(model, state, subs, units)
    else:
        raise ValueError(f"unsupported partial-volume strategy {strategy!r}")

    if diff == "none":
        dVdp = np.zeros_like(V)
    elif diff == "eos" and strategy == "eos":
        dVdp = _pressure_derivative_eos(model, state, subs, units, pressure_epsilon)
    elif diff == "numerical":
        w_prev = _get(state, "w", None)
        w_p = _get(state, "w_p", None)
        pressure = np.asarray(_state_field(state, "pressure"), dtype=float).reshape(-1)[subs]
        if w_prev is not None and w_p is not None and float(iteration) > 1.0:
            prev = np.asarray(w_prev, dtype=float)
            if prev.shape[0] == nc:
                prev = prev[subs]
            prev_sum = np.sum(np.abs(prev), axis=1, keepdims=True)
            curr_sum = np.sum(np.abs(V), axis=1, keepdims=True)
            prev_norm = np.divide(prev, prev_sum, out=np.zeros_like(prev), where=prev_sum != 0.0)
            curr_norm = np.divide(V, curr_sum, out=np.zeros_like(V), where=curr_sum != 0.0)
            previous_pressure = np.asarray(w_p, dtype=float).reshape(-1)
            if previous_pressure.size == nc:
                previous_pressure = previous_pressure[subs]
            dp = pressure - previous_pressure
            dVdp = np.divide(
                curr_norm - prev_norm,
                dp[:, None],
                out=np.zeros_like(V),
                where=dp[:, None] != 0.0,
            )
        else:
            dVdp = np.zeros_like(V)
    elif diff == "analytical" and strategy == "analytical":
        dVdp = _pressure_derivative_analytical(
            model, state, subs, units, pressure_epsilon
        )
    else:
        raise ValueError(
            f"partial-volume differentiation {diff!r} is invalid for strategy {strategy!r}"
        )
    return V, dVdp


def getPartialVolumes(model, state, acc=None, **kwargs):
    """Return ``(V, dVdp)`` partial-volume weights.

    Supported MRST strategies in this PyMRST port:
    - ``strategy='numerical'``: use ``model.partial_volume_weights(acc)`` when
      AD accumulation terms are supplied; numeric accumulation falls back to
      unit weights, matching MRST's numeric fallback.
    - ``strategy='analytical'``: single-phase density estimate from ``state.rho``.
    - ``strategy='eos'``: EOS-flash finite-difference partial molar volumes for
      two-phase cells, matching MRST's physical branch without requiring the
      deeper diagonal-AD state-function internals.
    """
    units = _option(kwargs, "units", "units", "mass")
    pressure_epsilon = _option(kwargs, "pressureEpsilon", "pressure_epsilon", 0.1 * 6894.757293168)
    single_strategy = _option(kwargs, "singlePhaseStrategy", "single_phase_strategy", "numerical")
    two_strategy = _option(kwargs, "twoPhaseStrategy", "two_phase_strategy", single_strategy)
    single_diff = _option(kwargs, "singlePhaseDifferentiation", "single_phase_differentiation", "numerical")
    two_diff = _option(kwargs, "twoPhaseDifferentiation", "two_phase_differentiation", single_diff)
    subset = _option(kwargs, "subset", "subset", None)
    reorder = _option(kwargs, "reorder", "reorder", None)
    well_var_indices = _option(kwargs, "wellVarIndices", "well_var_indices", None)
    iteration = _option(kwargs, "iteration", "iteration", np.nan)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected option(s): {unknown}")

    nc = _num_cells(state)
    subs = np.ones(nc, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    if subs.size != nc:
        raise ValueError("subset must have one entry per cell")

    L = _get(state, "L", None)
    if L is None:
        two_phase = np.zeros(nc, dtype=bool)
        single_phase = np.ones(nc, dtype=bool)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)
        two_phase = (L > 1.0e-8) & (L < 1.0 - 1.0e-8)
        single_phase = ~two_phase

    selected = np.flatnonzero(subs)
    output_position = np.full(nc, -1, dtype=int)
    output_position[selected] = np.arange(selected.size)
    groups = (
        (single_phase & subs, single_strategy, single_diff),
        (two_phase & subs, two_strategy, two_diff),
    )
    V = dVdp = None
    for group, strategy, diff in groups:
        if not np.any(group):
            continue
        values, derivatives = _partial_volume_group(
            model,
            state,
            acc,
            group,
            strategy,
            diff,
            units,
            pressure_epsilon,
            reorder=reorder,
            well_var_indices=well_var_indices,
            iteration=iteration,
        )
        if V is None:
            V = np.empty((selected.size, values.shape[1]), dtype=float)
            dVdp = np.empty_like(V)
        positions = output_position[np.flatnonzero(group)]
        V[positions] = values
        dVdp[positions] = derivatives
    if V is None:
        ncomp = _num_components(model, state)
        V = np.empty((0, ncomp), dtype=float)
        dVdp = np.empty_like(V)
    return V, dVdp


def get_partial_volumes(model, state, acc=None, **kwargs):
    """Pythonic alias for :func:`getPartialVolumes`."""
    return getPartialVolumes(model, state, acc=acc, **kwargs)


__all__ = ["getPartialVolumes", "get_partial_volumes"]
