"""EOS backend dispatch for compositional flow models.

MRST lets compositional flow models use different EOS/property backends.  The
early PyMRST port was hard-wired to the Peng--Robinson helpers.  This module is
the native bridge that lets the same flow equations consume either the original
PR backend or the black-oil-derived K-value backend.
"""

from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np

from pymrst.modules.compositional.flow_props import (
    FlowProps,
    mass_fraction,
    phase_saturations,
    two_phase_flow_props,
)
from pymrst.utils.adi import ADI, _diag_mul, value


def has_backend(eos_model: Any) -> bool:
    """True when an alternate EOS/property backend is supplied."""
    return eos_model is not None


def backend_flash(mix, eos_model, p, T, z):
    """Flash through the active backend."""
    if eos_model is None:
        from pymrst.modules.compositional.eos import flash

        return flash(mix, p, T, z)
    return eos_model.flash(p, T, z)


def backend_flow_props(mix, eos_model, p, T, z, flash_result=None) -> FlowProps:
    """Value-level ``(p, T, z) -> FlowProps`` for PR or K-value EOS models."""
    if eos_model is None:
        return two_phase_flow_props(mix, p, T, z, flash_result=flash_result)

    p = np.asarray(p, dtype=float).ravel()
    T = np.broadcast_to(np.asarray(T, dtype=float).ravel(), p.shape)
    z = _normalize_rows(np.asarray(z, dtype=float))
    fr = eos_model.flash(p, T, z) if flash_result is None else flash_result
    liquid_fraction = 1.0 - fr.V
    x = _normalize_rows(fr.x)
    y = _normalize_rows(fr.y)
    x_cols = [x[:, i] for i in range(mix.n)]
    y_cols = [y[:, i] for i in range(mix.n)]
    x_m = mass_fraction(mix, x_cols)
    y_m = mass_fraction(mix, y_cols)
    s_l, s_v = phase_saturations(mix, p, T, liquid_fraction, x, y, fr.Z_L, fr.Z_V)
    prop = eos_model.property_model
    rho_l = prop.compute_density(p, x, is_liquid=True)
    rho_v = prop.compute_density(p, y, is_liquid=False)
    mu_l = prop.compute_viscosity(p, x, is_liquid=True)
    mu_v = prop.compute_viscosity(p, y, is_liquid=False)
    return FlowProps(
        xM=x_m,
        yM=y_m,
        sL=s_l,
        sV=s_v,
        rhoL=rho_l,
        rhoV=rho_v,
        muL=mu_l,
        muV=mu_v,
        x=x,
        y=y,
        L=liquid_fraction,
        Z_L=np.asarray(fr.Z_L, dtype=float),
        Z_V=np.asarray(fr.Z_V, dtype=float),
    )


def backend_flow_props_ad(mix, eos_model, p: ADI, T, z_list: List[ADI]):
    """ADI flow properties for the active backend.

    The PR backend keeps its analytic AD path.  The K-value backend is tabular,
    so PyMRST differentiates it locally by finite differences and composes those
    sensitivities with the ADI primary variables.
    """
    if eos_model is None:
        from pymrst.modules.compositional.eos_ad import (
            flash_ad,
            phase_compressibility_ad,
        )
        from pymrst.modules.compositional.properties import mass_density, viscosity_lbc

        x, y, L, zl_val, zv_val = flash_ad(mix, p, T, z_list)
        z_l, z_v = phase_compressibility_ad(mix, p, T, x, y, zl_val, zv_val)
        x_m = mass_fraction(mix, x)
        y_m = mass_fraction(mix, y)
        s_l, s_v = phase_saturations(mix, p, T, L, x, y, z_l, z_v)
        rho_l = mass_density(mix, p, T, x, z_l)
        rho_v = mass_density(mix, p, T, y, z_v)
        mu_l = viscosity_lbc(mix, p, T, x, z_l)
        mu_v = viscosity_lbc(mix, p, T, y, z_v)
        return x_m, y_m, s_l, s_v, rho_l, rho_v, mu_l, mu_v

    p_val = np.asarray(value(p), dtype=float)
    z_val = _normalize_rows(np.column_stack([value(zi) for zi in z_list]))
    base = backend_flow_props(mix, eos_model, p_val, T, z_val)
    primaries = [p, *z_list]
    derivs = _flow_prop_derivatives(mix, eos_model, p_val, T, z_val, base)

    x_m = [
        _adi_from_local_derivatives(base.xM[i], derivs[f"xM{i}"], primaries)
        for i in range(mix.n)
    ]
    y_m = [
        _adi_from_local_derivatives(base.yM[i], derivs[f"yM{i}"], primaries)
        for i in range(mix.n)
    ]
    s_l = _adi_from_local_derivatives(base.sL, derivs["sL"], primaries)
    s_v = _adi_from_local_derivatives(base.sV, derivs["sV"], primaries)
    rho_l = _adi_from_local_derivatives(base.rhoL, derivs["rhoL"], primaries)
    rho_v = _adi_from_local_derivatives(base.rhoV, derivs["rhoV"], primaries)
    mu_l = _adi_from_local_derivatives(base.muL, derivs["muL"], primaries)
    mu_v = _adi_from_local_derivatives(base.muV, derivs["muV"], primaries)
    return x_m, y_m, s_l, s_v, rho_l, rho_v, mu_l, mu_v


def backend_natural_props(mix, eos_model, p, T, x, y):
    """Value-level natural-variable phase properties."""
    if eos_model is None:
        from pymrst.modules.compositional.eos import fugacity_coefficients
        from pymrst.modules.compositional.properties import (
            mass_density,
            molar_density,
            viscosity_lbc,
        )

        _, z_l = fugacity_coefficients(mix, p, T, x, vapour=False)
        _, z_v = fugacity_coefficients(mix, p, T, y, vapour=True)
        return {
            "Z_L": z_l,
            "Z_V": z_v,
            "rhoL": mass_density(mix, p, T, x, z_l),
            "rhoV": mass_density(mix, p, T, y, z_v),
            "nL": molar_density(mix, p, T, x, z_l),
            "nV": molar_density(mix, p, T, y, z_v),
            "muL": viscosity_lbc(mix, p, T, x, z_l),
            "muV": viscosity_lbc(mix, p, T, y, z_v),
        }

    prop = eos_model.property_model
    return {
        "Z_L": _z_factor_from_density(prop, p, T, x, is_liquid=True),
        "Z_V": _z_factor_from_density(prop, p, T, y, is_liquid=False),
        "rhoL": prop.compute_density(p, x, is_liquid=True),
        "rhoV": prop.compute_density(p, y, is_liquid=False),
        "nL": prop.compute_molar_density(p, x, is_liquid=True),
        "nV": prop.compute_molar_density(p, y, is_liquid=False),
        "muL": prop.compute_viscosity(p, x, is_liquid=True),
        "muV": prop.compute_viscosity(p, y, is_liquid=False),
    }


def backend_natural_props_ad(mix, eos_model, p: ADI, T, x: List[ADI], y: List[ADI]):
    """ADI natural-variable props for PR or K-value backends."""
    if eos_model is None:
        from pymrst.modules.compositional.eos import fugacity_coefficients
        from pymrst.modules.compositional.eos_ad import (
            _compressibility_and_fugacity,
            phase_compressibility_ad,
        )
        from pymrst.modules.compositional.properties import mass_density, viscosity_lbc

        p_val = np.asarray(value(p), dtype=float)
        x_val = np.column_stack([value(xi) for xi in x])
        y_val = np.column_stack([value(yi) for yi in y])
        _, z_l_val = fugacity_coefficients(mix, p_val, T, x_val, vapour=False)
        _, z_v_val = fugacity_coefficients(mix, p_val, T, y_val, vapour=True)
        z_l, z_v = phase_compressibility_ad(mix, p, T, x, y, z_l_val, z_v_val)
        _, _, f_l, f_v = _compressibility_and_fugacity(mix, p, T, x, y, z_l_val, z_v_val)
        x_m = mass_fraction(mix, x)
        y_m = mass_fraction(mix, y)
        rho_l = mass_density(mix, p, T, x, z_l)
        rho_v = mass_density(mix, p, T, y, z_v)
        mu_l = viscosity_lbc(mix, p, T, x, z_l)
        mu_v = viscosity_lbc(mix, p, T, y, z_v)
        return x_m, y_m, rho_l, rho_v, mu_l, mu_v, f_l, f_v

    p_val = np.asarray(value(p), dtype=float)
    x_val = _normalize_rows(np.column_stack([value(xi) for xi in x]))
    y_val = _normalize_rows(np.column_stack([value(yi) for yi in y]))
    base = _natural_output_values(mix, eos_model, p_val, T, x_val, y_val)
    primaries = [p, *x, *y]
    derivs = _natural_derivatives(mix, eos_model, p_val, T, x_val, y_val, base)
    x_m = [
        _adi_from_local_derivatives(base[f"xM{i}"], derivs[f"xM{i}"], primaries)
        for i in range(mix.n)
    ]
    y_m = [
        _adi_from_local_derivatives(base[f"yM{i}"], derivs[f"yM{i}"], primaries)
        for i in range(mix.n)
    ]
    rho_l = _adi_from_local_derivatives(base["rhoL"], derivs["rhoL"], primaries)
    rho_v = _adi_from_local_derivatives(base["rhoV"], derivs["rhoV"], primaries)
    mu_l = _adi_from_local_derivatives(base["muL"], derivs["muL"], primaries)
    mu_v = _adi_from_local_derivatives(base["muV"], derivs["muV"], primaries)
    f_l = [
        _adi_from_local_derivatives(base[f"fL{i}"], derivs[f"fL{i}"], primaries)
        for i in range(mix.n)
    ]
    f_v = [
        _adi_from_local_derivatives(base[f"fV{i}"], derivs[f"fV{i}"], primaries)
        for i in range(mix.n)
    ]
    return x_m, y_m, rho_l, rho_v, mu_l, mu_v, f_l, f_v


def _normalize_rows(a: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=float), 1.0e-300)
    return a / a.sum(axis=1, keepdims=True)


def _adi_from_local_derivatives(vals, deriv, primaries: Iterable[ADI]) -> ADI:
    vals = np.asarray(vals, dtype=float).ravel()
    primaries = list(primaries)
    jac = []
    for block in range(len(primaries[0].jac)):
        j = _diag_mul(deriv[:, 0], primaries[0].jac[block])
        for col, prm in enumerate(primaries[1:], start=1):
            j = j + _diag_mul(deriv[:, col], prm.jac[block])
        jac.append(j.tocsr())
    return ADI(vals, jac)


def _flow_prop_derivatives(mix, eos_model, p, T, z, base: FlowProps) -> dict[str, np.ndarray]:
    names = ["sL", "sV", "rhoL", "rhoV", "muL", "muV"]
    names += [f"xM{i}" for i in range(mix.n)] + [f"yM{i}" for i in range(mix.n)]
    return _finite_difference(
        lambda pp, zz: _flow_output_values(backend_flow_props(mix, eos_model, pp, T, zz)),
        p,
        z,
        _flow_output_values(base),
        names,
    )


def _natural_derivatives(mix, eos_model, p, T, x, y, base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ["rhoL", "rhoV", "muL", "muV"]
    names += [f"xM{i}" for i in range(mix.n)] + [f"yM{i}" for i in range(mix.n)]
    names += [f"fL{i}" for i in range(mix.n)] + [f"fV{i}" for i in range(mix.n)]
    return _finite_difference_two_compositions(
        lambda pp, xx, yy: _natural_output_values(mix, eos_model, pp, T, xx, yy),
        p,
        x,
        y,
        base,
        names,
    )


def _flow_output_values(fp: FlowProps) -> dict[str, np.ndarray]:
    out = {
        "sL": fp.sL,
        "sV": fp.sV,
        "rhoL": fp.rhoL,
        "rhoV": fp.rhoV,
        "muL": fp.muL,
        "muV": fp.muV,
    }
    for i, c in enumerate(fp.xM):
        out[f"xM{i}"] = c
    for i, c in enumerate(fp.yM):
        out[f"yM{i}"] = c
    return out


def _natural_output_values(mix, eos_model, p, T, x, y) -> dict[str, np.ndarray]:
    props = backend_natural_props(mix, eos_model, p, T, x, y)
    x_m = mass_fraction(mix, [x[:, i] for i in range(mix.n)])
    y_m = mass_fraction(mix, [y[:, i] for i in range(mix.n)])
    K = eos_model.evaluate_equilibrium_constants(p, T, _normalize_rows(0.5 * (x + y)))
    out = {
        "rhoL": props["rhoL"],
        "rhoV": props["rhoV"],
        "muL": props["muL"],
        "muV": props["muV"],
    }
    for i in range(mix.n):
        out[f"xM{i}"] = x_m[i]
        out[f"yM{i}"] = y_m[i]
        # Natural-equilibrium residual becomes (K_i x_i - y_i).  The caller
        # divides fugacity residuals by BARSA, so scale here to reuse that path.
        out[f"fL{i}"] = BARSA_SCALE * K[:, i] * x[:, i]
        out[f"fV{i}"] = BARSA_SCALE * y[:, i]
    return out


BARSA_SCALE = 1.0e5


def _finite_difference(fn, p, z, base, names) -> dict[str, np.ndarray]:
    nc, ncomp = z.shape
    out = {name: np.zeros((nc, 1 + ncomp), dtype=float) for name in names}
    dp = np.maximum(np.abs(p) * 1.0e-6, 10.0)
    plus = fn(p + dp, z)
    minus = fn(p - dp, z)
    for name in names:
        out[name][:, 0] = (plus[name] - minus[name]) / (2.0 * dp)
    for j in range(ncomp):
        dz = 1.0e-6
        zp = z.copy()
        zm = z.copy()
        zp[:, j] += dz
        zm[:, j] = np.maximum(zm[:, j] - dz, 1.0e-12)
        zp = _normalize_rows(zp)
        zm = _normalize_rows(zm)
        plus = fn(p, zp)
        minus = fn(p, zm)
        for name in names:
            out[name][:, 1 + j] = (plus[name] - minus[name]) / (2.0 * dz)
    return out


def _finite_difference_two_compositions(fn, p, x, y, base, names) -> dict[str, np.ndarray]:
    nc, ncomp = x.shape
    out = {name: np.zeros((nc, 1 + 2 * ncomp), dtype=float) for name in names}
    dp = np.maximum(np.abs(p) * 1.0e-6, 10.0)
    plus = fn(p + dp, x, y)
    minus = fn(p - dp, x, y)
    for name in names:
        out[name][:, 0] = (plus[name] - minus[name]) / (2.0 * dp)
    for j in range(ncomp):
        dx = 1.0e-6
        xp = x.copy()
        xm = x.copy()
        xp[:, j] += dx
        xm[:, j] = np.maximum(xm[:, j] - dx, 1.0e-12)
        plus = fn(p, _normalize_rows(xp), y)
        minus = fn(p, _normalize_rows(xm), y)
        for name in names:
            out[name][:, 1 + j] = (plus[name] - minus[name]) / (2.0 * dx)
    for j in range(ncomp):
        dy = 1.0e-6
        yp = y.copy()
        ym = y.copy()
        yp[:, j] += dy
        ym[:, j] = np.maximum(ym[:, j] - dy, 1.0e-12)
        plus = fn(p, x, _normalize_rows(yp))
        minus = fn(p, x, _normalize_rows(ym))
        for name in names:
            out[name][:, 1 + ncomp + j] = (plus[name] - minus[name]) / (2.0 * dy)
    return out


def _z_factor_from_density(prop, p, T, comp, is_liquid: bool):
    from pymrst.modules.compositional.fluids.blackoil.kvalue_model import R_GAS

    rho = prop.compute_molar_density(p, comp, is_liquid)
    return np.asarray(p, dtype=float).ravel() / (rho * R_GAS * np.asarray(T, dtype=float).ravel())


__all__ = [
    "backend_flash",
    "backend_flow_props",
    "backend_flow_props_ad",
    "backend_natural_props",
    "backend_natural_props_ad",
    "has_backend",
]
