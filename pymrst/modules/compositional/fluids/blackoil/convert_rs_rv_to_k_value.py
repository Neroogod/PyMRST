"""Port of MRST ``convertRSRVtoKValue.m``.

The routine samples a dissolved-gas black-oil fluid over pressure and
solution-gas ratio, converts each sample to two pseudo-component phase
fractions, and interpolates the result onto regular pressure/composition
tables.  The resulting tables are what MRST's
``convertBlackOilModelToCompositionalModel`` uses to build a K-value
compositional model for SPE1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator, griddata

from pymrst.modules.compositional.fluids.blackoil.black_oil_to_mass_fraction import (
    black_oil_to_mass_fraction,
)
from pymrst.utils.units import BARSA


@dataclass
class KValueTables:
    """MRST ``props.k_values`` table group."""

    K_o: np.ndarray
    K_g: np.ndarray
    freeGas: np.ndarray
    z_g: np.ndarray
    z_o: np.ndarray
    pressure: np.ndarray


@dataclass
class BlackOilPropertyTables:
    """MRST ``props.properties`` table group."""

    rhoO: np.ndarray
    rhoG: np.ndarray
    muO: np.ndarray
    muG: np.ndarray
    pressure: np.ndarray
    x_g: np.ndarray
    x_o: np.ndarray
    y_g: np.ndarray
    y_o: np.ndarray


@dataclass
class RSRVKValueProperties:
    """Output of ``convert_rs_rv_to_k_value``."""

    k_values: KValueTables
    properties: BlackOilPropertyTables


def convert_rs_rv_to_k_value(
    model,
    *,
    np_: int = 100,
    nz: int = 500,
    np_sample: int | None = None,
    nz_sample: int | None = None,
    interpolation: str = "linear",
    min_pressure: float = 300.0 * BARSA,
    max_pressure: float = 500.0 * BARSA,
) -> RSRVKValueProperties:
    """Convert black-oil Rs/Rv tables to pseudo-component K-value tables.

    Parameters mirror MRST ``convertRSRVtoKValue``.  ``np_sample`` and
    ``nz_sample`` are accepted for API parity; MRST currently preserves
    ``np``/``nz`` even when those options are supplied, so PyMRST does the same.
    RV/vaporised-oil support is intentionally not added here because the MRST
    source notes that it is not supported by this converter.
    """
    del np_sample, nz_sample
    fluid = model.fluid
    p_axis = np.linspace(min_pressure, max_pressure, int(np_))
    if getattr(model, "disgas", True):
        rs_sat = fluid.rs_sat(p_axis)
        rs_axis = np.linspace(1.0, 2.0 * float(np.max(rs_sat)), int(nz))
    else:
        rs_axis = np.linspace(1.0, 100.0, int(nz))

    P, RS = np.meshgrid(p_axis, rs_axis)
    SG = np.zeros_like(RS)
    for p_no, p_val in enumerate(p_axis):
        rssat = float(fluid.rs_sat(np.array([p_val]))[0]) if getattr(model, "disgas", True) else 0.0
        active = RS[:, p_no] > rssat
        RS[active, p_no] = rssat
        if np.any(active):
            SG[active, p_no] = np.linspace(0.0, 1.0, int(np.count_nonzero(active)))

    p = P.ravel(order="F")
    s_g = SG.ravel(order="F")
    rs = RS.ravel(order="F")
    s_o = 1.0 - s_g
    props = black_oil_to_mass_fraction(fluid, p, s_o, s_g, rs, np.nan, disgas=getattr(model, "disgas", True))

    with np.errstate(divide="ignore", invalid="ignore"):
        K_o = props.yo / props.xo
        K_g = props.yg / props.xg

    z_axis = np.linspace(0.0, 1.0, int(nz))
    p_sample = np.linspace(p_axis[0], p_axis[-1], int(np_))
    p_s, zg_s = np.meshgrid(p_sample, z_axis)
    max_k = 1.0e8

    k_o = _interpolate_to_mesh_2d(p, props.zg, K_o, p_s, zg_s, interpolation)
    k_g = _interpolate_to_mesh_2d(p, props.zg, K_g, p_s, zg_s, interpolation)
    # MRST computes this with scatteredInterpolant(..., 'natural'), but the
    # converted model immediately sets isStableFn = [] and does not use it for
    # the SPE1 K-value run.  Store a bounded transition table instead of a cubic
    # approximation, which can overshoot outside [0, 1] in SciPy.
    free_gas = _interpolate_to_mesh_2d(p, props.zg, props.freeGas.astype(float), p_s, zg_s, "linear")
    free_gas = np.clip(free_gas, 0.0, 1.0)
    k_o[~np.isfinite(k_o)] = max_k
    k_g[~np.isfinite(k_g)] = max_k

    k_values = KValueTables(
        K_o=k_o,
        K_g=k_g,
        freeGas=free_gas,
        z_g=zg_s,
        z_o=1.0 - zg_s,
        pressure=p_s,
    )

    p_s2, z_s = np.meshgrid(p_sample, z_axis)
    properties = BlackOilPropertyTables(
        rhoO=_interpolate_to_mesh_2d(p, props.xo, props.rhoO, p_s2, z_s, interpolation),
        rhoG=_interpolate_to_mesh_2d(p, props.yg, props.rhoG, p_s2, z_s, interpolation),
        muO=_interpolate_to_mesh_2d(p, props.xo, props.muO, p_s2, z_s, interpolation),
        muG=_interpolate_to_mesh_2d(p, props.yg, props.muG, p_s2, z_s, interpolation),
        pressure=p_s2,
        x_g=1.0 - z_s,
        x_o=z_s,
        y_g=z_s,
        y_o=1.0 - z_s,
    )
    return RSRVKValueProperties(k_values=k_values, properties=properties)


def _interpolate_to_mesh_2d(X, Y, Z, sx, sy, kind: str = "linear") -> np.ndarray:
    """Python equivalent of MRST's local ``interpolateToMesh2D`` helper."""
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)

    if X.size == 1:
        unique, idx = np.unique(Y, return_index=True)
        return np.interp(sy.ravel(), unique, Z[idx]).reshape(sy.shape)
    if Y.size == 1:
        unique, idx = np.unique(X, return_index=True)
        return np.interp(sx.ravel(), unique, Z[idx]).reshape(sx.shape)

    xy = np.column_stack([X, Y])
    points, idx = np.unique(xy, axis=0, return_index=True)
    values = Z[idx]

    if np.allclose(points[:, 1].max(), points[:, 1].min()):
        order = np.argsort(points[:, 0])
        return np.interp(sx.ravel(), points[order, 0], values[order]).reshape(sx.shape)
    if np.allclose(points[:, 0].max(), points[:, 0].min()):
        order = np.argsort(points[:, 1])
        return np.interp(sy.ravel(), points[order, 1], values[order]).reshape(sy.shape)

    query = np.column_stack([sx.ravel(), sy.ravel()])
    mins = points.min(axis=0)
    spans = np.ptp(points, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    points_i = (points - mins) / spans
    query_i = (query - mins) / spans
    if kind == "nearest":
        out = NearestNDInterpolator(points_i, values)(query_i)
    elif kind == "natural":
        try:
            out = CloughTocher2DInterpolator(points_i, values, fill_value=np.nan)(query_i)
        except Exception:
            out = griddata(points_i, values, query_i, method="linear")
    else:
        method = "cubic" if kind == "pchip" else kind
        try:
            out = griddata(points_i, values, query_i, method=method)
        except Exception:
            out = LinearNDInterpolator(points_i, values, fill_value=np.nan)(query_i)
    nearest = NearestNDInterpolator(points, values)(query)
    out = np.where(np.isfinite(out), out, nearest)
    return np.asarray(out, dtype=float).reshape(sx.shape)


if __name__ == "__main__":
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import setup_spe1_tutorial

    def test_spe1_k_value_tables():
        _, model, _ = setup_spe1_tutorial()
        props = convert_rs_rv_to_k_value(model, np_=12, nz=16)
        assert props.k_values.K_o.shape == (16, 12)
        assert props.k_values.K_g.shape == (16, 12)
        assert props.properties.rhoO.shape == (16, 12)
        assert np.all(np.isfinite(props.k_values.K_o))
        assert np.all(np.isfinite(props.k_values.K_g))
        assert np.nanmax(props.properties.rhoO) > np.nanmax(props.properties.rhoG)
        print("ok test_spe1_k_value_tables")

    test_spe1_k_value_tables()
    print("\nAll tests passed!")
