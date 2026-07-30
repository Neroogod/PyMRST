"""Shared kernels for MRST black-oil phase flux/property utilities."""

import warnings

import numpy as np

from pymrst.modules.ad_core.utils.split_face_cell_value import split_face_cell_value
from pymrst.modules.ad_blackoil.utils.get_b_g_bo import get_b_g_bo
from pymrst.modules.ad_blackoil.utils.get_b_o_bo import get_b_o_bo
from pymrst.utils.adi import value


def _fluid(model):
    return getattr(model, "raw_fluid", getattr(model, "fluid", None))


def _sizes(model):
    return model.operators.T.size, model.G.cells.num if hasattr(model, "G") else model.g.cells.num


def _split(model, flag, values):
    return split_face_cell_value(model.operators, flag, values, _sizes(model))


def _warn_negative(name, values):
    if np.any(np.asarray(value(values)) < 0):
        warnings.warn(f"negative {name} compressibility present", RuntimeWarning, stacklevel=3)


def get_flux_and_props_water_bo(model, p_o, s_w, kr_w, transmissibility, gdz):
    fluid, ops = _fluid(model), model.operators
    pc = fluid.pc_ow(s_w) if getattr(fluid, "pc_ow", None) is not None and s_w is not None else 0.0
    p_w = p_o - pc
    mu_w = fluid.mu_w(p_w)
    b_w = fluid.b_w(p_w)
    rho_w = b_w*fluid.rhoS[0]
    dp_w = ops.grad(p_w) - ops.avg(rho_w)*gdz
    upwind = np.asarray(value(dp_w)) <= 0
    kr_face, kr_cell = _split(model, upwind, kr_w)
    mu_face, mu_cell = _split(model, upwind, mu_w)
    mobility = kr_cell/mu_cell
    flux = -(kr_face/mu_face)*transmissibility*dp_w
    _warn_negative("water", b_w)
    return flux, b_w, mobility, rho_w, p_w, upwind, dp_w, mu_cell


def get_flux_and_props_oil_bo(model, p_o, s_o, kr_o, transmissibility, gdz,
                              rs=None, is_saturated=None):
    fluid, ops = _fluid(model), model.operators
    if getattr(model, "disgas", False):
        if rs is None or is_saturated is None:
            raise ValueError("rs and is_saturated are required when disgas is enabled")
        b_o = get_b_o_bo(model, p_o, rs, is_saturated)
        mu_o = fluid.mu_o(p_o, rs, is_saturated)
        rho_o = b_o*(rs*fluid.rhoS[2] + fluid.rhoS[1])
    else:
        b_o = get_b_o_bo(model, p_o)
        boxmu = getattr(fluid, "bo_x_mu_o", getattr(fluid, "BOxmuO", None))
        mu_o = boxmu(p_o)*b_o if boxmu is not None else fluid.mu_o(p_o)
        rho_o = b_o*fluid.rhoS[1]
    dp_o = ops.grad(p_o) - ops.avg(rho_o)*gdz
    upwind = np.asarray(value(dp_o)) <= 0
    kr_face, kr_cell = _split(model, upwind, kr_o)
    mu_face, mu_cell = _split(model, upwind, mu_o)
    mobility = kr_cell/mu_cell
    flux = -(kr_face/mu_face)*transmissibility*dp_o
    _warn_negative("oil", b_o)
    return flux, b_o, mobility, rho_o, p_o, upwind, dp_o, mu_cell


def get_flux_and_props_gas_bo(model, p_o, s_g, kr_g, transmissibility, gdz,
                              rv=None, is_saturated=None):
    fluid, ops = _fluid(model), model.operators
    pc = fluid.pc_og(s_g) if getattr(fluid, "pc_og", None) is not None and s_g is not None else 0.0
    p_g = p_o + pc
    if getattr(model, "vapoil", False):
        if rv is None or is_saturated is None:
            raise ValueError("rv and is_saturated are required when vapoil is enabled")
        b_g = get_b_g_bo(model, p_g, rv, is_saturated)
        mu_g = fluid.mu_g(p_g, rv, is_saturated)
    else:
        rv = 0.0 if rv is None else rv
        b_g = get_b_g_bo(model, p_g)
        mu_g = fluid.mu_g(p_g)
    rho_g = b_g*(rv*fluid.rhoS[1] + fluid.rhoS[2])
    dp_g = ops.grad(p_g) - ops.avg(rho_g)*gdz
    upwind = np.asarray(value(dp_g)) <= 0
    kr_face, kr_cell = _split(model, upwind, kr_g)
    mu_face, mu_cell = _split(model, upwind, mu_g)
    mobility = kr_cell/mu_cell
    flux = -(kr_face/mu_face)*transmissibility*dp_g
    _warn_negative("gas", b_g)
    return flux, b_g, mobility, rho_g, p_g, upwind, dp_g, mu_cell
