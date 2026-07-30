"""Install MRST's no-flow-boundary MPFA pressure/gravity operators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pymrst.modules.mpfa import compute_multipoint_trans
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.adi import ad_matmul


def _field(value, *names):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _set_field(value, name, item):
    if isinstance(value, dict):
        value[name] = item
    else:
        setattr(value, name, item)


def _no_flow_operators(mpfa, grid):
    """Reduce the full MPFA system by eliminating boundary pressures."""
    trans = sp.csr_array(mpfa["cf_trans"])
    boundary = np.asarray(mpfa["sb"], dtype=bool)
    interior_sub = ~boundary
    nc = grid.cells.num

    a = trans[interior_sub, :nc]
    b = trans[interior_sub, nc:]
    c = trans[boundary, :nc]
    d = trans[boundary, nc:]
    correction = spsolve(d.tocsc(), c.toarray())
    reduced = sp.csr_array(a - b @ correction)

    internal_face = np.all(np.asarray(grid.faces.neighbors) >= 0, axis=1)
    d1 = sp.csr_array(mpfa["d1"])
    face_aggregate = d1[interior_sub][:, internal_face].T
    r_trans = sp.csr_array(face_aggregate @ reduced)

    g_trans = sp.csr_array(mpfa["_iDoBDo"] @ d1)
    gravity_reduced = (
        g_trans[interior_sub]
        + b @ spsolve(d.tocsc(), g_trans[boundary].toarray())
    )
    rg_trans = sp.csr_array(face_aggregate @ gravity_reduced).T
    return r_trans, rg_trans


def _install(model, matrix=None, **mpfa_options):
    if _field(model, "G") is None or _field(model, "rock") is None:
        raise TypeError("MPFA requires model.G and model.rock")
    if matrix is None:
        matrix = compute_multipoint_trans(model.G, model.rock, **mpfa_options)
    r_trans, rg_trans = _no_flow_operators(matrix, model.G)

    operators = _field(model, "operators")
    internal = np.all(np.asarray(model.G.faces.neighbors) >= 0, axis=1)
    neighbors = np.asarray(model.G.faces.neighbors)[internal]
    if not np.array_equal(neighbors, np.asarray(operators.N)):
        raise ValueError("operator neighborship does not match MPFA neighborship")

    face_t = get_face_transmissibility(model.G, model.rock)
    scale = -1.0 / (2.0 * face_t[internal])
    mpfa_grad = sp.diags_array(scale) @ r_trans
    gravity_weight = -sp.diags_array(scale / 2.0) @ rg_trans[internal]

    flow = _field(model, "FlowDiscretization")
    if flow is None:
        setup = _field(model, "setup_state_function_groupings",
                       "setupStateFunctionGroupings")
        if setup is None:
            raise TypeError("model does not expose FlowDiscretization")
        updated = setup()
        if updated is not None:
            model = updated
        flow = _field(model, "FlowDiscretization")

    pressure = flow.get_state_function("PressureGradient")
    pressure.Grad = lambda value: ad_matmul(mpfa_grad, value)
    flow.set_state_function("PressureGradient", pressure)

    gravity = flow.get_state_function("GravityPotentialDifference")
    gravity.weight = gravity_weight
    flow.set_state_function("GravityPotentialDifference", gravity)

    _set_field(operators, "mpfagrad", mpfa_grad)
    _set_field(model, "FlowDiscretization", flow)
    return model


def set_mpfa_discretization(model, M=None, **options):
    """Replace TPFA pressure and gravity gradients by MPFA operators."""
    parent = _field(model, "parentModel", "parent_model")
    if parent is not None:
        parent = _install(parent, M, **options)
        if hasattr(model, "parent_model"):
            _set_field(model, "parent_model", parent)
        else:
            _set_field(model, "parentModel", parent)
        return model
    return _install(model, M, **options)


setMPFADiscretization = set_mpfa_discretization


__all__ = ["set_mpfa_discretization", "setMPFADiscretization"]
