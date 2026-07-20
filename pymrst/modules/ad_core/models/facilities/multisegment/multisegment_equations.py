"""MRST ``setupMSWellEquationSingleWell.m`` style equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.models.facilities.well_bore_friction import (
    well_bore_friction,
)
from pymrst.utils.adi import ad_abs, ad_matmul, ad_sum, value, vertcat_adi


@dataclass
class MultisegmentEquationResult:
    connection_surface_rates: list
    well_rate_equations: list
    node_mass_equations: list
    segment_pressure_equations: object
    composition_equation: object


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _matvec(mat, vec):
    return mat @ vec


def _segment_upstream(flag, values, topo):
    vals = values
    upstream = np.where(np.asarray(value(flag), dtype=bool), topo[:, 0], topo[:, 1])
    return vals[upstream]


def _segment_average(values, topo):
    return 0.5 * (values[topo[:, 0]] + values[topo[:, 1]])


def assemble_multisegment_equations(
    w,
    model,
    q_s: Sequence,
    bhp,
    node_pressure,
    node_comp: Sequence,
    segment_flux,
    res_props,
    dt: float,
    previous_node_pressure=None,
    previous_node_comp=None,
    previous_mixture_density=None,
):
    """Assemble implicit multisegment well residuals.

    The unknowns mirror MRST: well phase rates ``q_s``, BHP, non-root node
    pressures, node component mass fractions, and segment mixture mass fluxes.
    """
    ms = w.multisegment
    raw_topo = np.array([[seg.parent, seg.index] for seg in ms.segments], dtype=np.int64)
    real_segments = np.flatnonzero(raw_topo[:, 0] >= 0)
    topo = raw_topo[real_segments]
    n_nodes = len(ms.segments)
    n_seg = topo.shape[0]
    nph = len(q_s)
    rho_s = np.asarray(model.fluid.rho_s[:nph], dtype=float)
    p_nodes = _full_node_vector(bhp, node_pressure, n_nodes)
    q_abs = [ad_abs(q[0:1])[0] if hasattr(q, "__len__") else ad_abs(q) for q in q_s]
    q_tot = q_abs[0]
    for q in q_abs[1:]:
        q_tot = q_tot + q
    alpha = []
    for ph, a in enumerate(node_comp):
        root = q_abs[ph] / q_tot
        alpha.append(vertcat_adi(root, a))

    pressure = res_props["pressure"]
    mob = _as_list(res_props["mob"])
    b = _as_list(res_props["b"])
    wc = np.asarray(w.cells, dtype=np.int64)
    perf_node = np.asarray(ms.perforation_segment, dtype=np.int64)
    if perf_node.size == 0:
        perf_node = np.arange(wc.size, dtype=np.int64) + 1
    perf_node = np.clip(perf_node, 0, n_nodes - 1)

    drawdown = pressure[wc] - p_nodes[perf_node]
    tw = np.asarray(w.WI, dtype=float)
    cq_r = [-tw * m[wc] * drawdown for m in mob[:nph]]
    cq_s = [b[i][wc] * cq_r[i] for i in range(nph)]

    cell2node = sp.csr_array(
        (np.ones(wc.size), (perf_node, np.arange(wc.size))),
        shape=(n_nodes, wc.size),
    )
    vols = np.array([seg.volume for seg in ms.segments], dtype=float)
    if not np.any(vols):
        vols[:] = 1.0

    rhom = sum(alpha[i] * rho_s[i] for i in range(nph))
    if previous_mixture_density is None:
        previous_mixture_density = np.asarray(value(rhom), dtype=float)
    if previous_node_comp is None:
        previous_node_comp = [np.asarray(value(a), dtype=float) for a in alpha]
    else:
        previous_node_comp = [
            np.concatenate([
                np.asarray(value(alpha[ph][0:1]), dtype=float),
                np.asarray(previous_node_comp[ph], dtype=float),
            ])
            for ph in range(nph)
        ]

    up = np.asarray(value(segment_flux), dtype=float) >= 0.0
    node_mass_eqs = []
    well_eqs = []
    div_mat = sp.csr_array(
        (
            np.ravel(np.column_stack([-np.ones(n_seg), np.ones(n_seg)])),
            (
                np.ravel(np.column_stack([topo[:, 0], topo[:, 1]])),
                np.ravel(np.column_stack([np.arange(n_seg), np.arange(n_seg)])),
            ),
        ),
        shape=(n_nodes, n_seg),
    )
    for ph in range(nph):
        seg_phase_flux = _segment_upstream(up, alpha[ph], topo) * segment_flux
        div = ad_matmul(div_mat, seg_phase_flux)
        conn = ad_matmul(cell2node, cq_s[ph]) * rho_s[ph]
        acc = (alpha[ph] * rhom - previous_node_comp[ph] * previous_mixture_density) * vols / dt
        ec = div + conn
        ec[1:] = ec[1:] + acc[1:]
        node_mass_eqs.append(ec[1:])
        well_eqs.append(q_s[ph] - ad_sum(ec) / rho_s[ph])

    depth = np.array([seg.depth for seg in ms.segments], dtype=float)
    length = np.array([seg.length for seg in ms.segments], dtype=float)[real_segments]
    diam = np.array([seg.diameter for seg in ms.segments], dtype=float)[real_segments]
    rough = np.array([seg.roughness for seg in ms.segments], dtype=float)[real_segments]
    grad_p = -(p_nodes[topo[:, 1]] - p_nodes[topo[:, 0]])
    grad_z = -(depth[topo[:, 1]] - depth[topo[:, 0]])
    rho_seg = _segment_average(rhom, topo)
    mu_seg = np.full(n_seg, 1e-3)
    friction = well_bore_friction(
        segment_flux, rho_seg, mu_seg, np.maximum(diam, 1e-9),
        np.maximum(length, 0.0), rough, flowtype="massRate",
    )
    pressure_eq = (grad_p - model.gravity_magnitude * rho_seg * grad_z - friction) / value(bhp)[0]
    comp_eq = 1.0
    for ph in range(nph):
        comp_eq = comp_eq - alpha[ph][1:]

    return MultisegmentEquationResult(cq_s, well_eqs, node_mass_eqs, pressure_eq, comp_eq)


def _full_node_vector(bhp, node_pressure, n_nodes):
    root = bhp[0:1] if hasattr(bhp, "__len__") else np.atleast_1d(bhp)
    return vertcat_adi(root, node_pressure)


__all__ = ["MultisegmentEquationResult", "assemble_multisegment_equations"]


if __name__ == "__main__":
    from dataclasses import dataclass

    from pymrst.params.wells_and_bc import MultisegmentWell, Well, WellSegment

    @dataclass
    class Fluid:
        rho_s: np.ndarray

    @dataclass
    class Model:
        fluid: Fluid
        gravity_magnitude: float = 9.80665

    def test_assemble_multisegment_equations_shapes():
        segments = [
            WellSegment(0, -1, depth=1000.0, diameter=0.1, volume=1.0),
            WellSegment(1, 0, length=10.0, depth=1010.0, diameter=0.1, roughness=1e-5, volume=1.0),
            WellSegment(2, 1, length=10.0, depth=1020.0, diameter=0.1, roughness=1e-5, volume=1.0),
        ]
        ms = MultisegmentWell(segments, np.array([1, 2]))
        w = Well(
            cells=np.array([0, 1]), type="rate", val=1.0, r=np.ones(2),
            dir="z", WI=np.ones(2) * 1e-12, dZ=np.zeros(2), name="MS",
            compi=np.array([1.0, 0.0]), refDepth=1000.0, multisegment=ms,
        )
        model = Model(Fluid(np.array([1000.0, 800.0])))
        res = assemble_multisegment_equations(
            w, model,
            q_s=[np.array([1.0]), np.array([0.0])],
            bhp=np.array([200e5]),
            node_pressure=np.array([199e5, 198e5]),
            node_comp=[np.array([1.0, 1.0]), np.array([0.0, 0.0])],
            segment_flux=np.array([1.0, 1.0]),
            res_props={
                "pressure": np.array([201e5, 202e5]),
                "mob": [np.ones(2) * 1e-9, np.ones(2) * 1e-10],
                "b": [np.ones(2), np.ones(2)],
            },
            dt=1.0,
        )
        assert len(res.well_rate_equations) == 2
        assert len(res.node_mass_equations) == 2
        assert np.asarray(res.segment_pressure_equations).size == 2
        print("ok test_assemble_multisegment_equations_shapes")

    test_assemble_multisegment_equations_shapes()
