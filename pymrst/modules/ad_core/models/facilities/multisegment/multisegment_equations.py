"""MRST ``setupMSWellEquationSingleWell.m`` style equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.models.facilities.well_bore_friction import (
    well_bore_friction,
)
from pymrst.modules.ad_core.models.facilities.nozzle_valve import nozzle_valve
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
    ms.validate(len(w.cells))
    raw_topo = np.array([[seg.parent, seg.index] for seg in ms.segments], dtype=np.int64)
    real_segments = np.flatnonzero(raw_topo[:, 0] >= 0)
    topo = raw_topo[real_segments]
    n_nodes = len(ms.segments)
    n_seg = topo.shape[0]
    nph = len(q_s)
    rho_s = np.asarray(model.fluid.rho_s[:nph], dtype=float)
    p_nodes = _full_node_vector(bhp, node_pressure, n_nodes)
    q_abs = [ad_abs(q[0:1])[0] if hasattr(q, "__len__") else ad_abs(q) for q in q_s]
    mass_tot = q_abs[0] * rho_s[0]
    for ph in range(1, nph):
        mass_tot = mass_tot + q_abs[ph] * rho_s[ph]
    alpha = []
    for ph, a in enumerate(node_comp):
        root = q_abs[ph] * rho_s[ph] / mass_tot
        alpha.append(vertcat_adi(root, a))

    mix_s, rhom, mum = _node_mix(model, p_nodes, alpha, q_abs, rho_s)

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
    if getattr(model, "disgas", False) and nph >= 3:
        rs_cells = res_props.get("rs")
        if rs_cells is not None:
            cq_s[2] = cq_s[2] + rs_cells[wc] * cq_s[1]

    cell2node = sp.csr_array(
        (np.ones(wc.size), (perf_node, np.arange(wc.size))),
        shape=(n_nodes, wc.size),
    )
    vols = np.array([seg.volume for seg in ms.segments], dtype=float)
    if not np.any(vols):
        vols[:] = 1.0

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
            # MRST's segment divergence is positive outflow: parent receives
            # positive parent->child flux, child receives its negative.
            np.ravel(np.column_stack([np.ones(n_seg), -np.ones(n_seg)])),
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
    # MRST's MS operator gradient is child minus parent. For a producer the
    # segment mass flux is negative, both friction models return a positive
    # pressure loss, and pressure therefore rises from the wellhead toward
    # the completion nodes.
    grad_p = p_nodes[topo[:, 1]] - p_nodes[topo[:, 0]]
    grad_z = depth[topo[:, 1]] - depth[topo[:, 0]]
    rho_seg = _segment_average(rhom, topo)
    mu_seg = _segment_upstream(up, mum, topo)
    tubing_drop = well_bore_friction(
        segment_flux, rho_seg, mu_seg, np.maximum(diam, 1e-9),
        np.maximum(length, 0.0), rough, flowtype="massRate",
    )
    real = [ms.segments[i] for i in real_segments]
    tubing = np.array([seg.flow_model == "tubing" for seg in real], dtype=float)
    valves = np.array([seg.flow_model == "valve" for seg in real], dtype=float)
    valve_d = np.array([
        seg.valve_diameter if seg.valve_diameter > 0 else 1.0 for seg in real
    ])
    discharge = np.array([seg.discharge_coefficient for seg in real])
    counts = np.array([max(seg.valve_count, 1) for seg in real], dtype=float)
    valve_drop = nozzle_valve(
        segment_flux / counts, rho_seg, valve_d, discharge,
        flowtype="massRate",
    )
    pressure_drop = tubing * tubing_drop + valves * valve_drop
    pressure_eq = (
        grad_p - model.gravity_magnitude * rho_seg * grad_z - pressure_drop
    ) / value(bhp)[0]
    comp_eq = 1.0
    for ph in range(nph):
        comp_eq = comp_eq - alpha[ph][1:]

    return MultisegmentEquationResult(cq_s, well_eqs, node_mass_eqs, pressure_eq, comp_eq)


def _node_mix(model, pressure, alpha, q_abs, rho_s):
    """MRST ``getNodeMix``: mass fractions -> PVT-aware node mixture."""
    nph = len(alpha)
    # Non-root alpha values are component mass fractions. Convert them to
    # surface volume fractions; root fractions come directly from q_s.
    mix_s = []
    denom_mass = alpha[0] / rho_s[0]
    for ph in range(1, nph):
        denom_mass = denom_mass + alpha[ph] / rho_s[ph]
    q_total = q_abs[0]
    for ph in range(1, nph):
        q_total = q_total + q_abs[ph]
    for ph in range(nph):
        nonroot = (alpha[ph] / rho_s[ph]) / denom_mass
        root = q_abs[ph] / q_total
        mix_s.append(vertcat_adi(root, nonroot[1:]))

    fluid = model.raw_fluid
    bw = fluid.b_w(pressure)
    if hasattr(fluid, "mu_w"):
        muw = fluid.mu_w(pressure)
    else:
        muw = pressure * 0.0 + float(fluid.mu[0])
    fw = mix_s[0]
    if nph == 2:
        fo = mix_s[1]
        bo = fluid.b_o(pressure)
        muo = (fluid.mu_o(pressure, None, None)
               if hasattr(fluid, "mu_o") else
               pressure * 0.0 + float(fluid.mu[1]))
        volume = fw / bw + fo / bo
        mass = fw * rho_s[0] + fo * rho_s[1]
        mixr_w = (fw / bw) / volume
        mixr_o = (fo / bo) / volume
        return mix_s, mass / volume, mixr_w * muw + mixr_o * muo

    fo, fg = mix_s[1], mix_s[2]
    rs_sat = fluid.rs_sat(pressure)
    gor = fg / (fo + 1e-30)
    saturated = np.asarray(value(gor), dtype=float) >= np.asarray(value(rs_sat), dtype=float)
    sat = saturated.astype(float)
    rs = sat * rs_sat + (1.0 - sat) * gor
    bo = fluid.b_o(pressure, rs, saturated)
    muo = fluid.mu_o(pressure, rs, saturated)
    bg = fluid.b_g(pressure, None, None)
    mug = fluid.mu_g(pressure, None, None)
    free_gas = (fg - rs * fo)
    volume = fw / bw + fo / bo + free_gas / bg
    mass = fw * rho_s[0] + fo * rho_s[1] + fg * rho_s[2]
    mixr_w = (fw / bw) / volume
    mixr_o = (fo / bo) / volume
    mixr_g = (free_gas / bg) / volume
    viscosity = mixr_w * muw + mixr_o * muo + mixr_g * mug
    return mix_s, mass / volume, viscosity


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
