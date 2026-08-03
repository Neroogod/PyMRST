#!/usr/bin/env python3
"""Qualify PyMRST AD Jacobians and gradients against finite differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState, TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_core.utils.boundary_conditions import (
    compute_sources_and_boundary_conditions_ad,
)
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.optimization.adjoint import (
    evaluate_objective, npv_objective, simulate_forward,
)
from pymrst.modules.optimization.utils.compute_gradient_adjoint_ad import (
    compute_gradient_adjoint,
)
from pymrst.params import BoundaryCondition, Source, add_well, make_rock
from pymrst.research_studio.validation import (
    EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence,
    qualify_directional_derivative,
)
from pymrst.utils.adi import combine_equations, init_variables_adi, vertcat_adi
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/ad_finite_difference_evidence_graph.json"


def _ow_case(nx=4, ny=3):
    grid = compute_geometry(cart_grid([nx, ny, 1], [160.0, 120.0, 12.0]))
    rock = make_rock(grid, 120 * MILLI * DARCY, 0.22)
    fluid = init_simple_adi_fluid(
        mu=[1 * CENTI * POISE, 4 * CENTI * POISE], rho=[1000.0, 780.0],
        n=[2.0, 2.5], c=[2e-10, 8e-10], p_ref=220 * BARSA,
    )
    wells = add_well(None, grid, rock, np.array([0]), well_type="rate",
                     val=25.0 / DAY, compi=np.array([1.0, 0.0]), name="I")
    wells = add_well(wells, grid, rock, np.array([grid.cells.num - 1]),
                     well_type="bhp", val=205 * BARSA,
                     compi=np.array([0.0, 1.0]), name="P")
    model = TwoPhaseOilWaterModelAD(grid, rock, fluid, wells)
    state0 = model.initial_state(220 * BARSA, 0.25)
    nc = grid.cells.num
    state = state0.copy()
    state.pressure += np.linspace(1.5, -2.0, nc) * BARSA
    state.sW += np.linspace(0.015, -0.01, nc)
    state.qWs = np.array([25.0 / DAY, -2.0 / DAY])
    state.qOs = np.array([0.0, -18.0 / DAY])
    state.bhp = np.array([224.0, 205.0]) * BARSA
    return model, state0, state, 10 * DAY


def _pack_ow(state: OWState):
    return np.concatenate((state.pressure, state.sW, state.qWs, state.qOs, state.bhp))


def _unpack_ow(vector, nc, nw):
    x = np.asarray(vector, float)
    offsets = np.cumsum([0, nc, nc, nw, nw, nw])
    parts = [x[offsets[i]:offsets[i + 1]].copy() for i in range(5)]
    return OWState(*parts)


def _full_residual_qualification():
    model, state0, state, dt = _ow_case()
    nc, nw = model.g.cells.num, len(model.wells)
    variables = init_variables_adi(state.pressure, state.sW, state.qWs,
                                   state.qOs, state.bhp)
    equations = model.assemble(state0, *variables, dt)
    assembled = combine_equations(*equations)
    point = _pack_ow(state)
    scales = np.concatenate((
        np.full(nc, BARSA), np.full(nc, 0.1),
        np.full(nw, 10.0 / DAY), np.full(nw, 10.0 / DAY),
        np.full(nw, BARSA),
    ))

    def numerical(vector):
        st = _unpack_ow(vector, nc, nw)
        return np.concatenate([np.asarray(eq, float).ravel()
                               for eq in model.assemble(
                                   state0, st.pressure, st.sW, st.qWs,
                                   st.qOs, st.bhp, dt)])

    row_groups = {
        "full_residual": np.arange(assembled.val.size),
        "reservoir_mass": np.arange(2 * nc),
        "well_closure": np.arange(2 * nc, 2 * nc + 2 * nw),
        "well_controls": np.arange(2 * nc + 2 * nw, assembled.val.size),
    }
    results = {}
    matrix = assembled.jac[0].tocsr()
    for name, rows in row_groups.items():
        results[name] = qualify_directional_derivative(
            matrix[rows], lambda x, rows=rows: numerical(x)[rows], point,
            scales=scales, relative_tolerance=3e-5,
            component_tolerance=5e-4,
        )
    return results, {"cells": nc, "wells": nw, "unknowns": point.size,
                     "equations": assembled.val.size}


def _force_qualification():
    # Two cells are enough to exercise pressure inflow, flux outflow,
    # injection composition and mobility-weighted production simultaneously.
    from types import SimpleNamespace
    grid = SimpleNamespace(
        cells=SimpleNamespace(num=2, centroids=np.array([[0.5, 0., 0.], [1.5, 0., 0.]])),
        faces=SimpleNamespace(neighbors=np.array([[-1, 0], [0, 1], [1, -1]]),
                              centroids=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])),
    )
    model = SimpleNamespace(
        G=grid, operators=SimpleNamespace(t_all=np.array([2.0, 1.0, 3.0])),
        fluid=SimpleNamespace(rho_s=np.array([1000., 800.])), gravity_magnitude=0.0,
    )
    bc = BoundaryCondition(face=np.array([0, 2]), type=["pressure", "flux"],
                           value=np.array([115., -3.]),
                           sat=np.array([[0.8, 0.2], [0.1, 0.9]]))
    src = Source(cell=np.array([0, 1]), rate=np.array([2.0, -1.5]),
                 sat=np.array([[1., 0.], [0., 1.]]))
    point = np.array([100., 92., 0.25, 0.55])

    def evaluate(x, automatic):
        if automatic:
            p, sw = init_variables_adi(x[:2], x[2:])
        else:
            p, sw = x[:2], x[2:]
        sat = [sw, 1.0 - sw]
        mob = [sw**2, (1.0 - sw)**2 / 3.0]
        density = [1000.0 * (1.0 + 1e-3 * (p - 100.0)),
                   800.0 * (1.0 + 2e-3 * (p - 100.0))]
        src_c, bc_c = compute_sources_and_boundary_conditions_ad(
            model, [p, p], sat, mob, density, None, {"bc": bc, "src": src})
        return vertcat_adi(*(src_c.phaseMass + bc_c.phaseMass))

    ad = evaluate(point, True)
    matrix = np.hstack([block.toarray() for block in ad.jac])
    result = qualify_directional_derivative(
        matrix, lambda x: evaluate(x, False), point,
        scales=np.array([10., 10., 0.1, 0.1]), relative_tolerance=2e-6,
        component_tolerance=2e-5, absolute_tolerance=1e-5,
    )
    return result


def _objective_gradient_qualification():
    model, state0, _, _ = _ow_case(2, 2)
    dts = [8.0 * DAY, 8.0 * DAY]
    controls = np.array([[25.0 / DAY, 205 * BARSA]] * 2)
    objective = npv_objective()
    adjoint = compute_gradient_adjoint(model, state0, dts, controls, objective)
    finite = np.zeros_like(controls)
    for step in range(controls.shape[0]):
        for well in range(controls.shape[1]):
            h = 1e-6 * max(abs(controls[step, well]), 1.0)
            up, down = controls.copy(), controls.copy()
            up[step, well] += h
            down[step, well] -= h
            values = []
            for candidate in (up, down):
                states = simulate_forward(model, state0, dts, candidate)
                values.append(evaluate_objective(model, objective, states, dts, candidate))
            finite[step, well] = (values[0] - values[1]) / (2 * h)
    difference = adjoint.gradient - finite
    relative = np.abs(difference) / np.maximum(np.abs(finite), 1e-30)
    return {
        "controls": int(controls.size),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "maximum_relative_error": float(np.max(relative)),
        "passed": bool(np.max(relative) < 1e-5),
    }


def _edge(edge_id, source, claim, result, *, shared):
    values = result.to_dict() if hasattr(result, "to_dict") else result
    metrics = {key: value for key, value in values.items()
               if key != "passed" and isinstance(value, (int, float))}
    return EvidenceEdge(
        edge_id, source, "pymrst_ad", "finite_difference_derivative", claim,
        Independence.PARTIAL, shared_kernels=tuple(shared), metrics=metrics,
        accepted=bool(values["passed"]), converged=True,
        mappings=("dimensionless_variable_scaling:v1", "central_richardson:v1"),
    )


def qualify():
    residuals, case = _full_residual_qualification()
    force = _force_qualification()
    objective = _objective_gradient_qualification()
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(
        "pymrst_ad", EvidenceNodeKind.PYMRST, "PyMRST sparse AD derivatives",
        version="working-tree", metadata=case,
    ))
    shared = ("residual value implementation", "case setup")
    for name, result in residuals.items():
        node = f"fd_{name}"
        graph.add_node(EvidenceNode(node, EvidenceNodeKind.FINITE_DIFFERENCE,
                                   f"Richardson finite difference: {name}"))
        graph.add_edge(_edge(
            f"{name}_ad_vs_fd", node,
            f"AD {name.replace('_', ' ')} derivative agrees with finite differences",
            result, shared=shared,
        ))
    graph.add_node(EvidenceNode("fd_forces", EvidenceNodeKind.FINITE_DIFFERENCE,
                               "Finite difference: source and boundary fluxes"))
    graph.add_edge(_edge(
        "forces_ad_vs_fd", "fd_forces",
        "AD source and boundary-condition derivatives agree with finite differences",
        force, shared=("force value implementation", "two-cell force setup"),
    ))
    graph.add_node(EvidenceNode("fd_objective", EvidenceNodeKind.FINITE_DIFFERENCE,
                               "Full forward rerun finite-difference objective gradient"))
    graph.add_edge(_edge(
        "objective_gradient_adjoint_vs_fd", "fd_objective",
        "Adjoint objective gradient agrees with full-simulation finite differences",
        objective, shared=("forward reservoir model", "objective definition"),
    ))
    all_results = [*residuals.values(), force, objective]
    summary = {
        "case": "oil_water_residual_wells_forces_and_objective",
        "passed": all((r.passed if hasattr(r, "passed") else r["passed"])
                      for r in all_results),
        "model": case,
        "residual_derivatives": {key: value.to_dict()
                                 for key, value in residuals.items()},
        "force_derivatives": force.to_dict(),
        "objective_gradient": objective,
    }
    return graph, summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify()
    payload = {"artifact": "pymrst_ad_finite_difference_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(),
               "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {len(graph.edges)} derivative edges")
    print(args.output)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
