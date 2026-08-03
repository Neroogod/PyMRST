#!/usr/bin/env python3
"""Generate the integrated M4.1--M4.6 qualification artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from pymrst.research_studio.components import (
    ComponentRequest, GuardedSurrogateRuntime, ReferenceSamplingPolicy,
    SurrogateContract, SurrogatePrediction, TensorField, ValidityDomain,
    registered_reference_components,
)
from pymrst.research_studio.optimization import (
    ConstraintDefinition, ObjectiveDefinition, OptimizationProblem,
    evaluate_with_gradient,
)
from pymrst.research_studio.performance import (
    BackendDescriptor, PerformanceLaboratory, add_at_reduction,
    bincount_reduction, reference_solver_registry, sorted_segment_reduction,
    sparse_construction_benchmark,
)


def _physics(values):
    return {"response": 2.0*values["input"]}


def _predict(values, device):
    return SurrogatePrediction(
        {"response": 2.0*values["input"]+1e-7},
        uncertainty=np.zeros_like(values["input"]))


def _objective(x, context):
    return context["price"]*x[0]-0.5*np.dot(x, x)


def _gradient(x, context):
    return _objective(x, context), np.array([context["price"]-x[0], -x[1]])


def _constraint(x, context):
    return np.array([x[0]+1.0, x[1]+2.0])


def qualify(output: Path):
    registry = registered_reference_components()
    selected = registry.validate(ComponentRequest(
        {"model": "ad-blackoil-reference", "linear_solver": "scipy-direct",
         "backend": "numpy-scipy-reference"},
        phases=("water", "oil", "gas"), physics=("black-oil",),
        derivative="adi", restart=True))

    contract = SurrogateContract(
        "m4-qualified-linear-surrogate",
        (TensorField("input", "dimensionless", (None, 1)),),
        (TensorField("response", "dimensionless", (None, 1)),),
        ValidityDomain({"input": (-2.0, 2.0)}), "sha256:manufactured-training",
        "variance", ("finite",), "builtins:dict",
        ReferenceSamplingPolicy(0.5, 1, 13), maximum_batch_size=16)
    guarded = GuardedSurrogateRuntime(
        contract, _predict, fallback=_physics, uncertainty_limit=1e-3)
    guarded({"input": np.array([[-1.0], [0.0], [1.0]])})
    guarded({"input": np.array([[3.0]])})

    problem = OptimizationProblem(
        ObjectiveDefinition("quadratic economics", _objective), ("rate", "bhp"),
        ((-10.0, 10.0), (-10.0, 10.0)),
        (ConstraintDefinition("normalized pressure", _constraint,
                              scale=2.0, limit=3.0, rho=20.0),),
        metadata={"qualification": "M4.4"})
    differentiation = evaluate_with_gradient(
        problem, [2.0, -1.0], {"price": 4.0}, gradient_owner=_gradient,
        evidence={"source": "integrated M4 qualification"})

    rng = np.random.default_rng(4)
    indices = rng.integers(0, 200, 20000)
    values = rng.normal(size=indices.size)
    performance = PerformanceLaboratory(
        repeats=5, absolute_tolerance=1e-12, relative_tolerance=1e-13).qualify(
        "duplicate reduction",
        (BackendDescriptor("np.add.at", np.__version__, reference=True),
         add_at_reduction),
        {"bincount": (BackendDescriptor("bincount", np.__version__),
                      bincount_reduction),
         "sorted-segment": (BackendDescriptor("sorted-segment", np.__version__),
                            sorted_segment_reduction)},
        (indices, values, 200))
    sparse_evidence = sparse_construction_benchmark(
        sp.random(1000, 1000, density=.01, format="csr", random_state=4), repeats=5)
    matrix = sp.diags((-np.ones(99), np.full(100, 3.0), -np.ones(99)),
                      (-1, 0, 1), format="csr")
    solvers = reference_solver_registry().benchmark(matrix, np.ones(100), tolerance=1e-9)

    artifact = {
        "schema_version": 1, "work_packages": [f"M4.{i}" for i in range(1, 7)],
        "undefined_work_packages": ["M4.7", "M4.8", "M4.9"],
        "capabilities": {slot: value.to_dict() for slot, value in selected.items()},
        "guarded_surrogate": guarded.report(),
        "differentiation": {
            "problem_id": differentiation.problem_id,
            "value": differentiation.value,
            "gradient": differentiation.gradient.tolist(),
            "finite_difference_gradient": differentiation.finite_difference_gradient.tolist(),
            "maximum_fd_error": differentiation.maximum_fd_error,
            "constraints": dict(differentiation.constraints),
            "evidence": dict(differentiation.evidence),
        },
        "backend_qualification": {
            "name": performance.name, "passed": performance.passed,
            "reference_backend": performance.reference_backend,
            "hardware": dict(performance.hardware),
            "measurements": {name: asdict(value)
                             for name, value in performance.measurements.items()},
            "sparse_construction": sparse_evidence,
        },
        "solver_qualification": {name: asdict(value) for name, value in solvers.items()},
        "acceptance": {
            "component_stack_compatible": True,
            "all_surrogate_calls_accounted": guarded.report()["calls"] == 2,
            "ood_fallback_recorded": guarded.report()["ood_events"] == 1,
            "gradient_fd_passed": differentiation.maximum_fd_error < 1e-9,
            "qualified_backends_passed": performance.passed,
            "scipy_solvers_passed": all(solvers[name].passed for name in
                                        ("scipy-direct", "scipy-cg", "scipy-gmres")),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True,
                                    allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(output)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/research_studio/experiments/m4_qualification.json"))
    arguments = parser.parse_args()
    artifact = qualify(arguments.output)
    print(json.dumps(artifact["acceptance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
