#!/usr/bin/env python3
"""Build physics-versus-Victory evidence from a completed flash benchmark.

This qualification deliberately separates a *completed experiment* from an
*accepted production surrogate*.  Victory receives reference phase flags and
therefore cannot earn a stability-classification claim from this dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.research_studio.validation import (
    EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeKind, Independence,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/victory_beta2.2_10000"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/qualification/victory_surrogate_evidence_graph.json"
OUTPUTS = ("x1", "x3", "x14", "y1", "y3", "y14", "L")
PHYSICS_LIMITS = {
    "x1": 5e-4, "x3": 5e-4, "x14": 5e-4,
    "y1": 5e-4, "y3": 5e-4, "y14": 5e-4,
    "L": 2e-3, "Z_L": 5e-4, "Z_V": 5e-4,
}
SURROGATE_RMSE_LIMIT = 1e-2
SURROGATE_MAX_LIMIT = 1e-1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_index(rows):
    return {(row["left"], row["right"], row["quantity"]): row for row in rows}


def _float(row, name):
    return float(row[name])


def _edge(edge_id, source, target, evidence_class, claim, *, accepted,
          metrics, independence=Independence.PARTIAL, shared=(), exceptions=()):
    return EvidenceEdge(
        edge_id, source, target, evidence_class, claim, independence,
        mappings=("state_digest:sha256", "component_name:exact", "row_identity:exact"),
        shared_kernels=tuple(shared), converged=True, accepted=bool(accepted),
        exceptions=tuple(exceptions), metrics=metrics,
    )


def _error_atlas(predictions: Path):
    data = np.genfromtxt(predictions, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    p = np.asarray(data["p"], float)
    reference_phase = np.asarray(data["reference_phase"], str)
    uncertainty = np.column_stack([
        np.asarray(data[f"victory_{name}_std"], float) for name in OUTPUTS
    ])
    error = np.column_stack([
        np.abs(np.asarray(data[f"victory_{name}"], float)
               - np.asarray(data[f"reference_{name}"], float))
        for name in OUTPUTS
    ])
    pressure_edges = np.unique(np.quantile(p, np.linspace(0, 1, 6)))

    def describe(mask):
        if not np.any(mask):
            return {"count": 0}
        return {
            "count": int(np.sum(mask)),
            "mean_absolute_error": {
                name: float(np.mean(error[mask, i])) for i, name in enumerate(OUTPUTS)
            },
            "p95_absolute_error": {
                name: float(np.quantile(error[mask, i], 0.95)) for i, name in enumerate(OUTPUTS)
            },
        }

    pressure = []
    for i in range(pressure_edges.size - 1):
        upper_closed = i == pressure_edges.size - 2
        mask = ((p >= pressure_edges[i]) &
                ((p <= pressure_edges[i + 1]) if upper_closed else
                 (p < pressure_edges[i + 1])))
        pressure.append({"lower_Pa": float(pressure_edges[i]),
                         "upper_Pa": float(pressure_edges[i + 1]), **describe(mask)})
    phase = {name: describe(reference_phase == name)
             for name in ("gas", "liquid", "two-phase")}
    score = np.mean(uncertainty, axis=1)
    uncertainty_edges = np.unique(np.quantile(score, np.linspace(0, 1, 5)))
    uncertainty_bins = []
    for i in range(uncertainty_edges.size - 1):
        mask = ((score >= uncertainty_edges[i]) &
                ((score <= uncertainty_edges[i + 1]) if i == uncertainty_edges.size - 2
                 else (score < uncertainty_edges[i + 1])))
        uncertainty_bins.append({"lower": float(uncertainty_edges[i]),
                                 "upper": float(uncertainty_edges[i + 1]), **describe(mask)})
    return {"pressure_quintiles": pressure, "reference_phase": phase,
            "ensemble_uncertainty_quartiles": uncertainty_bins}


def qualify(results: Path = DEFAULT_RESULTS):
    results = Path(results)
    required = ("summary.json", "per_output_metrics.csv", "fold_metrics.csv",
                "predictions.csv", "inputs.mat", "mrst_results.mat")
    missing = [name for name in required if not (results / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Victory benchmark: missing {missing}")
    summary = json.loads((results / "summary.json").read_text())
    if not summary.get("mrst_included") or not summary.get("victory_phase_flags_from_reference"):
        raise ValueError("qualification requires MRST results and disclosed reference phase flags")
    inputs = loadmat(results / "inputs.mat", squeeze_me=True)
    mrst = loadmat(results / "mrst_results.mat", squeeze_me=True)
    input_digest = str(inputs.get("input_digest", "")).strip()
    mrst_digest = str(mrst.get("input_digest", "")).strip()
    if input_digest != summary["input_digest"] or mrst_digest != input_digest:
        raise ValueError("Victory/MRST qualification input digests do not match")

    pair_rows = _rows(results / "per_output_metrics.csv")
    index = _metric_index(pair_rows)
    fold_rows = _rows(results / "fold_metrics.csv")
    graph = EvidenceGraph()
    graph.add_node(EvidenceNode(
        "bundled_reference", EvidenceNodeKind.MANUFACTURED,
        "Victory bundled phase-labelled test dataset", metadata={
            "rows": summary["rows"], "input_digest": input_digest,
            "phase_flags_are_model_inputs": True,
        },
    ))
    graph.add_node(EvidenceNode("pymrst_physics", EvidenceNodeKind.PYMRST,
                                "PyMRST Peng-Robinson stability and flash"))
    graph.add_node(EvidenceNode("mrst_physics", EvidenceNodeKind.MRST,
                                "MRST Peng-Robinson stability and flash",
                                version="MRST 2024a"))
    graph.add_node(EvidenceNode("victory_ensemble", EvidenceNodeKind.PYMRST,
                                "Victory beta2.2 seven-fold ensemble",
                                metadata={"experimental": True, "folds": 7}))
    graph.add_node(EvidenceNode("physical_invariants", EvidenceNodeKind.INVARIANT,
                                "Flash material balance, closure, bounds and fugacity"))
    for fold in range(1, 8):
        graph.add_node(EvidenceNode(f"victory_fold_{fold}", EvidenceNodeKind.PYMRST,
                                    f"Victory beta2.2 fold {fold}",
                                    metadata={"experimental": True}))

    physics_edges = []
    for quantity, limit in PHYSICS_LIMITS.items():
        row = index[("pymrst", "mrst", quantity)]
        maximum = _float(row, "max_ae")
        edge = _edge(
            f"pymrst_mrst_{quantity.lower()}", "mrst_physics", "pymrst_physics",
            "physics_flash_parity", f"PyMRST and MRST {quantity} agree",
            accepted=maximum <= limit,
            metrics={"rmse": _float(row, "rmse"), "maximum_absolute_error": maximum,
                     "absolute_tolerance": limit, "count": int(row["count"])},
            shared=("Peng-Robinson formulation",),
        )
        graph.add_edge(edge); physics_edges.append(edge)

    surrogate_edges = []
    for quantity in OUTPUTS:
        row = index[("reference", "victory", quantity)]
        rmse, maximum = _float(row, "rmse"), _float(row, "max_ae")
        edge = _edge(
            f"victory_reference_{quantity.lower()}", "bundled_reference",
            "victory_ensemble", "surrogate_accuracy_gate",
            f"Victory ensemble {quantity} meets the declared production accuracy gate",
            accepted=(rmse <= SURROGATE_RMSE_LIMIT and maximum <= SURROGATE_MAX_LIMIT),
            metrics={"rmse": rmse, "maximum_absolute_error": maximum,
                     "rmse_tolerance": SURROGATE_RMSE_LIMIT,
                     "maximum_tolerance": SURROGATE_MAX_LIMIT,
                     "p95_absolute_error": _float(row, "p95_ae")},
            shared=("bundled scaler/test dataset",),
            exceptions=("phase class is supplied to Victory; stability is not tested",),
        )
        graph.add_edge(edge); surrogate_edges.append(edge)

    fold_mean_rmse = {}
    for fold in range(1, 8):
        selected = [r for r in fold_rows if int(r["fold"]) == fold]
        mean_rmse = float(np.mean([_float(r, "rmse") for r in selected]))
        fold_mean_rmse[fold] = mean_rmse
        graph.add_edge(_edge(
            f"fold_{fold}_characterization", "bundled_reference", f"victory_fold_{fold}",
            "surrogate_fold_characterization",
            f"Victory fold {fold} error is measured on the common held-out states",
            accepted=True, metrics={"mean_output_rmse": mean_rmse,
                                    "outputs": len(selected)},
            shared=("bundled scaler/test dataset",),
            exceptions=("characterization is not a production-accuracy claim",),
        ))

    physical = summary["physical_metrics"]
    safety_edges = []
    for name, node in (("pymrst", "pymrst_physics"), ("mrst", "mrst_physics"),
                       ("victory", "victory_ensemble")):
        values = physical[name]
        accepted = (
            values["nonfinite_count"] == 0 and values["out_of_bounds_count"] == 0
            and values["material_balance_rmse"] <= 1e-6
            and values["material_balance_max"] <= 1e-4
            and values["log_fugacity_rmse"] <= 1e-3
        )
        edge = _edge(
            f"{name}_physical_safety", "physical_invariants", node,
            "physical_flash_safety", f"{name} flash satisfies production physical gates",
            accepted=accepted, metrics={
                "nonfinite_count": values["nonfinite_count"],
                "out_of_bounds_count": values["out_of_bounds_count"],
                "material_balance_rmse": values["material_balance_rmse"],
                "material_balance_max": values["material_balance_max"],
                "log_fugacity_rmse": values["log_fugacity_rmse"],
            }, independence=Independence.INDEPENDENT,
        )
        graph.add_edge(edge); safety_edges.append(edge)

    py_conf = summary["phase_confusion_vs_reference"]["pymrst"]
    mrst_conf = summary["phase_confusion_vs_reference"]["mrst"]
    graph.add_edge(_edge(
        "pymrst_mrst_phase_decisions", "mrst_physics", "pymrst_physics",
        "phase_classification_parity", "PyMRST and MRST make identical phase decisions",
        accepted=py_conf == mrst_conf, metrics={"states": summary["rows"]},
        shared=("Peng-Robinson formulation",),
    ))
    speedup = summary["timings_seconds"]["pymrst"] / summary["timings_seconds"]["victory"]
    graph.add_edge(_edge(
        "victory_inference_speed", "pymrst_physics", "victory_ensemble",
        "performance", "Warm Victory ensemble inference is faster than physical flash",
        accepted=speedup > 1, metrics={"speedup": speedup,
                                      "pymrst_seconds": summary["timings_seconds"]["pymrst"],
                                      "victory_seconds": summary["timings_seconds"]["victory"]},
        independence=Independence.SHARED_KERNEL,
        shared=("same hardware and state batch",),
        exceptions=("Victory timing excludes stability classification",),
    ))
    correlations = [v for v in summary["victory_uncertainty_error_correlation"].values()
                    if v is not None]
    graph.add_edge(_edge(
        "victory_uncertainty_signal", "bundled_reference", "victory_ensemble",
        "uncertainty_characterization",
        "Ensemble spread has a positive error-ranking signal for every output",
        accepted=bool(correlations and min(correlations) > 0),
        metrics={"minimum_correlation": min(correlations),
                 "median_correlation": float(np.median(correlations))},
        shared=("bundled scaler/test dataset",),
    ))

    atlas = _error_atlas(results / "predictions.csv")
    result = {
        "case": "victory_beta2.2_10000_conditional_flash",
        "qualification_completed": True,
        "physics_parity_accepted": all(edge.accepted for edge in physics_edges),
        "surrogate_accuracy_accepted": all(edge.accepted for edge in surrogate_edges),
        "production_surrogate_accepted": all(edge.accepted for edge in surrogate_edges)
                                         and safety_edges[-1].accepted,
        "stability_classifier_qualified": False,
        "reason_stability_not_qualified": "Victory receives reference phase flags as inputs",
        "accepted_edges": sum(edge.accepted for edge in graph.edges),
        "total_edges": len(graph.edges),
        "timings_seconds": summary["timings_seconds"], "speedup": speedup,
        "physical_metrics": physical, "fold_mean_rmse": fold_mean_rmse,
        "error_atlas": atlas, "input_digest": input_digest,
        "source_artifacts": {name: _sha256(results / name) for name in required},
    }
    return graph, result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    graph, summary = qualify(args.results)
    payload = {"artifact": "pymrst_victory_surrogate_evidence_graph",
               "artifact_schema_version": 1, "graph": graph.to_dict(),
               "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE: {summary['accepted_edges']}/{summary['total_edges']} edges accepted")
    print(f"physics parity: {summary['physics_parity_accepted']}")
    print(f"production surrogate: {summary['production_surrogate_accepted']}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
