#!/usr/bin/env python3
"""Build the versioned Research Studio experiment registry artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from pymrst.research_studio.experiments import (
    CaseReference, DatasetReference, ExecutionProfile, ExperimentDefinition,
    ExperimentRegistry, OutputRequest, OwnerReference, ValidationRequirement,
    VariantDefinition, explicit_variant_coordinates,
)


ROOT = Path(__file__).resolve().parents[1]
SPE1_DECK = ROOT.parent / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
DEFAULT_OUTPUT = ROOT / "docs/research_studio/experiments/registry.json"


def spe1_time_integration_experiment() -> ExperimentDefinition:
    dataset = DatasetReference.from_path(
        "spe1_benchmark_deck", SPE1_DECK, media_type="application/x-eclipse-deck")
    return ExperimentDefinition(
        name="SPE1 time-integration formulation study",
        hypothesis=("Adaptive implicit transport preserves fully implicit pressure "
                    "and production accuracy while reducing implicit transport work."),
        description=("A reproducible two-report-step FIM, explicit and AIM comparison "
                     "anchored to the signed MRST 2024a qualification fixture."),
        base_case=CaseReference(
            "case_spe1_blackoil",
            OwnerReference(
                "pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example:" 
                "run_blackoil_time_integration_example",
                {"case": "spe1", "report_steps": 2}),
            dataset.sha256,
        ),
        model=OwnerReference(
            "pymrst.modules.ad_blackoil.models:GenericBlackOilModelAD",
            {"disgas": True, "vapoil": False}),
        schedule=OwnerReference(
            "pymrst.modules.ad_core.initialization:setupSPE1",
            {"report_steps": 2, "controls": "deck"}),
        variants=(
            VariantDefinition("fim", components={"time_integration": "fully-implicit"},
                              comparison_group="time_integration"),
            VariantDefinition("explicit", components={"time_integration": "explicit"},
                              comparison_group="time_integration"),
            VariantDefinition("aim", parameters={"aim.threshold": 0.05},
                              components={"time_integration": "adaptive-implicit"},
                              comparison_group="time_integration"),
        ),
        datasets=(dataset,),
        execution_profile=ExecutionProfile(
            "quick", {"workers": 1, "checkpoint_every_report": True,
                      "failure_isolation": "member"}),
        validation=(
            ValidationRequirement("mrst_time_integration", {
                "fixture": "examples/validation/ref_time_integration_spe1.mat",
                "pressure_absolute_tolerance_pa": 100.0,
                "saturation_absolute_tolerance": 1e-6,
            }),
        ),
        outputs=(
            OutputRequest("reservoir", ("pressure", "sW", "sG", "rs")),
            OutputRequest("wells", ("bhp", "qWs", "qOs", "qGs")),
            OutputRequest("solver", ("iterations", "ministeps", "implicit_fraction")),
        ),
        publication_artifacts=("evidence_graph", "comparison_report"),
        random_seed=20240803,
        tags=("blackoil", "spe1", "time-integration", "mrst-qualified"),
    )


def build_registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    registry.register(spe1_time_integration_experiment())
    return registry


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    registry = build_registry()
    registry.save(args.output)
    definition = registry.get(registry.experiment_ids[0])
    members = explicit_variant_coordinates(definition)
    print(definition.experiment_id)
    print(definition.execution_fingerprint)
    print(f"{len(members)} deterministic members")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
