#!/usr/bin/env python3
"""Run the local PyMRST Research Studio workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from pymrst.research_studio.studio import (
    BenchmarkEntry, BenchmarkPortal, ResearchStudioHTTPServer,
    ResearchStudioService, ResultCatalog, StudioMetadataStore,
    TeachingRuntime, Workflow,
)


def demonstration(context, parameters):
    steps = int(parameters.get("steps", 6))
    pressure = float(parameters.get("initial_pressure_bar", 270.0))
    decline = float(parameters.get("decline_bar_per_step", 1.7))
    history = []
    for index in range(steps):
        context.cancellation.raise_if_requested()
        time.sleep(.08)
        pressure -= decline
        history.append(pressure)
        context.emit("report_step", (index+1)/steps,
                     f"Accepted report step {index+1} of {steps}",
                     {"pressure_bar": pressure, "iterations": 3+(index % 2)})
    return {"field_pressure_bar": history, "report_steps": steps,
            "evidence": {"mass_balance": "within tolerance"}}


def build_service(root: Path):
    metadata = StudioMetadataStore(root / "results/research_studio/studio/metadata.sqlite")
    service = ResearchStudioService(max_workers=2, metadata_store=metadata)
    service.register_workflow(Workflow(
        "teaching-waterflood", "Teaching waterflood",
        "Bounded report-step workflow for workspace verification.",
        demonstration, teaching_safe=True))
    project = service.create_project(
        "SPE1 development study",
        description="Branches, evidence and qualification runs",
        tags=("black-oil", "internal"))
    service.submit(project.project_id, "teaching-waterflood",
                   {"steps": 6, "initial_pressure_bar": 270.0})
    entries = []
    for benchmark_id, title, relative in (
        ("m3", "Branch and package qualification",
         "docs/research_studio/experiments/m3_branch_package_qualification.json"),
        ("m4", "Hybrid platform qualification",
         "docs/research_studio/experiments/m4_qualification.json")):
        path = root / relative
        if path.exists():
            entries.append(BenchmarkEntry.from_path(
                benchmark_id, title, "1", path,
                publication_status="internal",
                summary="Internal qualification evidence"))
    return service, BenchmarkPortal(entries)


def build_result_catalog(root: Path):
    """Discover one completed SPE1 store and attach the real 3-D grid viewer."""
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
        setup_spe1_tutorial,
    )
    from pymrst.plotting.extensions.browser import plot_reservoir_browser
    from pymrst.research_studio.experiments import MemberResultStore

    catalog = ResultCatalog((root / "results",))
    candidates = sorted((root / "results/research_studio").glob(
        "spe1_ensemble_final/**/member.json"))
    if not candidates:
        return catalog
    store = candidates[0].parent
    result = MemberResultStore(store).lazy()
    _, model, _ = setup_spe1_tutorial()
    fields = {name: result.array("fields", name)
              for name in result.names("fields")}
    well_series = {name: result.array("wells", name)
                   for name in result.names("wells")}
    viewer = root / "results/research_studio/studio/spe1_result.html"
    viewer.parent.mkdir(parents=True, exist_ok=True)
    plot_reservoir_browser(
        model.G, fields, wells=model.wells, times=result.times,
        filename=viewer, title="SPE1 time-integration result",
        well_series=well_series, rock=model.rock, open_browser=False)
    catalog.register("spe1-time-integration", "SPE1 time-integration result",
                     store, viewer_path=viewer)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    service, portal = build_service(root)
    results = build_result_catalog(root)
    server = ResearchStudioHTTPServer(
        (arguments.host, arguments.port), service,
        teaching=TeachingRuntime(), portal=portal, results=results)
    print(f"PyMRST Research Studio: http://{arguments.host}:{server.server_port}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close(); service.shutdown()


if __name__ == "__main__":
    main()
