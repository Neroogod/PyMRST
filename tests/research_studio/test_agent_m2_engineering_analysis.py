"""End-to-end deterministic engineering checks and specialist findings."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.research_studio.agents import (
    AgentPermission,
    DeterministicOrchestrator,
    InvestigationContext,
    InvestigationPlan,
    InvestigationStatus,
    PermissionGrant,
    ToolCall,
)
from pymrst.research_studio.agents.specialists.grid_quality import GridQualitySpecialist
from pymrst.research_studio.agents.specialists.initialization import InitializationSpecialist
from pymrst.research_studio.agents.specialists.rock_fluid import RockFluidSpecialist
from pymrst.research_studio.agents.specialists.wells_schedule import WellsScheduleSpecialist
from pymrst.research_studio.agents.specialists.conservation import (
    ConservationSpecialist, InvariantSpecialist,
)
from pymrst.research_studio.agents.tools.case_tools import (
    CaseAccessError,
    CaseAnalysisCatalog,
    GridQualityTool,
    InitializationQualityTool,
    RockFluidQualityTool,
    WellsScheduleQualityTool,
)
from pymrst.modules.ad_core.utils.schedule import Control, Schedule
from pymrst.research_studio.agents.tools.validation_tools import (
    ConservationTool, InvariantTool, ValidationDataCatalog,
)


def _grid():
    return compute_geometry(cart_grid([2, 1, 1], [20.0, 10.0, 5.0]))


def _catalog(*, invalid=False):
    grid = _grid()
    rock = SimpleNamespace(
        poro=np.array([0.2, 1.2 if invalid else 0.25]),
        perm=np.array([[100.0], [-10.0 if invalid else 50.0]]),
    )
    wells = [
        {
            "name": "P1", "cells": [0], "type": "bhp", "val": 1.0e7,
            "status": True, "cstatus": [True], "sign": -1,
        },
        {
            "name": "P1" if invalid else "I1",
            "cells": [5] if invalid else [1], "type": "rate", "val": 100.0,
            "status": True, "cstatus": [False] if invalid else [True], "sign": 1,
        },
    ]
    schedule = Schedule(
        step_val=np.array([10.0, -1.0 if invalid else 20.0]),
        step_control=np.array([0, 0]), controls=[Control(W=wells)],
    )
    catalog = CaseAnalysisCatalog()
    catalog.register(
        "case_analysis", grid=grid, rock=rock,
        fluid_tables={
            "SWOF": np.array([
                [0.1, 0.0, 1.0, 0.0],
                [0.05 if invalid else 0.9, 1.2 if invalid else 1.0, 0.0, 0.0],
            ])
        },
        regions={"SATNUM": np.array([1, 3 if invalid else 1])},
        initial_state={
            "pressure": np.array([2.0e7, -1.0 if invalid else 2.1e7]),
            "sW": np.array([0.2, 0.3]),
            "sO": np.array([0.8, 0.9 if invalid else 0.7]),
        },
        wells=wells, schedule=schedule,
    )
    return catalog


def _context(tool_names):
    case = _catalog().get("case_analysis")
    return InvestigationContext(
        "investigation_quality", "Qualify model inputs", "case_analysis",
        case.content_hash,
        PermissionGrant(AgentPermission.ANALYZE, allowed_tools=tuple(tool_names)),
    )


def test_valid_case_passes_all_four_deterministic_checks():
    catalog = _catalog()
    tools = (
        GridQualityTool(catalog), RockFluidQualityTool(catalog),
        InitializationQualityTool(catalog), WellsScheduleQualityTool(catalog),
    )
    names = tuple(tool.descriptor.name for tool in tools)
    context = InvestigationContext(
        "investigation_valid", "Qualify valid case", "case_analysis",
        catalog.get("case_analysis").content_hash,
        PermissionGrant(AgentPermission.ANALYZE, allowed_tools=names),
    )
    plan = InvestigationPlan(tuple(
        ToolCall(f"call_{index}", name, {"case_id": "case_analysis"})
        for index, name in enumerate(names)
    ))
    record = DeterministicOrchestrator(tools).execute(context, plan)
    assert record.status is InvestigationStatus.COMPLETED
    results = [entry.payload["value"] for entry in record.entries
               if entry.kind.value == "tool_result"]
    assert all(result["ok"] for result in results)


def test_invalid_case_produces_specific_typed_specialist_findings():
    catalog = _catalog(invalid=True)
    pairs = (
        (GridQualityTool(catalog), GridQualitySpecialist()),
        (RockFluidQualityTool(catalog), RockFluidSpecialist()),
        (InitializationQualityTool(catalog), InitializationSpecialist()),
        (WellsScheduleQualityTool(catalog), WellsScheduleSpecialist()),
    )
    statements = []
    categories = set()
    for index, (tool, specialist) in enumerate(pairs):
        result = tool.invoke(ToolCall(
            f"call_{index}", tool.descriptor.name, {"case_id": "case_analysis"},
        ))
        assert not result.value["ok"]
        findings = specialist.analyze(result)
        assert findings
        assert all(item.evidence[0].checksum == catalog.get("case_analysis").content_hash
                   for item in findings)
        assert all(item.confidence == 1.0 for item in findings)
        statements.extend(item.statement for item in findings)
        categories.update(item.category for item in findings)
    assert any("porosity outside" in statement for statement in statements)
    assert any("first column is not strictly monotone" in statement
               for statement in statements)
    assert any("pressure is non-finite or non-positive" in statement
               for statement in statements)
    assert any("duplicate well names" in statement for statement in statements)
    assert {"grid.rock", "fluid_table", "initialization", "wells", "schedule"} \
        <= categories


def test_analysis_registration_snapshots_arrays_against_later_mutation():
    grid = _grid()
    pressure = np.array([2.0e7, 2.1e7])
    table = np.array([[0.1, 0.0], [0.9, 1.0]])
    catalog = CaseAnalysisCatalog()
    case = catalog.register(
        "case_snapshot", grid=grid, fluid_tables={"SWFN": table},
        initial_state={"pressure": pressure},
    )
    pressure[0] = -1.0
    table[0, 0] = 2.0
    assert case.initial_state["pressure"][0] == 2.0e7
    assert case.fluid_tables["SWFN"][0][0, 0] == 0.1
    assert not case.initial_state["pressure"].flags.writeable


def test_analysis_catalog_detects_mutation_of_live_grid_objects():
    grid = _grid()
    catalog = CaseAnalysisCatalog()
    catalog.register("case_guarded", grid=grid)
    grid.cells.volumes[0] *= 2.0
    import pytest
    with pytest.raises(CaseAccessError, match="changed after registration"):
        catalog.get("case_guarded")


def test_conservation_tool_reports_component_specific_imbalance():
    tool = ConservationTool()
    result = tool.invoke(ToolCall("balance", "analyze_conservation", {
        "component_names": ["water", "oil"], "units": "surface_m3",
        "initial_inventory": [[10.0, 20.0], [5.0, 5.0]],
        "final_inventory": [[11.0, 20.0], [5.0, 5.0]],
        "well_rates": [0.0, 0.0], "dt": 1.0,
        "normalized_tolerance": 1e-8,
    }))
    assert result.value["passed"] is False
    assert result.value["maximum_normalized_imbalance"] > 0
    findings = ConservationSpecialist().analyze(result)
    assert len(findings) == 1
    assert findings[0].affected_objects == ("component:water",)


def test_conservation_tool_accepts_a_closed_component_balance():
    result = ConservationTool().invoke(ToolCall("closed", "analyze_conservation", {
        "component_names": ["water"],
        "initial_inventory": [[10.0], [5.0]],
        "final_inventory": [[11.0], [5.0]],
        "well_rates": [0.5], "dt": 2.0,
        "normalized_tolerance": 1e-12,
    }))
    assert result.value["passed"] is True
    assert ConservationSpecialist().analyze(result) == ()


def test_invariant_tool_localizes_nonfinite_and_out_of_bounds_values():
    catalog = ValidationDataCatalog()
    dataset = catalog.register(
        "state_57", {"pressure": [2.0e7, float("nan")], "sW": [0.2, 1.1]},
    )
    result = InvariantTool(catalog).invoke(ToolCall("invariants", "analyze_invariants", {
        "dataset_id": "state_57",
        "checks": [
            {"kind": "finite", "field": "pressure"},
            {"kind": "bounded", "field": "sW", "lower": 0.0, "upper": 1.0},
        ],
    }))
    assert result.value["passed"] is False
    findings = InvariantSpecialist().analyze(result)
    assert len(findings) == 2
    assert {item.category for item in findings} == {"invariant"}
    assert all(item.affected_objects == ("index:1",) for item in findings)
    assert findings[0].evidence[0].checksum == dataset.checksum
