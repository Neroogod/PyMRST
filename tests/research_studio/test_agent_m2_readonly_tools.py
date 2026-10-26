"""Agent M2 read-only tool acceptance and security tests."""

from pathlib import Path

import numpy as np
import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    DeterministicOrchestrator,
    InvestigationContext,
    InvestigationPlan,
    InvestigationStatus,
    PermissionGrant,
    ToolCall,
)
from pymrst.research_studio.agents.tools import ToolRegistry
from pymrst.research_studio.agents.tools.case_tools import (
    CaseAccessError,
    CaseCatalog,
    InspectCaseTool,
)
from pymrst.research_studio.agents.tools.deck_tools import (
    DeckAccessError,
    DeckCatalog,
    GetKeywordTool,
    InspectDeckTool,
)
from pymrst.research_studio.agents.tools.result_tools import (
    GetResultArrayTool,
    InspectResultTool,
)
from pymrst.research_studio.agents.tools.validation_tools import (
    CompareSeriesTool,
    ComparisonLimitError,
)
from pymrst.research_studio.case.definition import CaseComponent, ReservoirCase
from pymrst.research_studio.experiments import (
    MemberPayload,
    MemberResultStore,
    VariantCoordinate,
)
from pymrst.research_studio.studio import ResultCatalog


DIGEST = "0" * 64


def _context(permission, tools):
    return InvestigationContext(
        "investigation_m2", "Inspect existing engineering evidence",
        "case_test", DIGEST,
        PermissionGrant(permission, allowed_tools=tuple(tools)),
    )


def test_deck_catalog_tracks_include_lineage_keywords_and_checksums(tmp_path):
    include = tmp_path / "PROPS.INC"
    include.write_text("PORO\n 0.2 /\nPERMX\n 100 /\n", encoding="utf-8")
    deck = tmp_path / "CASE.DATA"
    deck.write_text(
        "RUNSPEC\nDIMENS\n 1 1 1 /\nGRID\nINCLUDE\n 'PROPS.INC' /\n"
        "SCHEDULE\nTSTEP\n 10 /\nEND\n",
        encoding="utf-8",
    )
    catalog = DeckCatalog((tmp_path,))
    catalog.register("deck_test", deck)
    summary = catalog.summary("deck_test")
    assert summary["file_count"] == 2
    assert summary["keyword_counts"]["PORO"] == 1
    assert set(summary["sections"]) >= {"RUNSPEC", "GRID", "SCHEDULE"}
    poro = catalog.keyword("deck_test", "PORO")
    assert poro["occurrences"][0]["source"] == str(include.resolve())
    assert poro["occurrences"][0]["line"] == 1
    assert len(poro["occurrences"][0]["checksum"]) == 64

    tools = (InspectDeckTool(catalog), GetKeywordTool(catalog))
    record = DeterministicOrchestrator(tools).execute(
        _context(AgentPermission.INSPECT, ("inspect_deck", "get_keyword")),
        InvestigationPlan((
            ToolCall("call_deck", "inspect_deck", {"deck_id": "deck_test"}),
            ToolCall("call_poro", "get_keyword", {
                "deck_id": "deck_test", "keyword": "PORO",
            }),
        )),
    )
    assert record.status is InvestigationStatus.COMPLETED


def test_deck_catalog_refuses_escape_cycles_and_unbounded_occurrences(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "OUT.INC"
    outside.write_text("PORO\n1 /\n")
    escaping = root / "ESCAPE.DATA"
    escaping.write_text("GRID\nINCLUDE\n '../OUT.INC' /\n")
    with pytest.raises(DeckAccessError, match="outside configured roots"):
        DeckCatalog((root,)).register("escape", escaping)

    first, second = root / "A.DATA", root / "B.INC"
    first.write_text("GRID\nINCLUDE\n 'B.INC' /\n")
    second.write_text("INCLUDE\n 'A.DATA' /\n")
    with pytest.raises(DeckAccessError, match="cycle"):
        DeckCatalog((root,)).register("cycle", first)

    repeated = root / "REPEAT.DATA"
    repeated.write_text("GRID\nPORO\n1 /\nPORO\n2 /\n")
    catalog = DeckCatalog((root,))
    catalog.register("repeat", repeated)
    with pytest.raises(DeckAccessError, match="limit is 1"):
        catalog.keyword("repeat", "PORO", maximum_occurrences=1)


def _case():
    return ReservoirCase(
        "Read-only case",
        CaseComponent("does.not.exist:builder", {"unused": True}),
        grid={"kind": "cartesian", "cartDims": [1, 1, 1]},
        properties={"poro": [0.2]}, physics={"phases": ["water"]},
        discretization={"method": "tpfa"}, facilities={},
        schedule={"steps": [10.0]}, solver={"name": "newton"},
    )


def test_case_tool_inspects_definition_without_resolving_builder(tmp_path):
    catalog = CaseCatalog((tmp_path,))
    catalog.register("case_test", _case())
    summary = catalog.summary("case_test")
    assert summary["builder"] == "does.not.exist:builder"
    assert summary["sections"]["grid"]["configured"]
    assert len(summary["content_hash"]) == 64
    # If inspection called construct(), the intentionally nonexistent module
    # above would raise. Reaching this assertion pins the read-only boundary.
    result = InspectCaseTool(catalog).invoke(
        ToolCall("call_case", "inspect_case", {"case_id": "case_test"})
    )
    assert result.ok
    with pytest.raises(CaseAccessError, match="outside configured roots"):
        CaseCatalog((tmp_path / "allowed",)).register_file(
            "outside", tmp_path / "case.json",
        )


def _member_store(tmp_path: Path):
    coordinate = VariantCoordinate(
        "experiment_test", "member_1", 0, "base", "default", {}, {},
    )
    store = MemberResultStore(tmp_path / "member")
    store.begin(coordinate, 42, "test:runner")
    store.commit(
        MemberPayload(
            times=np.array([1.0, 2.0]),
            fields={"pressure": np.array([[100.0, 101.0], [90.0, 92.0]])},
            convergence={"newton_iterations": np.array([[2], [3]])},
        ),
        coordinate=coordinate, seed=42, runner="test:runner", elapsed_seconds=0.1,
    )
    return store.path


def test_result_tools_delegate_to_bounded_lazy_catalog(tmp_path):
    store = _member_store(tmp_path)
    catalog = ResultCatalog((tmp_path,), maximum_json_values=2)
    catalog.register("result_1", "Synthetic result", store)
    metadata = InspectResultTool(catalog).invoke(
        ToolCall("meta", "inspect_result", {"result_id": "result_1"})
    )
    assert metadata.value["categories"]["fields"] == ("pressure",)
    values = GetResultArrayTool(catalog).invoke(ToolCall(
        "array", "get_result_array",
        {"result_id": "result_1", "category": "fields", "name": "pressure",
         "time_index": 1, "start": 0, "stop": 2},
    ))
    assert values.value["values"] == (90.0, 92.0)
    with pytest.raises(ValueError, match="limit"):
        GetResultArrayTool(catalog).invoke(ToolCall(
            "too_many", "get_result_array",
            {"result_id": "result_1", "category": "fields", "name": "pressure"},
        ))


def _comparison_call(call_id="compare"):
    return ToolCall(call_id, "compare_series", {
        "reference": {
            "times": [1.0, 2.0], "values": [[10.0], [20.0]],
            "entity_names": ["P1"], "unit": "Pa",
        },
        "candidate": {
            "times": [1.0, 2.0], "values": [[10.01], [19.99]],
            "entity_names": ["P1"], "unit": "Pa",
        },
        "tolerance": {"absolute": 0.02, "relative": 0.0, "scale": 1.0},
    })


def test_validation_tool_is_permission_gated_and_returns_normalized_metrics():
    tool = CompareSeriesTool(maximum_values=10)
    denied = DeterministicOrchestrator((tool,)).execute(
        _context(AgentPermission.INSPECT, ("compare_series",)),
        InvestigationPlan((_comparison_call(),)),
    )
    assert denied.status is InvestigationStatus.FAILED
    complete = DeterministicOrchestrator((tool,)).execute(
        _context(AgentPermission.ANALYZE, ("compare_series",)),
        InvestigationPlan((_comparison_call(),)),
    )
    assert complete.status is InvestigationStatus.COMPLETED
    metrics = complete.entries[-1].payload["value"]
    assert metrics["passed"] is True
    assert metrics["worst_entity"] == "P1"
    with pytest.raises(ComparisonLimitError, match="limit"):
        CompareSeriesTool(maximum_values=1).invoke(_comparison_call("large"))


def test_tool_registry_is_explicit_sorted_and_duplicate_safe(tmp_path):
    catalog = DeckCatalog((tmp_path,))
    registry = ToolRegistry((GetKeywordTool(catalog), InspectDeckTool(catalog)))
    assert tuple(item.name for item in registry.descriptors()) == (
        "get_keyword", "inspect_deck",
    )
    with pytest.raises(KeyError, match="already registered"):
        registry.register(InspectDeckTool(catalog))
