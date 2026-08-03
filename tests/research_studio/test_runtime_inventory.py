"""The Foundation 0.1 inventory must keep describing the code.

`runtime_entrypoints.csv` is the only accepted work package in the programme,
and it is a document. Documents drift: a file gets renamed, a function moves
module, and the inventory quietly becomes a description of a codebase that no
longer exists. Every later milestone reasons from it - which entry points are
canonical, which are specialized, which must never be forced through a
forward-only session kernel - so its decay is not cosmetic.

These tests keep it honest by resolving every row against the tree. They are
deliberately cheap: no simulation runs here, only imports and attribute lookups.

They also guard the covenant in section 2.1 that Research Studio stays additive.
If an entry point ever imports `research_studio` at module scope, the replica
layer has acquired a dependency on the extension, and that is the one thing the
architecture forbids outright.
"""

import csv
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "research_studio" / "runtime_entrypoints.csv"

CLASSIFICATIONS = {
    "direct runtime delegate",
    "canonical numerical workflow",
    "specialized session adapter",
    "specialized process adapter",
    "high-level example wrapper",
    "deprecated duplicate",
    "intentionally specialized workflow",
}


def _rows():
    with INVENTORY.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _module_name(owner: str) -> str:
    return owner.removesuffix(".py").replace("/", ".")


def test_the_inventory_is_not_empty_and_has_the_expected_columns():
    rows = _rows()
    assert rows, "the inventory is empty"
    assert set(rows[0]) == {
        "entrypoint", "owner", "classification", "current_return_contract",
        "integration_strategy",
    }


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["entrypoint"])
def test_every_inventoried_entry_point_still_exists(row):
    """The owner file must exist and expose the symbol the row names.

    A row that no longer resolves is worse than a missing row: it asserts a
    classification for something that is not there.
    """
    owner = ROOT / row["owner"]
    assert owner.exists(), f"{row['owner']} no longer exists"

    module = importlib.import_module(_module_name(row["owner"]))
    # "NonLinearSolver.simulate_schedule" and "compositional._shared.run_schedule"
    # both end in the symbol; only the last component is defined in the module,
    # unless it is a method, in which case its class is.
    parts = row["entrypoint"].split(".")
    for candidate in (parts[-1], parts[-2] if len(parts) > 1 else None):
        if candidate and hasattr(module, candidate):
            owner_object = getattr(module, candidate)
            if candidate != parts[-1]:
                assert hasattr(owner_object, parts[-1]), (
                    f"{candidate} exists but has no {parts[-1]}")
            return
    pytest.fail(f"{row['entrypoint']} not found in {row['owner']}")


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["entrypoint"])
def test_every_classification_is_one_the_plan_defines(row):
    assert row["classification"] in CLASSIFICATIONS, (
        f"{row['entrypoint']} is classified {row['classification']!r}, which is "
        f"not one of the categories section 5.1 defines"
    )


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["entrypoint"])
def test_no_entry_point_imports_research_studio_at_module_scope(row):
    """Section 2.1: the replica layer must not depend on the extension.

    Import-time coupling is the failure that matters. `NonLinearSolver` takes an
    `observer` callable and never imports the studio to do it, which is what
    makes the hook additive rather than a dependency.
    """
    import ast

    source = (ROOT / row["owner"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = []
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if "research_studio" in a.name]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if "research_studio" in node.module:
                offenders.append(node.module)
    assert not offenders, (
        f"{row['owner']} imports {offenders} at module scope; the replica layer "
        f"must not depend on Research Studio"
    )


def test_the_canonical_owner_is_named_exactly_once():
    """Two canonical numerical workflows would make "canonical" meaningless."""
    canonical = [r["entrypoint"] for r in _rows()
                 if r["classification"] == "canonical numerical workflow"]
    assert canonical == ["NonLinearSolver.simulate_schedule"], canonical
