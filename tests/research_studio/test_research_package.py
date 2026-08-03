"""M3.7 complete, verified and reproducible package directories."""

import json
import subprocess

import pytest

from pymrst.research_studio.experiments import (
    InvalidResearchPackage, PackageArtifact, ResearchPackageDefinition,
    generate_research_package, verify_research_package,
)


def _definition(tmp_path):
    case = tmp_path / "case.json"
    case.write_text('{"case":"spe1"}\n', encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"passed":true}\n', encoding="utf-8")
    return ResearchPackageDefinition(
        "SPE1 branch study", "experiment_abc", ("member_a", "member_b"),
        ("python3", "-c", "print('reproduced')"),
        artifacts=(PackageArtifact(str(case), "cases"),
                   PackageArtifact(str(evidence), "evidence")),
        metadata={"comparison_group": "control-study"})


def test_package_has_complete_layout_inventory_and_runnable_recipe(tmp_path):
    definition = _definition(tmp_path)
    root = generate_research_package(definition, tmp_path / "package")
    manifest = verify_research_package(root)
    paths = {item["path"] for item in manifest["inventory"]}
    assert {"README.md", "definition.json", "environment.json", "reproduce.sh",
            "cases/case.json", "evidence/evidence.json"} <= paths
    completed = subprocess.run([str(root / "reproduce.sh")], cwd=root,
                               text=True, capture_output=True, check=True)
    assert completed.stdout.strip() == "reproduced"


def test_package_identity_is_content_addressed_and_reproducible(tmp_path):
    definition = _definition(tmp_path)
    first = generate_research_package(definition, tmp_path / "one")
    second = generate_research_package(definition, tmp_path / "two")
    a = json.loads((first / "package_manifest.json").read_text())
    b = json.loads((second / "package_manifest.json").read_text())
    assert a["package_id"] == b["package_id"]
    # Platform details may vary over time; every scientific/package input is stable.
    assert definition.package_id == a["package_id"]


def test_tampering_is_detected(tmp_path):
    root = generate_research_package(_definition(tmp_path), tmp_path / "package")
    (root / "cases" / "case.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(InvalidResearchPackage, match="content changed"):
        verify_research_package(root)


def test_unknown_artifact_kind_and_nonempty_destination_are_refused(tmp_path):
    source = tmp_path / "x"
    source.write_text("x")
    with pytest.raises(ValueError, match="artifact kind"):
        PackageArtifact(str(source), "misc")
    destination = tmp_path / "nonempty"
    destination.mkdir()
    (destination / "existing").write_text("keep")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_research_package(_definition(tmp_path), destination)
