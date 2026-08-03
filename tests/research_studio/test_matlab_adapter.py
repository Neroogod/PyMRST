"""M2.5 MATLAB/MRST adapter contracts and stale-fixture refusal."""

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from pymrst.research_studio.validation.matlab import (
    MatlabFixtureManifest, MatlabFixtureSpec, StaleMatlabFixtureError,
    generation_command, result_mappings, sha256_file, validate_fixture,
    write_manifest,
)
from scripts.qualify_blackoil_time_integration import (
    SPE1_FIXTURE_SPEC, spe1_reference_adapter,
)


ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path):
    reference = tmp_path / "reference.mat"
    generator = tmp_path / "generate_case.m"
    mrst = tmp_path / "mrst-2024a"
    reference.write_bytes(b"MATLAB fixture")
    generator.write_text("function generate_case(); end\n", encoding="utf-8")
    mrst.mkdir()
    deck = tmp_path / "CASE.DATA"
    deck.write_text("RUNSPEC\n", encoding="utf-8")
    spec = MatlabFixtureSpec(
        fixture_id="case-v1", case_name="case", reference_path=reference,
        generator_path=generator, generator_function="generate_case",
        mrst_root=mrst, modules=("ad-core",), mappings=("cell:v1",),
        tolerances={"pressure": {"absolute": 1.0}},
        input_artifacts={"deck": deck},
        solver_settings={"maximum_iterations": 10},
    )
    manifest = MatlabFixtureManifest(
        fixture_id=spec.fixture_id, case_name=spec.case_name,
        matlab_executable="/Applications/MATLAB.app/bin/matlab",
        matlab_release="2021b", mrst_root=str(spec.mrst_root),
        mrst_version="MRST 2024a", modules=spec.modules,
        generator_function=spec.generator_function,
        generator_sha256=sha256_file(generator),
        reference_sha256=sha256_file(reference), reference_bytes=reference.stat().st_size,
        mappings=spec.mappings, tolerances=spec.tolerances,
        input_artifacts={
            "deck": {"path": str(deck), "sha256": sha256_file(deck),
                     "bytes": deck.stat().st_size},
        },
        solver_settings=spec.solver_settings,
        regeneration_command=("matlab", "-batch", "generate_case"),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_manifest(spec.manifest_path, manifest)
    return spec, manifest


def test_fixture_manifest_round_trip_and_exact_validation(tmp_path):
    spec, expected = _fixture(tmp_path)
    actual = validate_fixture(spec)
    assert actual == expected
    payload = json.loads(spec.manifest_path.read_text())
    assert payload["reference_sha256"] == sha256_file(spec.reference_path)


@pytest.mark.parametrize("target", ["reference", "generator", "input"])
def test_fixture_validation_refuses_changed_inputs(tmp_path, target):
    spec, _ = _fixture(tmp_path)
    path = ({"reference": spec.reference_path, "generator": spec.generator_path,
             "input": spec.input_artifacts["deck"]})[target]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StaleMatlabFixtureError, match="provenance mismatch"):
        validate_fixture(spec)


def test_generation_command_is_one_batch_argument_and_quotes_paths(tmp_path):
    spec, _ = _fixture(tmp_path)
    executable = tmp_path / "MATLAB executable"
    command = generation_command(spec, executable)
    assert command[:2] == (str(executable), "-batch")
    assert len(command) == 3
    assert "MATLAB_RELEASE=" in command[2]
    assert spec.generator_function in command[2]


def test_named_result_mappings_reorder_wells_without_guessing():
    mappings = result_mappings(
        report_times=(1.0, 2.0), candidate_times=(1.0, 2.0),
        reference_cells=(0, 1), candidate_cells=(0, 1),
        reference_wells=("injector", "producer"),
        candidate_wells=("producer", "injector"),
        reference_phases=("water", "gas"), candidate_phases=("water", "gas"),
    )
    assert mappings.mappings["well"].candidate_indices.tolist() == [1, 0]
    assert mappings.tokens == ("cell:v1", "phase:v1", "report_time:v1", "well:v1")


def test_spe1_fixture_is_manifested_and_loadable():
    manifest = spe1_reference_adapter().validate()
    assert manifest.fixture_id == SPE1_FIXTURE_SPEC.fixture_id
    assert manifest.reference_sha256 == sha256_file(SPE1_FIXTURE_SPEC.reference_path)
    assert manifest.generator_sha256 == sha256_file(SPE1_FIXTURE_SPEC.generator_path)
    assert manifest.modules == SPE1_FIXTURE_SPEC.modules
    assert manifest.mappings == SPE1_FIXTURE_SPEC.mappings
