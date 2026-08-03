"""M3.9 portable case schema, construction and exact replay."""

import json

import numpy as np
import pytest

from pymrst.research_studio.case import (
    CaseComponent, ReservoirCase, read_reservoir_case, write_reservoir_case,
)


def _spe1_case():
    return ReservoirCase(
        name="SPE1 two-step FIM",
        builder=CaseComponent(
            "pymrst.modules.ad_blackoil.examples.blackoil_time_integration_example:_run_spe1_variant",
            {"mode": "fully-implicit", "nsteps": 2}),
        grid={"kind": "SPE1 deck"},
        properties={"source": "BENCH_SPE1.DATA"},
        physics={"model": "black-oil", "phases": ["water", "oil", "gas"],
                 "disgas": True, "vapoil": False},
        discretization={"flow": "tpfa", "time": "fully-implicit"},
        facilities={"wells": "deck"}, schedule={"report_steps": 2},
        solver={"nonlinear": "NonLinearSolver", "max_iterations": 30},
        metadata={"qualification": "M3.9"})


def test_json_round_trip_preserves_hash_and_constructs_bit_identical_run(tmp_path):
    original = _spe1_case()
    path = write_reservoir_case(tmp_path / "spe1.case.json", original)
    loaded = read_reservoir_case(path)
    assert loaded == original
    assert loaded.content_hash == original.content_hash
    a, b = original.construct(), loaded.construct()
    np.testing.assert_array_equal(a.result.time, b.result.time)
    for name in ("pressure", "sW", "sG", "rs", "bhp", "qOs"):
        np.testing.assert_array_equal(a.result.history[name], b.result.history[name])
    np.testing.assert_array_equal(a.diagnostics.iterations, b.diagnostics.iterations)


def test_execution_settings_are_not_part_of_portable_case_schema():
    value = _spe1_case().to_dict()
    value["device"] = "gpu"
    with pytest.raises(ValueError, match="invalid reservoir-case fields"):
        ReservoirCase.from_dict(value)


def test_unknown_schema_and_tampered_content_are_refused():
    value = _spe1_case().to_dict()
    value["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        ReservoirCase.from_dict(value)
    value = _spe1_case().to_dict()
    value["physics"]["disgas"] = False
    with pytest.raises(ValueError, match="content hash mismatch"):
        ReservoirCase.from_dict(value)


def test_eclipse_case_becomes_first_class_and_verifies_source_artifacts(tmp_path):
    from pymrst.research_studio.eclipse import write_egrid
    prefix = tmp_path / "IMPORTED"
    write_egrid(prefix.with_suffix(".EGRID"), {
        "cartDims": [1, 1, 1],
        "COORD": [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                  0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        "ZCORN": [0, 0, 0, 0, 1, 1, 1, 1], "ACTNUM": [1]})
    case = ReservoirCase.from_eclipse(prefix)
    assert case.metadata["source"] == "ECLIPSE"
    constructed = case.construct()
    assert constructed.grid.cells.num == 1
    prefix.with_suffix(".EGRID").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        case.construct()
