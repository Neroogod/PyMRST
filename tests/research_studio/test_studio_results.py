"""Secure lazy member-store and reservoir-viewer delivery."""

from dataclasses import replace

import numpy as np
import pytest

from pymrst.research_studio.experiments import (
    ExplicitVariantMatrix, LocalEnsembleExecutor, VariantDefinition,
    materialize_variant_matrix,
)
from pymrst.research_studio.studio import ResultAccessError, ResultCatalog
from scripts.build_research_studio_experiment_registry import (
    spe1_time_integration_experiment,
)


def _store(tmp_path):
    definition = replace(spe1_time_integration_experiment(), variants=())
    matrix = ExplicitVariantMatrix((VariantDefinition(
        "result", parameters={"test.multiplier": 1.0}),))
    report = LocalEnsembleExecutor(tmp_path / "stores", max_workers=1,
                                   memory_per_worker_bytes=1024).run(
        materialize_variant_matrix(definition, matrix),
        runner="pymrst.research_studio.experiments.runners:deterministic_manufactured_member")
    return report.completed[0].store_path


def test_catalog_exposes_metadata_and_bounded_lazy_slices(tmp_path):
    store = _store(tmp_path)
    viewer = tmp_path / "viewer.html"; viewer.write_text("<html>viewer</html>")
    catalog = ResultCatalog((tmp_path,), maximum_json_values=4)
    catalog.register("r", "Result", store, viewer_path=viewer)
    metadata = catalog.metadata("r")
    assert metadata["categories"]["fields"] == ["pressure", "saturation"]
    assert metadata["viewer_available"]
    values = catalog.array("r", "fields", "pressure", time_index=1,
                           start=1, stop=3)
    np.testing.assert_array_equal(values["values"], [10000003.0, 10000004.0])
    assert catalog.viewer("r") == viewer.resolve()
    with pytest.raises(ResultAccessError, match="limit"):
        catalog.array("r", "fields", "pressure")


def test_paths_outside_roots_and_invalid_queries_are_refused(tmp_path):
    store = _store(tmp_path)
    catalog = ResultCatalog((tmp_path / "different"), maximum_json_values=10)
    with pytest.raises(ResultAccessError, match="outside"):
        catalog.register("r", "Result", store)
    catalog = ResultCatalog((tmp_path,), maximum_json_values=10)
    catalog.register("r", "Result", store)
    with pytest.raises(ResultAccessError, match="invalid result category"):
        catalog.array("r", "secrets", "pressure")
    with pytest.raises(ResultAccessError, match="outside the result history"):
        catalog.array("r", "fields", "pressure", time_index=99)
