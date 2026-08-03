"""M3.5 normalized xarray, Zarr, CSV and Parquet result formats."""

from dataclasses import replace

import numpy as np
import pandas as pd

from pymrst.research_studio.experiments import (
    ExplicitVariantMatrix, LocalEnsembleExecutor, MemberResultStore,
    VariantDefinition, materialize_variant_matrix,
)
from pymrst.research_studio.export import (
    NormalizedResultDataset, read_zarr, summary_rows, write_csv_summary,
    write_parquet_summary, write_zarr,
)
from scripts.build_research_studio_experiment_registry import (
    spe1_time_integration_experiment,
)


RUNNER = "pymrst.research_studio.experiments.runners:deterministic_manufactured_member"


def _result(tmp_path):
    definition = replace(spe1_time_integration_experiment(), variants=())
    matrix = ExplicitVariantMatrix((
        VariantDefinition("format", parameters={"test.multiplier": 1.0},
                          comparison_group="formats"),
    ))
    materialized = materialize_variant_matrix(definition, matrix)
    report = LocalEnsembleExecutor(
        tmp_path / "members", max_workers=1,
        memory_per_worker_bytes=1024).run(materialized, runner=RUNNER)
    assert len(report.completed) == 1
    return MemberResultStore(report.completed[0].store_path).lazy()


def test_normalized_xarray_preserves_values_dimensions_and_lineage(tmp_path):
    result = _result(tmp_path)
    dataset = NormalizedResultDataset(result).to_xarray()
    assert dataset.sizes["time"] == 3
    assert dataset.sizes["fields__pressure__axis0"] == 4
    np.testing.assert_array_equal(dataset.time.values, result.times)
    np.testing.assert_array_equal(
        dataset["fields__pressure"].values,
        result.array("fields", "pressure"))
    assert dataset.attrs["member_id"] == result.record["coordinate"]["member_id"]
    assert dataset.attrs["comparison_group"] == "formats"


def test_zarr_round_trip_supports_lazy_and_eager_reads(tmp_path):
    result = _result(tmp_path)
    target = write_zarr(tmp_path / "result.zarr", NormalizedResultDataset(result))
    lazy = read_zarr(target, lazy=True)
    eager = read_zarr(target, lazy=False)
    np.testing.assert_array_equal(
        lazy["fields__pressure"].isel(time=1).values,
        result.array("fields", "pressure", time_slice=1))
    np.testing.assert_array_equal(
        eager["wells__producer_rate"].values,
        result.array("wells", "producer_rate"))
    assert eager.attrs["member_id"] == result.record["coordinate"]["member_id"]
    lazy.close()
    eager.close()


def test_csv_and_parquet_share_exact_long_form_values(tmp_path):
    result = _result(tmp_path)
    expected = pd.DataFrame.from_records(summary_rows(result))
    csv_path = write_csv_summary(tmp_path / "summary.csv", result)
    parquet_path = write_parquet_summary(tmp_path / "summary.parquet", result)
    csv_frame = pd.read_csv(csv_path)
    parquet_frame = pd.read_parquet(parquet_path)
    assert tuple(csv_frame.columns) == tuple(expected.columns)
    assert tuple(parquet_frame.columns) == tuple(expected.columns)
    pd.testing.assert_frame_equal(csv_frame, expected, check_dtype=False)
    pd.testing.assert_frame_equal(parquet_frame, expected, check_dtype=False)
    assert set(expected.category) == {"regions", "wells", "convergence", "conservation"}
