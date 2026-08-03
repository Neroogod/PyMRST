"""Long-form CSV and Parquet summaries from normalized member results."""

from __future__ import annotations

import csv
from pathlib import Path

from pymrst.research_studio.experiments.execution import LazyMemberResult


TABULAR_SCHEMA_VERSION = 1


def summary_rows(result: LazyMemberResult, *, categories=("regions", "wells",
                                                          "convergence", "conservation")):
    member_id = result.record["coordinate"]["member_id"]
    group = result.record["coordinate"]["comparison_group"]
    times = result.times
    for category in categories:
        for name in result.names(category):
            values = result.array(category, name).reshape(times.size, -1)
            for time_index, time in enumerate(times):
                for entity, value in enumerate(values[time_index]):
                    yield {"schema_version": TABULAR_SCHEMA_VERSION,
                           "member_id": member_id, "comparison_group": group,
                           "time": float(time), "category": category,
                           "quantity": name, "entity": entity, "value": float(value)}


def write_csv_summary(path: str | Path, result: LazyMemberResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ("schema_version", "member_id", "comparison_group", "time",
              "category", "quantity", "entity", "value")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows(result))
    temporary.replace(target)
    return target


def write_parquet_summary(path: str | Path, result: LazyMemberResult) -> Path:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Parquet export requires pandas and pyarrow; install pymrst[studio]") from error
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(summary_rows(result))
    frame.to_parquet(target, index=False)
    return target


__all__ = ["TABULAR_SCHEMA_VERSION", "summary_rows", "write_csv_summary",
           "write_parquet_summary"]
