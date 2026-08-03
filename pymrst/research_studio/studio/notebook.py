"""M5.2 rich, dependency-light notebook representations over M5.1 data."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json


def _bar(value, color="#38bdf8"):
    percent = max(0.0, min(100.0, 100.0*float(value)))
    return (f'<div style="height:8px;background:#1e293b;border-radius:8px">'
            f'<div style="height:8px;width:{percent:.2f}%;background:{color};'
            'border-radius:8px"></div></div>')


@dataclass(frozen=True, slots=True)
class RunView:
    record: object

    def _repr_html_(self):
        value = self.record.to_dict()
        status = html.escape(value["status"])
        color = {"completed": "#22c55e", "failed": "#ef4444",
                 "cancelled": "#f59e0b"}.get(status, "#38bdf8")
        return (f'<section style="font:14px system-ui;background:#0f172a;color:#e2e8f0;'
                'padding:16px;border-radius:12px;max-width:680px">'
                f'<b>{html.escape(value["workflow_id"])}</b> '
                f'<span style="color:{color}">{status}</span>'
                f'<p>{html.escape(value["message"])}</p>{_bar(value["progress"], color)}'
                f'<small>{value["elapsed_seconds"]:.3f}s · {len(value["events"])} events</small>'
                '</section>')


@dataclass(frozen=True, slots=True)
class ExperimentView:
    definition: object

    def _repr_html_(self):
        value = self.definition.to_dict()
        outputs = ", ".join(item["name"] for item in value.get("outputs", ()))
        return (f'<article style="font:14px system-ui;border:1px solid #334155;'
                'border-radius:12px;padding:16px">'
                f'<h3>{html.escape(value["name"])}</h3>'
                f'<p>{html.escape(value["hypothesis"])}</p>'
                f'<b>Outputs:</b> {html.escape(outputs)}</article>')


@dataclass(frozen=True, slots=True)
class ComparisonView:
    comparison: object

    def _repr_html_(self):
        value = self.comparison.to_dict()
        rows = "".join(
            f'<tr><td>{html.escape(name)}</td><td>{result["maximum_absolute_error"]:.5g}</td>'
            f'<td>{"pass" if result["passed"] else "fail"}</td></tr>'
            for name, result in value["results"].items())
        return ('<table style="font:13px system-ui;border-collapse:collapse">'
                '<thead><tr><th>Quantity</th><th>Max |error|</th><th>Status</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>')


__all__ = ["ComparisonView", "ExperimentView", "RunView"]
