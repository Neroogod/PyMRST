"""Evidence-backed report assembly, rendering, and publishing tools."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Iterable

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..context import InvestigationContext
from ..permissions import AgentPermission
from ..schemas.engineering_report import EngineeringReport, ReportSection
from ..schemas.findings import ClaimKind, EngineeringFinding, FindingSeverity
from ..narrative import EvidenceLinkedNarrative


def _section_id(category: str) -> str:
    value = category.split(".", 1)[0].strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "general"


def _deduplicate_evidence(findings):
    output, seen = [], set()
    for finding in findings:
        for evidence in finding.evidence:
            key = (
                evidence.kind.value, evidence.uri, evidence.checksum,
                json.dumps(dict(evidence.locator), sort_keys=True, separators=(",", ":")),
            )
            if key not in seen:
                seen.add(key)
                output.append(evidence)
    return tuple(output)


class EngineeringReportBuilder:
    """Build a conservative report entirely from qualified typed findings."""

    def build(
        self,
        context: InvestigationContext,
        findings: Iterable[EngineeringFinding],
        *,
        title: str,
        qualified: bool = False,
        assumptions: Iterable[str] = (),
        limitations: Iterable[str] = (),
        review_notes: Iterable[str] = (),
        narrative: EvidenceLinkedNarrative | None = None,
    ) -> EngineeringReport:
        findings = tuple(findings)
        assumptions = tuple(str(item).strip() for item in assumptions)
        limitations = tuple(str(item).strip() for item in limitations)
        review_notes = tuple(str(item).strip() for item in review_notes)
        if any(not item for item in (*assumptions, *limitations, *review_notes)):
            raise ValueError("report annotations cannot contain empty values")
        identifiers = [item.finding_id for item in findings]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("report builder received duplicate finding ids")
        finding_lookup = {item.finding_id: item for item in findings}
        if narrative is not None:
            if not isinstance(narrative, EvidenceLinkedNarrative):
                raise TypeError("report narrative must be EvidenceLinkedNarrative")
            if narrative.status != "approved":
                raise ValueError("only a human-approved narrative may enter a report")
            referenced = {
                identifier for section in narrative.sections
                for identifier in section.finding_ids
            }
            if referenced != set(finding_lookup):
                raise ValueError("approved narrative does not match report findings")
            sections = [ReportSection(
                item.section_id, item.heading,
                "\n\n".join((item.narrative, *item.uncertainty_caveats)),
                item.finding_ids, item.evidence,
            ) for item in narrative.sections]
            summary = narrative.executive_summary
        else:
            grouped = defaultdict(list)
            for finding in findings:
                grouped[_section_id(finding.category)].append(finding)
            sections = []
            for section_id in sorted(grouped):
                items = tuple(grouped[section_id])
                title_text = section_id.replace("_", " ").title()
                sections.append(ReportSection(
                    section_id, title_text,
                    f"This section contains {len(items)} evidence-backed "
                    f"engineering finding(s).",
                    tuple(item.finding_id for item in items),
                    _deduplicate_evidence(items),
                ))
            if not sections:
                sections = [ReportSection(
                    "overview", "Overview",
                    "The completed investigation produced no engineering findings.",
                )]
            counts = Counter(item.severity.value for item in findings)
            summary = (
                f"The investigation completed with {len(findings)} evidence-backed "
                f"finding(s): {counts.get('critical', 0)} critical, "
                f"{counts.get('error', 0)} error, {counts.get('warning', 0)} warning, "
                f"and {counts.get('info', 0)} informational."
            )
        if review_notes:
            sections.append(ReportSection(
                "human_review", "Human review",
                "\n".join(f"- {item}" for item in review_notes),
            ))
        identity = hashlib.sha256(json.dumps({
            "investigation_id": context.investigation_id,
            "case_hash": context.case_content_hash,
            "finding_ids": identifiers,
            "qualified": bool(qualified),
            "title": str(title),
            "assumptions": assumptions,
            "limitations": limitations,
            "review_notes": review_notes,
            "narrative_id": None if narrative is None else narrative.narrative_id,
        }, sort_keys=True).encode()).hexdigest()[:20]
        return EngineeringReport(
            f"report_{identity}", context.investigation_id, title, summary,
            findings, tuple(sections), assumptions, limitations,
            bool(qualified),
        )


def render_markdown(report: EngineeringReport) -> str:
    finding_lookup = {item.finding_id: item for item in report.findings}
    lines = [
        f"# {report.title}", "",
        f"- Report ID: `{report.report_id}`",
        f"- Investigation ID: `{report.investigation_id}`",
        f"- Status: **{'Qualified' if report.qualified else 'Draft / not qualified'}**",
        f"- Content SHA-256: `{report.content_hash}`", "",
        "## Executive summary", "", report.executive_summary, "",
    ]
    for section in report.sections:
        lines.extend((f"## {section.title}", "", section.narrative, ""))
        for identifier in section.finding_ids:
            finding = finding_lookup[identifier]
            lines.extend((
                f"### {finding.severity.value.upper()}: {finding.category}", "",
                finding.statement, "",
                f"- Claim type: `{finding.claim_kind.value}`",
                f"- Confidence: `{finding.confidence:.3f}`",
                f"- Finding ID: `{finding.finding_id}`",
            ))
            if finding.affected_objects:
                lines.append("- Affected: " + ", ".join(
                    f"`{item}`" for item in finding.affected_objects
                ))
            lines.append("")
    for heading, values in (
        ("Assumptions", report.assumptions), ("Limitations", report.limitations),
    ):
        if values:
            lines.extend((f"## {heading}", ""))
            lines.extend(f"- {item}" for item in values)
            lines.append("")
    lines.extend(("## Evidence appendix", ""))
    evidence = _deduplicate_evidence(report.findings)
    if not evidence:
        lines.extend(("No evidence references were attached.", ""))
    else:
        for item in evidence:
            checksum = item.checksum or "not recorded"
            lines.extend((
                f"- **{item.label}**", f"  - URI: `{item.uri}`",
                f"  - Kind: `{item.kind.value}`", f"  - Checksum: `{checksum}`",
            ))
    return "\n".join(lines).rstrip() + "\n"


def render_html(report: EngineeringReport) -> str:
    """Render a standalone, script-free technical report."""
    escape = lambda value: html.escape(str(value), quote=True)
    finding_lookup = {item.finding_id: item for item in report.findings}
    parts = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{escape(report.title)}</title>",
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;"
        "margin:2rem auto;padding:0 1.5rem;color:#17212b}code{background:#eef2f5;"
        "padding:.12rem .3rem}section{border-top:1px solid #ccd5dd;margin-top:2rem}"
        ".meta{color:#52606d}.warning{border-left:4px solid #b7791f;padding-left:1rem}"
        "</style></head><body>",
        f"<h1>{escape(report.title)}</h1>",
        f'<p class="meta">Report <code>{escape(report.report_id)}</code> · '
        f"Investigation <code>{escape(report.investigation_id)}</code> · "
        f"{'Qualified' if report.qualified else 'Draft / not qualified'}</p>",
        "<h2>Executive summary</h2>", f"<p>{escape(report.executive_summary)}</p>",
    ]
    for section in report.sections:
        parts.extend(("<section>", f"<h2>{escape(section.title)}</h2>",
                      f"<p>{escape(section.narrative)}</p>"))
        for identifier in section.finding_ids:
            finding = finding_lookup[identifier]
            parts.extend((
                f'<article class="{escape(finding.severity.value)}">',
                f"<h3>{escape(finding.severity.value.upper())}: "
                f"{escape(finding.category)}</h3>",
                f"<p>{escape(finding.statement)}</p>",
                f'<p class="meta">Claim: {escape(finding.claim_kind.value)} · '
                f"Confidence: {finding.confidence:.3f} · "
                f"ID: <code>{escape(finding.finding_id)}</code></p></article>",
            ))
        parts.append("</section>")
    for title, values in (("Assumptions", report.assumptions),
                          ("Limitations", report.limitations)):
        if values:
            parts.extend((f"<section><h2>{title}</h2><ul>", *(
                f"<li>{escape(item)}</li>" for item in values), "</ul></section>"))
    parts.extend(("<section><h2>Evidence appendix</h2><ul>",))
    for item in _deduplicate_evidence(report.findings):
        parts.append(
            f"<li><strong>{escape(item.label)}</strong> · "
            f"<code>{escape(item.uri)}</code> · {escape(item.kind.value)} · "
            f"SHA: <code>{escape(item.checksum or 'not recorded')}</code></li>"
        )
    parts.extend(("</ul></section>", "</body></html>"))
    return "\n".join(parts) + "\n"


class ReportCatalog:
    def __init__(self, allowed_roots=(), *, maximum_markdown_characters=5_000_000):
        if isinstance(allowed_roots, (str, Path)):
            allowed_roots = (allowed_roots,)
        self.roots = tuple(Path(item).resolve() for item in allowed_roots)
        self.maximum_markdown_characters = int(maximum_markdown_characters)
        if self.maximum_markdown_characters <= 0:
            raise ValueError("maximum report size must be positive")
        self._reports = {}

    def register(self, report: EngineeringReport):
        if not isinstance(report, EngineeringReport):
            raise TypeError("report catalog accepts EngineeringReport values")
        if report.report_id in self._reports:
            raise KeyError(f"report already registered: {report.report_id}")
        self._reports[report.report_id] = report
        return report

    def get(self, report_id: str):
        try:
            return self._reports[report_id]
        except KeyError as error:
            raise KeyError(f"unknown report: {report_id}") from error

    def markdown(self, report_id: str):
        value = render_markdown(self.get(report_id))
        if len(value) > self.maximum_markdown_characters:
            raise ValueError("rendered report exceeds configured size limit")
        return value

    def render(self, report_id: str, format: str = "markdown") -> str:
        format = str(format).lower()
        if format in {"markdown", "md"}:
            value = render_markdown(self.get(report_id))
        elif format in {"html", "htm"}:
            value = render_html(self.get(report_id))
        elif format == "json":
            value = self.get(report_id).to_json(indent=2) + "\n"
        else:
            raise ValueError(f"unsupported report format: {format}")
        if len(value) > self.maximum_markdown_characters:
            raise ValueError("rendered report exceeds configured size limit")
        return value

    def write_markdown(self, report_id: str, path: str | Path):
        return self.write_report(report_id, path, format="markdown")

    def write_report(
        self, report_id: str, path: str | Path, *, format: str | None = None,
    ):
        if not self.roots:
            raise ValueError("report publication requires configured output roots")
        target = Path(path).resolve()
        formats = {
            ".md": "markdown", ".markdown": "markdown",
            ".html": "html", ".htm": "html", ".json": "json",
        }
        inferred = formats.get(target.suffix.lower())
        if inferred is None:
            raise ValueError("report output must be Markdown, HTML, or JSON")
        selected = inferred if format is None else str(format).lower()
        aliases = {"md": "markdown", "htm": "html"}
        selected = aliases.get(selected, selected)
        if selected != inferred:
            raise ValueError("report format and output extension disagree")
        if not any(target == root or root in target.parents for root in self.roots):
            raise ValueError(f"report path lies outside configured roots: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(self.render(report_id, selected), encoding="utf-8")
        temporary.replace(target)
        return target


class RenderEngineeringReportTool:
    descriptor = ToolDescriptor(
        "render_engineering_report", "Render a registered report as Markdown",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ReportCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall):
        report_id = str(call.arguments["report_id"])
        format = str(call.arguments.get("format", "markdown"))
        report = self.catalog.get(report_id)
        return ToolResult(call.call_id, True, {
            "report_id": report_id, "content_hash": report.content_hash,
            "format": format, "content": self.catalog.render(report_id, format),
            **({"markdown": self.catalog.markdown(report_id)}
               if format in {"markdown", "md"} else {}),
        })


class PublishEngineeringReportTool:
    descriptor = ToolDescriptor(
        "publish_engineering_report", "Write a registered report to an approved path",
        AgentPermission.PUBLISH, requires_approval=True,
    )

    def __init__(self, catalog: ReportCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall):
        target = self.catalog.write_report(
            str(call.arguments["report_id"]), str(call.arguments["path"]),
            format=call.arguments.get("format"),
        )
        return ToolResult(call.call_id, True, {"path": str(target)})


def _finding(severity, category, statement, *, affected=(), details=None):
    return {
        "severity": severity, "category": category, "statement": statement,
        "affected_objects": list(affected), "details": details or {},
    }


class ReportCompositionTool:
    """Audit an assembled report's evidence and claim-type completeness.

    This inspects the report itself, not the underlying simulation, so it
    runs after :class:`EngineeringReportBuilder` has produced a candidate
    report and before that report is published.
    """

    descriptor = ToolDescriptor(
        "analyze_report_composition",
        "Audit an assembled engineering report for evidence and "
        "claim-type completeness",
        AgentPermission.ANALYZE,
    )

    def __init__(self, catalog: ReportCatalog):
        self.catalog = catalog

    def invoke(self, call: ToolCall) -> ToolResult:
        report = self.catalog.get(str(call.arguments["report_id"]))
        findings = []

        severe = tuple(
            item for item in report.findings
            if item.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL)
        )
        missing_recommendation = tuple(
            item for item in severe if item.recommendation is None
        )
        if missing_recommendation:
            findings.append(_finding(
                "warning", "report.missing_recommendation",
                f"{len(missing_recommendation)} error/critical finding(s) have "
                f"no proposed recommendation",
                affected=(item.finding_id for item in missing_recommendation),
            ))

        unverifiable = sorted({
            item.uri for finding in report.findings for item in finding.evidence
            if not item.checksum
        })
        if unverifiable:
            findings.append(_finding(
                "warning", "report.unverifiable_evidence",
                f"{len(unverifiable)} evidence reference(s) have no checksum "
                f"and cannot be independently re-verified",
                affected=unverifiable,
            ))

        if severe and not report.limitations:
            findings.append(_finding(
                "warning", "report.missing_limitations",
                "report contains error/critical findings but declares no "
                "limitations",
            ))

        uncertain = any(
            item.claim_kind in (ClaimKind.INFERRED, ClaimKind.UNCERTAIN)
            for item in report.findings
        )
        if uncertain and not report.assumptions:
            findings.append(_finding(
                "warning", "report.missing_assumptions",
                "report contains inferred or uncertain findings but declares "
                "no assumptions",
            ))

        if not report.qualified:
            findings.append(_finding(
                "info", "report.not_qualified",
                "report is not qualified; findings should be treated as "
                "draft pending human qualification",
            ))

        value = {
            "analysis_kind": self.descriptor.name,
            "report_id": report.report_id,
            "ok": not any(item["severity"] in {"error", "critical"} for item in findings),
            "finding_count": len(report.findings),
            "severity_counts": dict(Counter(item.severity.value for item in report.findings)),
            "claim_kind_counts": dict(Counter(item.claim_kind.value for item in report.findings)),
            "qualified": report.qualified, "findings": findings,
            "evidence": {
                "kind": "validation", "uri": f"report://{report.report_id}",
                "label": f"Composition audit for {report.report_id}",
                "locator": {}, "checksum": report.content_hash,
            },
        }
        return ToolResult(call.call_id, True, value)


__all__ = [
    "EngineeringReportBuilder", "PublishEngineeringReportTool", "ReportCatalog",
    "ReportCompositionTool", "RenderEngineeringReportTool", "render_html",
    "render_markdown",
]
