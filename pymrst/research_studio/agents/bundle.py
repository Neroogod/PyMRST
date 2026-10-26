"""Atomic, tamper-evident persistence for single-case investigations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from ._serialization import require_fields
from .orchestrator import InvestigationOutcome
from .schemas.engineering_report import EngineeringReport
from .tools.report_tools import render_markdown


INVESTIGATION_BUNDLE_SCHEMA_VERSION = 1
DEFAULT_MAXIMUM_BUNDLE_FILE_BYTES = 20_000_000


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe investigation bundle member: {relative!r}")
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"bundle members cannot be symbolic links: {relative}")
    return path


def write_investigation_bundle(
    path: str | Path,
    outcome: InvestigationOutcome,
) -> Path:
    """Create a new bundle atomically; existing destinations are never replaced."""
    if not isinstance(outcome, InvestigationOutcome):
        raise TypeError("bundle writer requires an InvestigationOutcome")
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"investigation bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.staging-", dir=target.parent,
    ))
    try:
        payloads: dict[str, bytes] = {
            "outcome.json": _encoded_json(outcome.to_dict()),
            "investigation.json": _encoded_json(outcome.record.to_dict()),
        }
        if outcome.report is not None:
            payloads["report.json"] = _encoded_json(outcome.report.to_dict())
            payloads["report.md"] = render_markdown(outcome.report).encode("utf-8")
        for name, value in payloads.items():
            (staging / name).write_bytes(value)
        manifest = {
            "schema_version": INVESTIGATION_BUNDLE_SCHEMA_VERSION,
            "investigation_id": outcome.record.context.investigation_id,
            "case_id": str(outcome.record.context.case_id),
            "case_content_hash": outcome.record.context.case_content_hash,
            "status": outcome.record.status.value,
            "report_id": None if outcome.report is None else outcome.report.report_id,
            "files": {
                name: {"bytes": len(value), "sha256": _sha256(value)}
                for name, value in sorted(payloads.items())
            },
        }
        (staging / "manifest.json").write_bytes(_encoded_json(manifest))
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def read_investigation_bundle(
    path: str | Path,
    *,
    maximum_file_bytes: int = DEFAULT_MAXIMUM_BUNDLE_FILE_BYTES,
) -> InvestigationOutcome:
    """Verify all declared bytes and redundant records before returning an outcome."""
    root = Path(path).resolve()
    limit = int(maximum_file_bytes)
    if limit <= 0:
        raise ValueError("maximum bundle file size must be positive")
    manifest_path = _safe_file(root, "manifest.json")
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(f"not an investigation bundle: {root}")
    if manifest_path.stat().st_size > limit:
        raise ValueError("investigation bundle manifest exceeds size limit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = require_fields(
        manifest,
        required={
            "schema_version", "investigation_id", "case_id",
            "case_content_hash", "status", "report_id", "files",
        },
        name="InvestigationBundleManifest",
    )
    if data["schema_version"] != INVESTIGATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported investigation bundle schema version")
    if not isinstance(data["files"], dict) or "outcome.json" not in data["files"]:
        raise ValueError("investigation bundle has no outcome payload")
    declared = set(data["files"])
    actual = {item.name for item in root.iterdir() if item.name != "manifest.json"}
    if actual != declared:
        raise ValueError(
            f"bundle member mismatch; missing={sorted(declared - actual)}, "
            f"unexpected={sorted(actual - declared)}"
        )
    payloads: dict[str, bytes] = {}
    for name, metadata in data["files"].items():
        if set(metadata) != {"bytes", "sha256"}:
            raise ValueError(f"invalid metadata for bundle member {name}")
        member = _safe_file(root, name)
        if not member.is_file():
            raise ValueError(f"bundle member is not a regular file: {name}")
        if member.stat().st_size > limit:
            raise ValueError(f"bundle member exceeds size limit: {name}")
        value = member.read_bytes()
        if len(value) != int(metadata["bytes"]) or _sha256(value) != metadata["sha256"]:
            raise ValueError(f"bundle member failed checksum validation: {name}")
        payloads[name] = value
    outcome = InvestigationOutcome.from_json(payloads["outcome.json"].decode("utf-8"))
    if json.loads(payloads["investigation.json"]) != outcome.record.to_dict():
        raise ValueError("bundle investigation record disagrees with outcome")
    context = outcome.record.context
    identity = (
        context.investigation_id, str(context.case_id), context.case_content_hash,
        outcome.record.status.value,
        None if outcome.report is None else outcome.report.report_id,
    )
    declared_identity = (
        data["investigation_id"], data["case_id"], data["case_content_hash"],
        data["status"], data["report_id"],
    )
    if identity != declared_identity:
        raise ValueError("bundle manifest identity disagrees with outcome")
    if outcome.report is None:
        if {"report.json", "report.md"} & declared:
            raise ValueError("bundle contains report files for an outcome without a report")
    else:
        if not {"report.json", "report.md"}.issubset(declared):
            raise ValueError("bundle report payload is incomplete")
        report = EngineeringReport.from_dict(json.loads(payloads["report.json"]))
        if report != outcome.report:
            raise ValueError("bundle report disagrees with outcome")
        if payloads["report.md"].decode("utf-8") != render_markdown(report):
            raise ValueError("bundle Markdown report disagrees with structured report")
    return outcome


__all__ = [
    "DEFAULT_MAXIMUM_BUNDLE_FILE_BYTES", "INVESTIGATION_BUNDLE_SCHEMA_VERSION",
    "read_investigation_bundle", "write_investigation_bundle",
]
