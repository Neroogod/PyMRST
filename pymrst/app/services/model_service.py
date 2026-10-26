"""Bounded, read-only ECLIPSE deck inspection for the application workspace."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.model import (
    DeckKeywordOccurrence, IssueSeverity, ModelInspection, SourceLocation,
    ValidationIssue,
)
from .project_service import ProjectService, _file_hash


_KEYWORD = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]{0,7})(?:\s|$)")
_SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION", "SUMMARY", "SCHEDULE")
_SECTION_SET = frozenset(_SECTIONS)
_UNITS = frozenset(("METRIC", "FIELD", "LAB"))
_NO_RECORD = _SECTION_SET | _UNITS | frozenset({
    "OIL", "WATER", "GAS", "DISGAS", "VAPOIL", "UNIFOUT", "UNIFIN",
    "NOSIM", "BLACKOIL", "ECHO", "NOECHO",
})


def _safe_text(path: Path, maximum_bytes: int) -> list[str]:
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"deck source exceeds {maximum_bytes} byte inspection limit")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _clean(line: str) -> str:
    return line.split("--", 1)[0].strip()


def _count_values(records: list[str]) -> tuple[int, int]:
    values = defaults = 0
    for record in records:
        for token in re.findall(r"'(?:[^']|'')*'|\S+", record.replace("/", " ")):
            match = re.fullmatch(r"(\d+)\*(.*)", token)
            if match:
                count = int(match.group(1))
                values += count
                if not match.group(2):
                    defaults += count
            else:
                values += 1
    return values, defaults


class ModelInspectionService:
    """Indexes source text without invoking a simulator or mutating an input."""

    def __init__(
        self, database: AppDatabase, projects: ProjectService, *,
        maximum_file_bytes: int = 64 * 1024**2, maximum_files: int = 512,
        maximum_lines: int = 2_000_000,
    ) -> None:
        self.database, self.projects = database, projects
        self.maximum_file_bytes = int(maximum_file_bytes)
        self.maximum_files = int(maximum_files)
        self.maximum_lines = int(maximum_lines)

    def inspect(self, project_id: str, asset_id: str, *, name: str = "") -> ModelInspection:
        asset = self.projects.asset(project_id, asset_id)
        root = self.projects.asset_path(project_id, asset_id)
        project = self.projects.get(project_id)
        allowed = (Path(project.root_path) / "inputs" / "original").resolve()
        occurrences: list[DeckKeywordOccurrence] = []
        issues: list[ValidationIssue] = []
        files: list[dict[str, Any]] = []
        source_records: dict[str, list[str]] = {}
        visited: set[Path] = set()
        total_lines = 0
        sequence = 0

        def issue(code, severity, category, message, path=None, line=1, details=None):
            location = None if path is None else SourceLocation(
                asset.asset_id, str(path.relative_to(allowed)), line,
            )
            issues.append(ValidationIssue(
                code, severity, category, message, location, details or {},
            ))

        def scan(path: Path, section: str = "", stack: tuple[Path, ...] = ()) -> str:
            nonlocal total_lines, sequence
            path = path.resolve()
            if path != allowed and allowed not in path.parents:
                issue("include.path_escape", IssueSeverity.ERROR, "include",
                      "INCLUDE resolves outside immutable project inputs", stack[-1] if stack else root)
                return section
            if path in stack:
                issue("include.cycle", IssueSeverity.ERROR, "include",
                      "INCLUDE cycle detected", stack[-1], details={"target": str(path)})
                return section
            if path in visited:
                return section
            if len(visited) >= self.maximum_files:
                issue("deck.file_limit", IssueSeverity.ERROR, "resource",
                      "deck exceeds configured file-count inspection limit", root)
                return section
            if not path.is_file() or path.is_symlink():
                issue("include.missing", IssueSeverity.ERROR, "include",
                      f"included source is missing: {path.name}", stack[-1] if stack else root)
                return section
            visited.add(path)
            try:
                lines = _safe_text(path, self.maximum_file_bytes)
            except ValueError as error:
                issue("deck.file_size", IssueSeverity.ERROR, "resource", str(error), path)
                return section
            total_lines += len(lines)
            if total_lines > self.maximum_lines:
                issue("deck.line_limit", IssueSeverity.ERROR, "resource",
                      "deck exceeds configured total line limit", path)
                return section
            checksum = _file_hash(path)
            files.append({
                "path": str(path.relative_to(allowed)), "content_hash": checksum,
                "size_bytes": path.stat().st_size, "line_count": len(lines),
            })
            index = 0
            while index < len(lines):
                text = _clean(lines[index])
                match = _KEYWORD.match(text) if text and not text.startswith(("'", '"')) else None
                if not match:
                    index += 1
                    continue
                keyword = match.group(1).upper()
                start = index + 1
                if keyword in _SECTION_SET:
                    section = keyword
                records: list[str] = []
                end = start
                if keyword not in _NO_RECORD:
                    tail = text[match.end():].strip()
                    if tail:
                        records.append(tail)
                    cursor = index
                    complete = "/" in tail
                    while not complete and cursor + 1 < len(lines):
                        cursor += 1
                        record = _clean(lines[cursor])
                        if record:
                            records.append(record)
                            complete = "/" in record
                    end = cursor + 1
                values, defaults = _count_values(records)
                sequence += 1
                occurrence_id = f"occ_{sequence:08d}"
                occurrences.append(DeckKeywordOccurrence(
                    occurrence_id, keyword, section,
                    SourceLocation(asset.asset_id, str(path.relative_to(allowed)), start),
                    end, len(records), values, defaults,
                ))
                source_records[occurrence_id] = records
                if keyword == "INCLUDE":
                    joined = re.sub(r"/\s*$", "", " ".join(records)).strip().strip("'\"")
                    if not joined:
                        issue("include.empty", IssueSeverity.ERROR, "include",
                              "INCLUDE has no source path", path, start)
                    else:
                        target = (path.parent / joined).resolve()
                        if target != allowed and allowed not in target.parents:
                            issue("include.path_escape", IssueSeverity.ERROR, "include",
                                  "INCLUDE resolves outside immutable project inputs", path, start,
                                  {"target": joined})
                        elif not target.is_file():
                            issue("include.missing", IssueSeverity.ERROR, "include",
                                  f"INCLUDE file not found: {joined}", path, start)
                        else:
                            section = scan(target, section, stack + (path,))
                index = max(index + 1, end)
            return section

        scan(root)
        sections = tuple(dict.fromkeys(
            item.keyword for item in occurrences if item.keyword in _SECTION_SET
        ))
        positions = [_SECTIONS.index(item) for item in sections]
        if positions != sorted(positions):
            issue("deck.section_order", IssueSeverity.ERROR, "structure",
                  "deck sections are not in canonical ECLIPSE order", root,
                  details={"sections": list(sections)})
        for required in ("RUNSPEC", "GRID", "PROPS", "SOLUTION", "SCHEDULE"):
            if required not in sections:
                issue("deck.section_missing", IssueSeverity.WARNING, "structure",
                      f"section {required} is not present", root,
                      details={"section": required})
        units = [item for item in occurrences if item.keyword in _UNITS]
        unit_system = units[0].keyword if len({item.keyword for item in units}) == 1 else "UNKNOWN"
        if not units:
            issue("units.unspecified", IssueSeverity.WARNING, "units",
                  "unit system is not declared; simulator defaults may differ", root)
        elif len({item.keyword for item in units}) > 1:
            issue("units.conflict", IssueSeverity.ERROR, "units",
                  "multiple unit systems are declared", root,
                  details={"units": sorted({item.keyword for item in units})})
        dimens = next((item for item in occurrences if item.keyword == "DIMENS"), None)
        if dimens is not None:
            tokens = re.findall(r"[-+]?\d+", " ".join(source_records[dimens.occurrence_id]))
            valid = len(tokens) >= 3 and all(int(item) > 0 for item in tokens[:3])
            if not valid:
                issue("grid.invalid_dimens", IssueSeverity.ERROR, "grid",
                      "DIMENS must contain three positive integers",
                      allowed / dimens.location.path, dimens.location.line)
        counts = Counter(item.keyword for item in occurrences)
        payload = {
            "sections": list(sections), "files": files,
            "occurrences": [item.to_dict() for item in occurrences],
            "issues": [item.to_dict() for item in issues],
            "keyword_counts": dict(sorted(counts.items())),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        inspection_hash = hashlib.sha256(encoded).hexdigest()
        model_id = "model_" + hashlib.sha256(
            f"{project_id}:{asset_id}".encode()
        ).hexdigest()[:24]
        now = utc_now()
        old = self.database.fetch_one("SELECT created_at FROM app_models WHERE model_id=?", (model_id,))
        created = old[0] if old else now
        status = "invalid" if any(item.severity is IssueSeverity.ERROR for item in issues) else "ready"
        inspection = ModelInspection(
            model_id, project_id, asset_id, name.strip() or asset.original_name,
            "eclipse_deck", unit_system, status, asset.content_hash,
            inspection_hash, sections, tuple(files), tuple(occurrences), tuple(issues),
            created, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO app_models VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id) DO UPDATE SET
                name=excluded.name, unit_system=excluded.unit_system,
                status=excluded.status, source_hash=excluded.source_hash,
                inspection_hash=excluded.inspection_hash, summary_json=excluded.summary_json,
                updated_at=excluded.updated_at""",
                (model_id, project_id, asset_id, inspection.name, inspection.kind,
                 unit_system, status, asset.content_hash, inspection_hash,
                 json.dumps(inspection.to_dict()), created, now),
            )
        return inspection

    @staticmethod
    def _from_json(raw: str) -> ModelInspection:
        data = json.loads(raw)
        occurrences = tuple(DeckKeywordOccurrence(
            item["occurrence_id"], item["keyword"], item["section"],
            SourceLocation(**item["location"]), item["end_line"], item["record_count"],
            item["value_count"], item["defaulted_value_count"],
        ) for item in data.pop("occurrences"))
        issues = tuple(ValidationIssue(
            item["code"], item["severity"], item["category"], item["message"],
            None if item["location"] is None else SourceLocation(**item["location"]),
            item.get("details", {}),
        ) for item in data.pop("issues"))
        data["sections"] = tuple(data["sections"])
        data["files"] = tuple(data["files"])
        return ModelInspection(occurrences=occurrences, issues=issues, **data)

    def get(self, model_id: str) -> ModelInspection:
        row = self.database.fetch_one("SELECT summary_json FROM app_models WHERE model_id=?", (model_id,))
        if row is None:
            raise KeyError(f"unknown model: {model_id}")
        return self._from_json(row[0])

    def list(self, project_id: str) -> tuple[ModelInspection, ...]:
        self.projects.get(project_id)
        return tuple(self._from_json(row[0]) for row in self.database.fetch_all(
            "SELECT summary_json FROM app_models WHERE project_id=? ORDER BY created_at, model_id",
            (project_id,),
        ))

    def source_lines(
        self, model_id: str, path: str, *, start_line: int = 1, line_count: int = 200,
    ) -> dict[str, Any]:
        model = self.get(model_id)
        project = self.projects.get(model.project_id)
        base = (Path(project.root_path) / "inputs" / "original").resolve()
        target = (base / path).resolve()
        allowed = {item["path"]: item["content_hash"] for item in model.files}
        if path not in allowed or (target != base and base not in target.parents):
            raise ValueError("source path is not part of the inspected model")
        if _file_hash(target) != allowed[path]:
            raise RuntimeError("inspected source changed after indexing")
        start, count = int(start_line), int(line_count)
        if start < 1 or not 1 <= count <= 2000:
            raise ValueError("source slice must request 1-2000 lines from a positive line")
        lines = _safe_text(target, self.maximum_file_bytes)
        selected = lines[start - 1:start - 1 + count]
        return {
            "model_id": model_id, "path": path, "start_line": start,
            "end_line": start + len(selected) - 1 if selected else start - 1,
            "total_lines": len(lines), "content_hash": allowed[path], "lines": selected,
        }


__all__ = ["ModelInspectionService"]
