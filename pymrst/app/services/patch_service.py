"""Auditable, non-destructive deck patch previews and variant materialization."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..database.models import AppDatabase
from ..schemas._common import nonempty, utc_now
from ..schemas.model import InputPatch, PatchOperation, PatchOperationKind
from .model_service import ModelInspectionService
from .project_service import ProjectService, _file_hash


def _patch_from_row(row) -> InputPatch:
    operations = tuple(PatchOperation(
        item["kind"], item["start_line"], item["end_line"], item.get("text", ""),
    ) for item in json.loads(row["operations_json"]))
    return InputPatch(
        row["patch_id"], row["project_id"], row["model_id"], row["title"],
        row["description"], row["base_hash"], row["patch_hash"], operations,
        row["status"], row["variant_relative_path"], row["variant_hash"],
        row["created_at"], row["materialized_at"],
    )


class InputPatchService:
    def __init__(
        self, database: AppDatabase, projects: ProjectService,
        models: ModelInspectionService, *, maximum_output_bytes: int = 128 * 1024**2,
    ) -> None:
        self.database, self.projects, self.models = database, projects, models
        self.maximum_output_bytes = int(maximum_output_bytes)

    @staticmethod
    def _validate_operations(
        operations: tuple[PatchOperation, ...], line_count: int,
    ) -> tuple[PatchOperation, ...]:
        if not operations:
            raise ValueError("input patch requires at least one operation")
        ordered = tuple(sorted(operations, key=lambda item: (item.start_line, item.end_line)))
        for index, item in enumerate(ordered):
            upper = line_count + 1 if item.kind in {
                PatchOperationKind.INSERT_BEFORE, PatchOperationKind.INSERT_AFTER,
            } else line_count
            if item.end_line > upper:
                raise ValueError("patch operation lies beyond source line count")
            if index and item.start_line <= ordered[index - 1].end_line:
                raise ValueError("patch operations overlap or target the same line")
        return ordered

    @staticmethod
    def _apply(lines: list[str], operations: tuple[PatchOperation, ...]) -> list[str]:
        output = list(lines)
        for operation in reversed(operations):
            start, end = operation.start_line - 1, operation.end_line
            replacement = operation.text.splitlines()
            if operation.kind is PatchOperationKind.REPLACE_LINES:
                output[start:end] = replacement
            elif operation.kind is PatchOperationKind.DELETE_LINES:
                output[start:end] = []
            elif operation.kind is PatchOperationKind.INSERT_BEFORE:
                output[start:start] = replacement
            elif operation.kind is PatchOperationKind.INSERT_AFTER:
                output[end:end] = replacement
        return output

    def create(
        self, model_id: str, title: str, operations: tuple[PatchOperation, ...],
        *, description: str = "",
    ) -> InputPatch:
        model = self.models.get(model_id)
        source = self.projects.asset_path(model.project_id, model.asset_id)
        if _file_hash(source) != model.source_hash:
            raise RuntimeError("model source no longer matches inspection lineage")
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        operations = self._validate_operations(tuple(operations), len(lines))
        canonical = json.dumps(
            [item.to_dict() for item in operations], sort_keys=True,
            separators=(",", ":"),
        )
        patch_hash = hashlib.sha256(
            (model.source_hash + "\0" + canonical).encode()
        ).hexdigest()
        patch_id = "patch_" + patch_hash[:24]
        now = utc_now()
        patch = InputPatch(
            patch_id, model.project_id, model_id, nonempty(title, "patch title"),
            str(description).strip(), model.source_hash, patch_hash, operations,
            "draft", "", "", now, "",
        )
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO app_input_patches VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(patch_id) DO NOTHING""",
                (patch.patch_id, patch.project_id, patch.model_id, patch.title,
                 patch.description, patch.base_hash, patch.patch_hash, canonical,
                 patch.status, "", "", patch.created_at, ""),
            )
        return self.get(patch_id)

    def get(self, patch_id: str) -> InputPatch:
        row = self.database.fetch_one(
            "SELECT * FROM app_input_patches WHERE patch_id=?", (str(patch_id),),
        )
        if row is None:
            raise KeyError(f"unknown input patch: {patch_id}")
        return _patch_from_row(row)

    def list(self, model_id: str) -> tuple[InputPatch, ...]:
        self.models.get(model_id)
        return tuple(_patch_from_row(row) for row in self.database.fetch_all(
            "SELECT * FROM app_input_patches WHERE model_id=? ORDER BY created_at, patch_id",
            (model_id,),
        ))

    def preview(self, patch_id: str, *, context_lines: int = 3) -> dict:
        patch = self.get(patch_id)
        model = self.models.get(patch.model_id)
        source = self.projects.asset_path(model.project_id, model.asset_id)
        if _file_hash(source) != patch.base_hash:
            raise RuntimeError("patch base source checksum changed")
        before = source.read_text(encoding="utf-8", errors="replace").splitlines()
        after = self._apply(before, patch.operations)
        encoded = ("\n".join(after) + "\n").encode()
        if len(encoded) > self.maximum_output_bytes:
            raise ValueError("patched variant exceeds configured output limit")
        diff = "\n".join(difflib.unified_diff(
            before, after, fromfile=f"original/{source.name}",
            tofile=f"variant/{source.name}", n=max(0, min(int(context_lines), 20)),
            lineterm="",
        ))
        return {
            "patch": patch.to_dict(), "diff": diff,
            "base_line_count": len(before), "variant_line_count": len(after),
            "variant_hash": hashlib.sha256(encoded).hexdigest(),
            "original_unchanged": _file_hash(source) == patch.base_hash,
        }

    def materialize(self, patch_id: str) -> InputPatch:
        patch = self.get(patch_id)
        if patch.status == "materialized":
            return patch
        preview = self.preview(patch_id)
        model = self.models.get(patch.model_id)
        project = self.projects.get(patch.project_id)
        source = self.projects.asset_path(model.project_id, model.asset_id)
        before = source.read_text(encoding="utf-8", errors="replace").splitlines()
        encoded = ("\n".join(self._apply(before, patch.operations)) + "\n").encode()
        relative = Path("variants") / patch.patch_id / source.name
        target = Path(project.root_path) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = target.parent / "lineage.json"
        if target.exists():
            if _file_hash(target) != preview["variant_hash"]:
                raise RuntimeError("existing variant conflicts with patch lineage")
        else:
            handle, temporary_name = tempfile.mkstemp(prefix=".variant-", dir=target.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o444)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        lineage = {
            "schema_version": 1, "patch_id": patch.patch_id,
            "model_id": patch.model_id, "source_asset_id": model.asset_id,
            "base_hash": patch.base_hash, "patch_hash": patch.patch_hash,
            "variant_hash": preview["variant_hash"],
            "variant_relative_path": relative.as_posix(),
            "operations": [item.to_dict() for item in patch.operations],
        }
        descriptor.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")
        os.chmod(descriptor, 0o444)
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE app_input_patches SET status='materialized',
                variant_relative_path=?, variant_hash=?, materialized_at=?
                WHERE patch_id=? AND status='draft'""",
                (relative.as_posix(), preview["variant_hash"], now, patch.patch_id),
            )
        if _file_hash(source) != patch.base_hash:
            raise RuntimeError("original source changed during variant materialization")
        return self.get(patch_id)


__all__ = ["InputPatchService"]
