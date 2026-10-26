"""Transactional project creation and immutable input import."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.navigation import default_project_navigation
from ..schemas.project import ProjectAsset, ProjectCreate, ProjectRecord


PROJECT_DIRECTORIES = (
    "inputs/original", "models", "variants", "runs", "reports", "experiments",
    "investigations", ".pymrst",
)


def _project_from_row(row) -> ProjectRecord:
    return ProjectRecord(
        row["project_id"], row["name"], row["description"],
        tuple(json.loads(row["tags_json"])), row["root_path"], row["content_hash"],
        row["created_at"], row["updated_at"], row["schema_version"],
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProjectService:
    def __init__(
        self, database: AppDatabase, workspace_root: str | Path,
        *, maximum_asset_bytes: int = 10 * 1024**3,
    ) -> None:
        self.database = database
        self.workspace_root = Path(workspace_root).resolve()
        self.projects_root = self.workspace_root / "projects"
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.maximum_asset_bytes = int(maximum_asset_bytes)
        if self.maximum_asset_bytes <= 0:
            raise ValueError("maximum project asset size must be positive")

    @staticmethod
    def _manifest(project_id: str, request: ProjectCreate) -> dict:
        return {
            "schema_version": 1, "project_id": project_id,
            "name": request.name, "description": request.description,
            "tags": list(request.tags), "input_policy": "immutable_originals",
        }

    def create(self, request: ProjectCreate) -> ProjectRecord:
        if not isinstance(request, ProjectCreate):
            raise TypeError("project creation requires ProjectCreate")
        project_id = f"project_{uuid4().hex}"
        target = self.projects_root / project_id
        staging = Path(tempfile.mkdtemp(
            prefix=f".{project_id}.staging-", dir=self.projects_root,
        ))
        created_at = utc_now()
        manifest = self._manifest(project_id, request)
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        content_hash = hashlib.sha256(encoded).hexdigest()
        try:
            for relative in PROJECT_DIRECTORIES:
                (staging / relative).mkdir(parents=True, exist_ok=True)
            (staging / "project.json").write_bytes(encoded)
            staging.replace(target)
            record = ProjectRecord(
                project_id, request.name, request.description, request.tags,
                str(target), content_hash, created_at, created_at,
            )
            try:
                with self.database.transaction() as connection:
                    connection.execute(
                        "INSERT INTO app_projects VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            record.project_id, record.name, record.description,
                            json.dumps(record.tags), record.root_path,
                            record.content_hash, record.created_at, record.updated_at,
                            record.schema_version,
                        ),
                    )
            except Exception:
                shutil.rmtree(target)
                raise
            return record
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get(self, project_id: str) -> ProjectRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_projects WHERE project_id=?", (str(project_id),),
        )
        if row is None:
            raise KeyError(f"unknown project: {project_id}")
        record = _project_from_row(row)
        if not Path(record.root_path).is_dir():
            raise RuntimeError(f"project directory is missing: {record.root_path}")
        return record

    def list(self) -> tuple[ProjectRecord, ...]:
        return tuple(_project_from_row(row) for row in self.database.fetch_all(
            "SELECT * FROM app_projects ORDER BY name COLLATE NOCASE, project_id"
        ))

    def navigation(self, project_id: str):
        self.get(project_id)
        return default_project_navigation(project_id)

    def import_input(self, project_id: str, source: str | Path, *, kind="deck") -> ProjectAsset:
        project = self.get(project_id)
        source = Path(source).resolve()
        if source.is_symlink() or not source.is_file():
            raise ValueError("project inputs must be regular non-symlink files")
        size = source.stat().st_size
        if size > self.maximum_asset_bytes:
            raise ValueError("project input exceeds configured size limit")
        content_hash = _file_hash(source)
        existing = self.database.fetch_one(
            "SELECT * FROM app_assets WHERE project_id=? AND content_hash=?",
            (project_id, content_hash),
        )
        if existing is not None:
            return ProjectAsset(**dict(existing))
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name).strip("._") or "input"
        relative = Path("inputs/original") / f"{content_hash[:16]}_{safe_name}"
        target = Path(project.root_path) / relative
        temporary = target.with_suffix(target.suffix + f".tmp-{uuid4().hex}")
        asset = ProjectAsset(
            "asset_" + hashlib.sha256(
                f"{project_id}:{content_hash}".encode()
            ).hexdigest()[:24],
            project_id, str(kind), source.name,
            relative.as_posix(), content_hash, size, utc_now(),
        )
        try:
            shutil.copyfile(source, temporary)
            if _file_hash(temporary) != content_hash:
                raise IOError("project input changed while it was being imported")
            os.chmod(temporary, 0o444)
            temporary.replace(target)
            try:
                with self.database.transaction() as connection:
                    connection.execute(
                        "INSERT INTO app_assets VALUES(?,?,?,?,?,?,?,?)",
                        (
                            asset.asset_id, asset.project_id, asset.kind,
                            asset.original_name, asset.relative_path,
                            asset.content_hash, asset.size_bytes, asset.created_at,
                        ),
                    )
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return asset
        finally:
            temporary.unlink(missing_ok=True)

    def assets(self, project_id: str) -> tuple[ProjectAsset, ...]:
        self.get(project_id)
        return tuple(ProjectAsset(**dict(row)) for row in self.database.fetch_all(
            "SELECT * FROM app_assets WHERE project_id=? ORDER BY created_at, asset_id",
            (project_id,),
        ))

    def asset(self, project_id: str, asset_id: str) -> ProjectAsset:
        self.get(project_id)
        row = self.database.fetch_one(
            "SELECT * FROM app_assets WHERE project_id=? AND asset_id=?",
            (str(project_id), str(asset_id)),
        )
        if row is None:
            raise KeyError(f"unknown project asset: {asset_id}")
        return ProjectAsset(**dict(row))

    def asset_path(self, project_id: str, asset_id: str, *, verify=True) -> Path:
        project = self.get(project_id)
        asset = self.asset(project_id, asset_id)
        root = Path(project.root_path).resolve()
        path = (root / asset.relative_path).resolve()
        if root not in path.parents or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"project asset path is unsafe or missing: {asset_id}")
        if verify and _file_hash(path) != asset.content_hash:
            raise RuntimeError(f"immutable project asset failed checksum: {asset_id}")
        return path


__all__ = ["PROJECT_DIRECTORIES", "ProjectService"]
