"""Deterministic installed-environment and first-launch qualification."""

from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.installation import (
    EnvironmentCheck, EnvironmentQualification, QualificationCheckStatus as Status,
)


def _encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value))
    return tuple(int(part or 0) for part in match.groups(default="0")) if match else ()


def _record(row) -> EnvironmentQualification:
    return EnvironmentQualification(
        row["qualification_id"], row["status"], row["content_hash"],
        json.loads(row["platform_json"]),
        tuple(EnvironmentCheck(
            item["check_id"], item["category"], item["title"], item["status"],
            item["required"], item["message"], item["details"], item["remediation"],
        ) for item in json.loads(row["checks_json"])),
        row["created_at"],
    )


class EnvironmentQualificationService:
    def __init__(
        self, database: AppDatabase, workspace_root: str | Path,
        *, environment: Mapping[str, str] | None = None,
        minimum_free_bytes: int = 2 * 1024**3,
    ) -> None:
        self.database = database
        self.workspace_root = Path(workspace_root).resolve()
        self.environment = os.environ if environment is None else environment
        self.minimum_free_bytes = int(minimum_free_bytes)
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum free bytes cannot be negative")

    @staticmethod
    def _check(
        check_id: str, category: str, title: str, status: Status,
        message: str, *, required: bool = False, details=None, remediation: str = "",
    ) -> EnvironmentCheck:
        return EnvironmentCheck(
            check_id, category, title, status, required, message,
            {} if details is None else details, remediation,
        )

    def _runtime_checks(self) -> list[EnvironmentCheck]:
        checks = []
        python_ok = sys.version_info >= (3, 11)
        checks.append(self._check(
            "runtime.python", "runtime", "Python runtime",
            Status.PASSED if python_ok else Status.FAILED,
            f"Python {platform.python_version()} on {platform.machine() or 'unknown architecture'}",
            required=True,
            details={"version": platform.python_version(), "executable": sys.executable,
                     "implementation": platform.python_implementation(),
                     "architecture": platform.machine()},
            remediation="Install Python 3.11 or newer." if not python_ok else "",
        ))
        for package, minimum in (("numpy", "1.26"), ("scipy", "1.11")):
            installed = _version(package)
            okay = bool(installed) and _version_tuple(installed) >= _version_tuple(minimum)
            checks.append(self._check(
                f"runtime.{package}", "runtime", package.capitalize(),
                Status.PASSED if okay else Status.FAILED,
                f"{package} {installed or 'not installed'}",
                required=True, details={"installed": installed, "minimum": minimum},
                remediation=f"Install {package}>={minimum}." if not okay else "",
            ))
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import sys; print(sys.version_info.major)"],
                capture_output=True, text=True, timeout=10, check=False,
                env={"PATH": self.environment.get("PATH", "")},
            )
            okay = result.returncode == 0 and result.stdout.strip() == str(sys.version_info.major)
            detail = {"return_code": result.returncode}
        except (OSError, subprocess.TimeoutExpired) as error:
            okay, detail = False, {"error_type": type(error).__name__}
        checks.append(self._check(
            "runtime.subprocess", "runtime", "Worker-process launch",
            Status.PASSED if okay else Status.FAILED,
            "Isolated Python worker launch passed" if okay else "Isolated worker launch failed",
            required=True, details=detail,
            remediation="Permit the managed Python executable to create child processes." if not okay else "",
        ))
        return checks

    def _storage_checks(self) -> list[EnvironmentCheck]:
        checks = []
        diagnostic_root = self.workspace_root / ".pymrst-app" / "diagnostics"
        try:
            diagnostic_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=diagnostic_root, prefix="write-probe-", delete=False) as stream:
                probe = Path(stream.name)
                stream.write(b"pymrst-write-probe")
                stream.flush()
                os.fsync(stream.fileno())
            okay = probe.read_bytes() == b"pymrst-write-probe"
            probe.unlink(missing_ok=True)
            detail = {"path": str(diagnostic_root)}
        except OSError as error:
            okay, detail = False, {"path": str(diagnostic_root), "error_type": type(error).__name__}
        checks.append(self._check(
            "storage.workspace", "storage", "Workspace durability",
            Status.PASSED if okay else Status.FAILED,
            "Atomic workspace writes are available" if okay else "Workspace is not safely writable",
            required=True, details=detail,
            remediation="Choose a writable local workspace with durable storage." if not okay else "",
        ))
        usage = shutil.disk_usage(self.workspace_root)
        enough = usage.free >= self.minimum_free_bytes
        checks.append(self._check(
            "storage.capacity", "storage", "Free workspace capacity",
            Status.PASSED if enough else Status.WARNING,
            f"{usage.free / 1024**3:.2f} GiB free",
            details={"free_bytes": usage.free, "total_bytes": usage.total,
                     "recommended_minimum_bytes": self.minimum_free_bytes},
            remediation="Free disk space or move the workspace before a field-scale run." if not enough else "",
        ))
        integrity = self.database.fetch_one("PRAGMA integrity_check")[0]
        checks.append(self._check(
            "storage.sqlite", "storage", "Metadata database integrity",
            Status.PASSED if integrity == "ok" else Status.FAILED,
            "SQLite integrity check passed" if integrity == "ok" else f"SQLite reported {integrity}",
            required=True, details={"result": integrity, "schema_version": self.database.schema_version},
            remediation="Restore the workspace from a verified backup." if integrity != "ok" else "",
        ))
        return checks

    def _capability_checks(self) -> list[EnvironmentCheck]:
        groups = {
            "application": ("fastapi", "uvicorn"),
            "state_storage": ("h5py", "zarr", "xarray"),
            "tables": ("pandas", "pyarrow"),
            "visualization": ("plotly", "pyvista"),
            "acceleration": ("numba", "torch", "cupy"),
            "hpc": ("mpi4py",),
        }
        checks = []
        for capability, packages in groups.items():
            found = {name: _version(name) for name in packages if util.find_spec(name) is not None}
            checks.append(self._check(
                f"capability.{capability}", "capability",
                capability.replace("_", " ").title(),
                Status.PASSED if found else Status.WARNING,
                ", ".join(f"{name} {version or 'available'}" for name, version in found.items())
                if found else "Optional capability is not installed",
                details={"available": found, "candidates": list(packages)},
                remediation=(
                    "Install the matching PyMRST optional dependency profile when needed."
                    if not found else ""
                ),
            ))
        return checks

    def _external_checks(self) -> list[EnvironmentCheck]:
        matlab_candidates = []
        explicit_matlab = str(self.environment.get("MATLAB_EXECUTABLE", "")).strip()
        if explicit_matlab:
            matlab_candidates.append(Path(explicit_matlab).expanduser())
        located = shutil.which("matlab", path=self.environment.get("PATH"))
        if located:
            matlab_candidates.append(Path(located))
        if sys.platform == "darwin":
            matlab_candidates.extend(sorted(Path("/Applications").glob(
                "MATLAB_R*.app/bin/matlab"
            ), reverse=True))
        matlab = next((item.resolve() for item in matlab_candidates if item.is_file()), None)
        matlab_check = self._check(
            "external.matlab", "external", "MATLAB reference runtime",
            Status.PASSED if matlab else Status.WARNING,
            f"MATLAB found at {matlab}" if matlab else "MATLAB was not discovered",
            details={"path": str(matlab) if matlab else ""},
            remediation="Set MATLAB_EXECUTABLE to enable direct MRST qualification." if not matlab else "",
        )
        mrst_candidates = []
        explicit_mrst = str(self.environment.get("MRST_ROOT", "")).strip()
        if explicit_mrst:
            mrst_candidates.append(Path(explicit_mrst).expanduser())
        mrst_candidates.extend((
            self.workspace_root.parent / "mrst-2024a-Geomech",
            self.workspace_root / "mrst-2024a-Geomech",
        ))
        mrst = next((item.resolve() for item in mrst_candidates
                     if item.is_dir() and (item / "modules").is_dir()), None)
        mrst_check = self._check(
            "external.mrst", "external", "MRST reference source",
            Status.PASSED if mrst else Status.WARNING,
            f"MRST found at {mrst}" if mrst else "MRST source was not discovered",
            details={"path": str(mrst) if mrst else ""},
            remediation="Set MRST_ROOT to enable MATLAB parity workflows." if not mrst else "",
        )
        return [matlab_check, mrst_check]

    def run(self) -> EnvironmentQualification:
        checks = tuple(
            self._runtime_checks() + self._storage_checks()
            + self._capability_checks() + self._external_checks()
        )
        if any(item.required and item.status is Status.FAILED for item in checks):
            status = "blocked"
        elif any(item.status is not Status.PASSED for item in checks):
            status = "degraded"
        else:
            status = "ready"
        platform_record = {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "pymrst": _version("pymrst") or "source-tree",
        }
        body = {"status": status, "platform": platform_record,
                "checks": [item.to_dict() for item in checks]}
        digest, now = hashlib.sha256(_encoded(body).encode()).hexdigest(), utc_now()
        record = EnvironmentQualification(
            f"environment_{uuid4().hex}", status, digest, platform_record, checks, now,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_environment_qualifications VALUES(?,?,?,?,?,?)",
                (record.qualification_id, record.status, record.content_hash,
                 _encoded(record.platform), _encoded([item.to_dict() for item in checks]), now),
            )
        return record

    def latest(self) -> EnvironmentQualification | None:
        row = self.database.fetch_one(
            "SELECT * FROM app_environment_qualifications ORDER BY created_at DESC LIMIT 1"
        )
        return None if row is None else _record(row)

    def history(self, limit: int = 20) -> tuple[EnvironmentQualification, ...]:
        limit = int(limit)
        if not 1 <= limit <= 1000:
            raise ValueError("qualification history limit must be between 1 and 1000")
        return tuple(_record(row) for row in self.database.fetch_all(
            "SELECT * FROM app_environment_qualifications ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ))


__all__ = ["EnvironmentQualificationService"]
