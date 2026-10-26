from __future__ import annotations

import json
import sqlite3

from pymrst.app.database.models import AppDatabase
from pymrst.app.services.installation_service import EnvironmentQualificationService


def test_first_launch_qualification_persists_reproducible_evidence(tmp_path):
    workspace = tmp_path / "workspace"
    database = AppDatabase(workspace / ".pymrst-app" / "metadata.sqlite")
    service = EnvironmentQualificationService(
        database, workspace, environment={"PATH": "", "MRST_ROOT": ""},
    )
    record = service.run()
    assert record.status in {"ready", "degraded"}
    checks = {item.check_id: item for item in record.checks}
    assert checks["runtime.python"].status.value == "passed"
    assert checks["runtime.numpy"].status.value == "passed"
    assert checks["runtime.scipy"].status.value == "passed"
    assert checks["runtime.subprocess"].status.value == "passed"
    assert checks["storage.workspace"].status.value == "passed"
    assert checks["storage.sqlite"].status.value == "passed"
    assert checks["external.matlab"].required is False
    assert service.latest() == record
    assert service.history() == (record,)
    row = database.fetch_one(
        "SELECT * FROM app_environment_qualifications WHERE qualification_id=?",
        (record.qualification_id,),
    )
    assert row["content_hash"] == record.content_hash
    assert json.loads(row["platform_json"])["python"]
    database.close()


def test_low_capacity_is_visible_but_does_not_block_required_runtime(tmp_path):
    workspace = tmp_path / "workspace"
    database = AppDatabase(workspace / "metadata.sqlite")
    record = EnvironmentQualificationService(
        database, workspace, minimum_free_bytes=10**18,
    ).run()
    capacity = next(item for item in record.checks if item.check_id == "storage.capacity")
    assert capacity.status.value == "warning"
    assert capacity.remediation
    assert record.status == "degraded"
    database.close()


def test_schema_18_migrates_to_environment_qualification_registry(tmp_path):
    path = tmp_path / "metadata.sqlite"
    database = AppDatabase(path)
    database.close()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE app_environment_qualifications")
    connection.execute("PRAGMA user_version=18")
    connection.commit(); connection.close()
    migrated = AppDatabase(path)
    assert migrated.schema_version == 20
    assert migrated.fetch_one(
        "SELECT name FROM sqlite_master WHERE name='app_environment_qualifications'"
    ) is not None
    migrated.close()


def test_qualification_history_is_bounded(tmp_path):
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = EnvironmentQualificationService(database, tmp_path)
    try:
        service.history(0)
    except ValueError as error:
        assert "between 1 and 1000" in str(error)
    else:
        raise AssertionError("zero history limit should fail")
    database.close()
