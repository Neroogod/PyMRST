"""Production-boundary qualification for M5.1."""

import json
from threading import Thread
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from pymrst.research_studio.studio import (
    BenchmarkPortal, IsolatedWorkflowExecutor, ProcessExecutionPolicy,
    ResearchStudioHTTPServer, ResearchStudioService, StudioMetadataStore,
    TeachingRuntime, Workflow, WorkflowProcessError, WorkflowTimeoutError,
)


OWNER = "pymrst.research_studio.studio._process_test_workflows"


def test_process_execution_success_failure_and_hard_timeout():
    executor = IsolatedWorkflowExecutor(ProcessExecutionPolicy(2))
    assert executor.execute(f"{OWNER}:echo", {"value": 7}) == {"value": 7}
    with pytest.raises(WorkflowProcessError, match="deliberate"):
        executor.execute(f"{OWNER}:fail", {"message": "deliberate"})
    started = time.monotonic()
    with pytest.raises(WorkflowTimeoutError):
        IsolatedWorkflowExecutor(ProcessExecutionPolicy(.15)).execute(
            f"{OWNER}:delay", {"seconds": 5})
    assert time.monotonic()-started < 2


def test_sqlite_metadata_survives_service_restart(tmp_path):
    database = tmp_path / "studio.sqlite"
    metadata = StudioMetadataStore(database)
    service = ResearchStudioService(max_workers=1, metadata_store=metadata)
    service.register_workflow(Workflow("echo", "Echo", "", lambda c, p: p))
    project = service.create_project("Durable")
    run = service.submit(project.project_id, "echo", {"x": 1})
    deadline = time.monotonic()+3
    while not run.status.value in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(.01)
    service.shutdown()
    reopened = StudioMetadataStore(database)
    assert reopened.get("project", project.project_id)["name"] == "Durable"
    assert reopened.get("run", run.run_id)["status"] == "completed"
    assert [event["kind"] for event in reopened.events(run.run_id)] == [
        "run_started", "run_completed"]
    reopened.close()


def test_bearer_authentication_and_malformed_json(tmp_path):
    service = ResearchStudioService(max_workers=1)
    try:
        server = ResearchStudioHTTPServer(
            ("127.0.0.1", 0), service, teaching=TeachingRuntime(),
            portal=BenchmarkPortal(()), auth_token="secret")
    except PermissionError:
        service.shutdown()
        pytest.skip("sandbox does not permit loopback socket binding")
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/snapshot"
    try:
        with pytest.raises(HTTPError) as denied:
            urlopen(url)
        assert denied.value.code == 401
        response = urlopen(Request(url, headers={"Authorization": "Bearer secret"}))
        assert json.load(response)["projects"] == []
        bad = Request(url.replace("snapshot", "projects"), data=b"{bad",
                      method="POST", headers={"Authorization": "Bearer secret",
                                              "Content-Type": "application/json"})
        with pytest.raises(HTTPError) as malformed:
            urlopen(bad)
        assert malformed.value.code == 400
    finally:
        server.shutdown(); server.server_close(); service.shutdown(); thread.join(2)


def test_remote_binding_requires_explicit_authentication():
    service = ResearchStudioService(max_workers=1)
    with pytest.raises(ValueError, match="allow_remote"):
        ResearchStudioHTTPServer(("0.0.0.0", 0), service,
                                 teaching=TeachingRuntime(), portal=BenchmarkPortal(()))
    with pytest.raises(ValueError, match="authentication"):
        ResearchStudioHTTPServer(("0.0.0.0", 0), service,
                                 teaching=TeachingRuntime(), portal=BenchmarkPortal(()),
                                 allow_remote=True)
    service.shutdown()
