"""M5.2--M5.5 notebook, HTTP, publication and teaching boundaries."""

import json
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from pymrst.research_studio.studio import (
    BenchmarkEntry, BenchmarkPortal, ResearchStudioHTTPServer,
    ResearchStudioService, RunView, TeachingLimitExceeded, TeachingLimits,
    TeachingRuntime, Workflow, studio_page,
)


def _service():
    service = ResearchStudioService(max_workers=1)
    service.register_workflow(Workflow(
        "echo", "Echo", "Return inputs",
        lambda context, parameters: parameters, True))
    return service


def test_notebook_html_escapes_untrusted_run_content():
    service = _service(); project = service.create_project("P")
    record = service.submit(project.project_id, "echo", {"x": 1})
    record.message = '<script>alert("x")</script>'
    rendered = RunView(record)._repr_html_()
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    service.shutdown()


def test_benchmark_portal_never_exposes_internal_evidence_publicly(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"acceptance": {"passed": True}}))
    entry = BenchmarkEntry.from_path("spe1", "SPE1", "1", evidence)
    portal = BenchmarkPortal((entry,))
    assert portal.entries(public_only=True) == ()
    with pytest.raises(PermissionError, match="not public"):
        portal.evidence("spe1", "1", public_request=True)
    with pytest.raises(PermissionError, match="explicit approval"):
        portal.promote("spe1", "1", "public")
    portal.promote("spe1", "1", "public", explicit_approval=True)
    assert portal.evidence("spe1", "1", public_request=True)["acceptance"]["passed"]
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        portal.evidence("spe1", "1")


def test_teaching_runtime_enforces_limits_and_discloses_no_field_scale_claim():
    runtime = TeachingRuntime(TeachingLimits(
        maximum_cells=100, maximum_report_steps=5, maximum_members=2,
        maximum_wall_seconds=1, maximum_result_bytes=1000))
    assert runtime.run(lambda: 42, cells=20, report_steps=2) == 42
    with pytest.raises(TeachingLimitExceeded, match="cells=101"):
        runtime.run(lambda: None, cells=101, report_steps=2)
    assert runtime.disclosure()["field_scale_claim"] is False


def test_loopback_http_api_and_workspace_use_service_data(tmp_path):
    service = _service(); teaching = TeachingRuntime()
    portal = BenchmarkPortal()
    try:
        server = ResearchStudioHTTPServer(("127.0.0.1", 0), service,
                                          teaching=teaching, portal=portal)
    except PermissionError:
        service.shutdown()
        pytest.skip("test sandbox prohibits loopback sockets")
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urlopen(base+"/").read().decode()
        assert "PyMRST Research Studio" in page and "field-scale browser solver" in page
        request = Request(base+"/api/projects", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"name": "Demo"}).encode())
        project = json.load(urlopen(request))
        snapshot = json.load(urlopen(base+"/api/snapshot"))
        assert snapshot["projects"][0]["project_id"] == project["project_id"]
        assert json.load(urlopen(base+"/api/benchmarks")) == []
    finally:
        server.shutdown(); server.server_close(); service.shutdown(); thread.join()


def test_server_refuses_non_loopback_binding():
    with pytest.raises(ValueError, match="loopback only"):
        ResearchStudioHTTPServer(("0.0.0.0", 0), _service(),
                                 teaching=TeachingRuntime(), portal=BenchmarkPortal())
