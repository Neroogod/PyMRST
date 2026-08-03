"""M5.1 service concurrency, events, cancellation, resume and isolation."""

import time

from pymrst.research_studio.studio import ResearchStudioService, RunStatus, Workflow


def _workflow(context, parameters):
    total = int(parameters.get("steps", 3))
    value = float(context.parent_result["value"] if context.parent_result else 0.0)
    for step in range(total):
        time.sleep(.005)
        context.emit("step", (step+1)/total, f"step {step+1}", {"step": step})
        value += float(parameters.get("increment", 1.0))
    return {"value": value, "steps": total}


def _wait(service, run_id):
    for _ in range(500):
        record = service.run(run_id)
        if record.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return record
        time.sleep(.005)
    raise AssertionError("run did not finish")


def test_projects_concurrent_runs_events_and_results_are_isolated():
    service = ResearchStudioService(max_workers=2)
    service.register_workflow(Workflow("manufactured", "Manufactured", "test", _workflow, True))
    project = service.create_project("Waterflood", tags=("teaching",))
    first = service.submit(project.project_id, "manufactured", {"steps": 3, "increment": 2})
    second = service.submit(project.project_id, "manufactured", {"steps": 2, "increment": 5})
    a, b = _wait(service, first.run_id), _wait(service, second.run_id)
    assert a.result == {"value": 6.0, "steps": 3}
    assert b.result == {"value": 10.0, "steps": 2}
    assert [event["kind"] for event in a.events] == [
        "run_started", "step", "step", "step", "run_completed"]
    assert service.snapshot()["projects"][0]["name"] == "Waterflood"
    resource = service.publish_resource(
        "evidence", "mass-balance", {"passed": True}, run_id=a.run_id)
    assert service.resource("evidence", "mass-balance") == resource
    assert service.resources("evidence") == (resource,)
    service.shutdown()


def test_cooperative_cancellation_and_parent_result_resume():
    service = ResearchStudioService(max_workers=1)
    service.register_workflow(Workflow("manufactured", "Manufactured", "test", _workflow, True))
    project = service.create_project("P")
    cancelled = service.submit(project.project_id, "manufactured", {"steps": 40})
    time.sleep(.02); service.cancel(cancelled.run_id)
    assert _wait(service, cancelled.run_id).status == RunStatus.CANCELLED
    base = service.submit(project.project_id, "manufactured", {"steps": 1, "increment": 3})
    assert _wait(service, base.run_id).result["value"] == 3
    child = service.resume(base.run_id, parameters={"increment": 4})
    child = _wait(service, child.run_id)
    assert child.parent_run_id == base.run_id and child.result["value"] == 7
    service.shutdown()


def test_failure_is_terminal_and_does_not_break_later_runs():
    def fail(context, parameters):
        raise RuntimeError("controlled failure")
    service = ResearchStudioService(max_workers=1)
    service.register_workflow(Workflow("fail", "Fail", "test", fail))
    service.register_workflow(Workflow("ok", "OK", "test", _workflow))
    project = service.create_project("P")
    bad = _wait(service, service.submit(project.project_id, "fail").run_id)
    good = _wait(service, service.submit(project.project_id, "ok", {"steps": 1}).run_id)
    assert bad.status == RunStatus.FAILED and bad.error_type == "RuntimeError"
    assert good.status == RunStatus.COMPLETED
    service.shutdown()
