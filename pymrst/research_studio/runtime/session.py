"""Observation-only session adapter for the canonical schedule solver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Any
import math

from pymrst.research_studio.evidence.event_bus import EventBus
from pymrst.research_studio.evidence.events import EventKind, SimulationEvent
from pymrst.research_studio.evidence.report import RunIdentity, SimulationEvidence
from pymrst.research_studio.evidence.conservation import conservation_report
from pymrst.research_studio.evidence.invariants import InvariantViolation
from pymrst.research_studio.evidence.failure_analysis import analyze_simulation_evidence
from pymrst.research_studio.evidence.manifest import capture_manifest
from pymrst.research_studio.evidence.failure_bundle import write_store_failure_bundle
from pymrst.research_studio.runtime.cancellation import CancellationToken
from pymrst.research_studio.runtime.identifiers import (
    AttemptId, CaseId, MinistepId, ReportStepId, RunId,
)
from pymrst.research_studio.runtime.protocols import ScheduleSolver
from pymrst.research_studio.storage.checkpoint import (
    CheckpointRecord, default_model_fingerprint, default_solver_fingerprint,
    restart_payload, schedule_fingerprint,
)
from pymrst.research_studio.runtime.identifiers import CheckpointId


class SimulationStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SimulationResult:
    final_state: Any
    states: Any
    report: Any
    evidence: SimulationEvidence
    status: SimulationStatus
    events: tuple[SimulationEvent, ...]
    store: Any = None


class _EventSequence:
    def __init__(self, run_id: RunId, bus: EventBus, started: float, durable_sink=None):
        self.run_id = run_id
        self.bus = bus
        self.started = started
        self.value = 0
        self.events: list[SimulationEvent] = []
        self.durable_sink = durable_sink

    def emit(self, kind: EventKind, **kwargs) -> SimulationEvent:
        event = SimulationEvent(
            kind=kind,
            run_id=self.run_id,
            sequence=self.value,
            elapsed_seconds=max(0.0, monotonic() - self.started),
            **kwargs,
        )
        self.value += 1
        if self.durable_sink is not None:
            self.durable_sink(event)
        self.events.append(event)
        self.bus.publish(event)
        return event


class SimulationSession:
    """Run an existing schedule solver and derive evidence without intervention.

    Version 1 deliberately invokes ``solver.simulate_schedule`` exactly once.
    It does not wrap Newton iterations or insert cancellation checks inside a
    timestep. Fine-grained live hooks will only be added after before/after
    numerical characterization proves they are observational.
    """

    def __init__(
        self,
        *,
        model: Any,
        initial_state: Any,
        schedule: Any,
        nonlinear_solver: ScheduleSolver,
        case_id: CaseId | str = "case_unspecified",
        run_id: RunId | str | None = None,
        event_bus: EventBus | None = None,
        cancellation: CancellationToken | None = None,
        evidence_profile: str = "observe",
        result_store: Any = None,
        checkpoint_every_report: bool = False,
        code_fingerprint: str | None = None,
        checkpoint_schedule: Any = None,
        live_observation: bool = False,
        streaming_execution: bool = False,
        conservation_adapter: Any = None,
        invariant_suite: Any = None,
        boundary_rate_provider: Any = None,
        source_rate_provider: Any = None,
        input_files=(),
        command=None,
        failure_bundle_path=None,
    ) -> None:
        if evidence_profile not in {"observe", "strict", "reference"}:
            raise ValueError("unknown evidence_profile")
        self.model = model
        self.initial_state = initial_state
        self.schedule = schedule
        self.nonlinear_solver = nonlinear_solver
        self.run_id = RunId.generate() if run_id is None else RunId(run_id)
        self.case_id = CaseId(case_id)
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.cancellation = cancellation or CancellationToken()
        self.evidence_profile = evidence_profile
        self.result_store = result_store
        self.checkpoint_every_report = bool(checkpoint_every_report)
        self.code_fingerprint = code_fingerprint
        self._checkpoint_schedule = (
            schedule if checkpoint_schedule is None else checkpoint_schedule
        )
        self._report_offset = 0
        self._elapsed_offset = 0.0
        self._lineage: dict[str, Any] = {}
        self.live_observation = bool(live_observation)
        self.streaming_execution = bool(streaming_execution)
        self.conservation_adapter = conservation_adapter
        self.invariant_suite = invariant_suite
        self.boundary_rate_provider = boundary_rate_provider
        self.source_rate_provider = source_rate_provider
        self.failure_bundle_path = failure_bundle_path
        self.reproducibility_manifest = (
            capture_manifest(
                self.run_id, code_fingerprint=code_fingerprint,
                inputs=input_files, command=command,
                configuration={"evidence_profile": evidence_profile},
            ) if code_fingerprint else None
        )
        if failure_bundle_path is not None:
            if result_store is None or not checkpoint_every_report:
                raise ValueError(
                    "failure_bundle_path requires result_store and checkpoint_every_report"
                )
            if self.reproducibility_manifest is None:
                raise ValueError("failure_bundle_path requires code_fingerprint")
        if self.streaming_execution and not hasattr(nonlinear_solver, "iter_schedule"):
            raise TypeError("streaming_execution requires solver.iter_schedule")
        if (evidence_profile == "strict" and invariant_suite is not None
                and not self.streaming_execution):
            raise ValueError(
                "strict invariants require streaming_execution so the accepted "
                "boundary is committed before validation can stop the run"
            )
        if self.checkpoint_every_report:
            if result_store is None:
                raise ValueError("checkpoint_every_report requires result_store")
            if not code_fingerprint:
                raise ValueError("checkpoint_every_report requires code_fingerprint")
            # Fail early for structured schedules until control-state v2 exists.
            schedule_fingerprint(self._checkpoint_schedule)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        source_store: Any,
        source_run_id: RunId | str,
        checkpoint_id: CheckpointId | str,
        model: Any,
        full_schedule: Any,
        nonlinear_solver: ScheduleSolver,
        code_fingerprint: str,
        state_factory=dict,
        **session_options,
    ) -> "SimulationSession":
        record = CheckpointRecord.from_dict(
            source_store.load_checkpoint(
                RunId(source_run_id), CheckpointId(checkpoint_id),
            )
        )
        model_fp = default_model_fingerprint(model)
        solver_fp = default_solver_fingerprint(nonlinear_solver)
        state, remaining = restart_payload(
            record,
            schedule=full_schedule,
            model_fingerprint=model_fp,
            solver_fingerprint=solver_fp,
            code_fingerprint=code_fingerprint,
            state_factory=state_factory,
        )
        session = cls(
            model=model,
            initial_state=state,
            schedule=remaining,
            nonlinear_solver=nonlinear_solver,
            code_fingerprint=code_fingerprint,
            checkpoint_schedule=full_schedule,
            **session_options,
        )
        session._report_offset = record.completed_report_steps
        session._elapsed_offset = record.accepted_elapsed_time
        session._lineage = {
            "source_run_id": str(record.source_run_id),
            "source_checkpoint_id": str(record.checkpoint_id),
        }
        if record.control_index is not None:
            if not hasattr(full_schedule, "controls"):
                raise ValueError("checkpoint contains control state for a bare schedule")
            # Restore the model-side control object before the remaining
            # schedule is handed to the native solver. The solver will apply
            # the first resumed control normally; matching old/new controls
            # prevents false injector reseeding at the restart boundary.
            apply_control = getattr(nonlinear_solver, "_apply_control", None)
            if apply_control is None:
                raise TypeError("solver cannot restore structured control state")
            apply_control(model, full_schedule.controls[record.control_index])
        return session

    def run(self) -> SimulationResult:
        # Version 1 supports cancellation before numerical ownership transfers
        # to the native solver. Check before creating any persistent run.
        self.cancellation.raise_if_requested()
        started = monotonic()
        evidence = SimulationEvidence(
            RunIdentity(self.run_id, self.case_id),
            manifest={
                "evidence_profile": self.evidence_profile,
                "adapter": "canonical_schedule_v1",
                "lineage": self._lineage,
                "reproducibility": (
                    self.reproducibility_manifest.to_dict()
                    if self.reproducibility_manifest is not None else None
                ),
            },
        )
        if self.result_store is not None:
            self.result_store.begin_run(self.run_id, evidence.manifest)
        durable_sink = (
            (lambda event: self.result_store.commit_event(self.run_id, event))
            if self.result_store is not None
            and hasattr(self.result_store, "commit_event") else None
        )
        stream = _EventSequence(
            self.run_id, self.event_bus, started, durable_sink=durable_sink,
        )
        stream.emit(EventKind.RUN_STARTED, payload={"case_id": self.case_id})
        if self.failure_bundle_path is not None:
            self._commit_initial_checkpoint(evidence, stream)
        previous_observer = getattr(self.nonlinear_solver, "observer", None)
        observer_installed = False
        if self.live_observation and hasattr(self.nonlinear_solver, "observer"):
            self.nonlinear_solver.observer = self._live_observer(
                stream, evidence, previous_observer,
            )
            observer_installed = True
        try:
            if self.streaming_execution:
                states, report, cancelled = self._run_streaming(
                    evidence, stream,
                )
            else:
                states, report = self.nonlinear_solver.simulate_schedule(
                    self.model, self.initial_state, self.schedule,
                )
                cancelled = False
        except Exception as error:
            evidence.failures.append({
                "type": type(error).__name__, "message": str(error),
            })
            stream.emit(
                EventKind.RUN_FAILED,
                payload={"type": type(error).__name__, "message": str(error)},
            )
            if self.failure_bundle_path is not None:
                bundle = write_store_failure_bundle(
                    self.failure_bundle_path, store=self.result_store,
                    run_id=self.run_id, manifest=self.reproducibility_manifest,
                    evidence=evidence,
                    failing_step={
                        "reason": "solver_exception",
                        "type": type(error).__name__, "message": str(error),
                    },
                )
                stream.emit(
                    EventKind.FAILURE_BUNDLE_WRITTEN,
                    payload={"path": str(bundle)},
                )
            evidence.timings["wall_seconds"] = max(0.0, monotonic() - started)
            if self.result_store is not None:
                self.result_store.commit_evidence(self.run_id, evidence)
                self.result_store.finalize_run(self.run_id, "failed")
            raise
        finally:
            if observer_installed:
                self.nonlinear_solver.observer = previous_observer

        self._translate_report(
            report, evidence, stream,
            include_solver_events=not observer_installed,
            report_offset=self._report_offset,
        )
        if not self.streaming_execution:
            previous_state = self.initial_state
            for state, step in zip(states, getattr(report, "steps", ())):
                if step.converged:
                    self._record_step_evidence(
                        previous_state, state, step, evidence, stream,
                    )
                    previous_state = state
        converged = bool(getattr(report, "converged", True))
        status = (
            SimulationStatus.CANCELLED if cancelled else
            SimulationStatus.COMPLETED if converged else SimulationStatus.FAILED
        )
        if status is SimulationStatus.FAILED:
            analysis = analyze_simulation_evidence(evidence, converged=False)
            evidence.failures.append({
                "type": "NumericalFailureAnalysis",
                "analysis": analysis.to_dict(),
            })
        final_state = states[-1] if len(states) else self.initial_state
        if self.result_store is not None and not self.streaming_execution:
            steps = list(getattr(report, "steps", ()))
            elapsed = self._elapsed_offset
            full_schedule_fp = (
                schedule_fingerprint(self._checkpoint_schedule)
                if self.checkpoint_every_report else None
            )
            for index, state in enumerate(states):
                accepted = index < len(steps) and bool(steps[index].converged)
                self.result_store.commit_state(
                    self.run_id, index, state, accepted=accepted,
                )
                if accepted:
                    elapsed += float(steps[index].dt)
                if accepted and self.checkpoint_every_report:
                    completed = self._report_offset + index + 1
                    checkpoint_id = CheckpointId(f"checkpoint_{completed:08d}")
                    record = CheckpointRecord(
                        checkpoint_id=checkpoint_id,
                        source_run_id=self.run_id,
                        completed_report_steps=completed,
                        accepted_elapsed_time=elapsed,
                        state=state,
                        schedule_fingerprint=full_schedule_fp,
                        model_fingerprint=default_model_fingerprint(self.model),
                        solver_fingerprint=default_solver_fingerprint(self.nonlinear_solver),
                        code_fingerprint=self.code_fingerprint,
                        control_index=(
                            int(self._checkpoint_schedule.step_control[completed-1])
                            if hasattr(self._checkpoint_schedule, "step_control")
                            else None
                        ),
                    )
                    self.result_store.commit_checkpoint(
                        self.run_id, checkpoint_id, record.to_dict(),
                    )
                    evidence.checkpoints.append({
                        "checkpoint_id": str(checkpoint_id),
                        "completed_report_steps": completed,
                        "accepted_elapsed_time": elapsed,
                    })
                    stream.emit(
                        EventKind.CHECKPOINT_COMMITTED,
                        payload={"checkpoint_id": str(checkpoint_id)},
                    )
        if status is SimulationStatus.FAILED and self.failure_bundle_path is not None:
            failing_step = (
                evidence.timestep_reports[-1]
                if evidence.timestep_reports else {"reason": "solver_exception"}
            )
            bundle = write_store_failure_bundle(
                self.failure_bundle_path, store=self.result_store,
                run_id=self.run_id, manifest=self.reproducibility_manifest,
                evidence=evidence, failing_step=failing_step,
            )
            stream.emit(
                EventKind.FAILURE_BUNDLE_WRITTEN,
                payload={"path": str(bundle)},
            )
        terminal = (
            EventKind.RUN_CANCELLED if cancelled else
            EventKind.RUN_COMPLETED if converged else EventKind.RUN_FAILED
        )
        stream.emit(terminal, payload={"report_steps": len(getattr(report, "steps", ()))})
        evidence.timings["wall_seconds"] = max(0.0, monotonic() - started)
        if self.result_store is not None:
            self.result_store.commit_evidence(self.run_id, evidence)
            self.result_store.finalize_run(self.run_id, status.value)
        return SimulationResult(
            final_state=final_state,
            states=states,
            report=report,
            evidence=evidence,
            status=status,
            events=tuple(stream.events),
            store=self.result_store,
        )

    def _commit_initial_checkpoint(self, evidence, stream):
        completed = self._report_offset
        checkpoint_id = CheckpointId(f"checkpoint_{completed:08d}")
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id, source_run_id=self.run_id,
            completed_report_steps=completed,
            accepted_elapsed_time=self._elapsed_offset,
            state=self.initial_state,
            schedule_fingerprint=schedule_fingerprint(self._checkpoint_schedule),
            model_fingerprint=default_model_fingerprint(self.model),
            solver_fingerprint=default_solver_fingerprint(self.nonlinear_solver),
            code_fingerprint=self.code_fingerprint,
            control_index=(
                int(self._checkpoint_schedule.step_control[completed-1])
                if completed and hasattr(self._checkpoint_schedule, "step_control")
                else None
            ),
        )
        self.result_store.commit_checkpoint(
            self.run_id, checkpoint_id, record.to_dict(),
        )
        evidence.checkpoints.append({
            "checkpoint_id": str(checkpoint_id),
            "completed_report_steps": completed,
            "accepted_elapsed_time": self._elapsed_offset,
        })
        stream.emit(
            EventKind.CHECKPOINT_COMMITTED,
            payload={"checkpoint_id": str(checkpoint_id), "initial": True},
        )

    def _run_streaming(self, evidence, stream):
        from pymrst.modules.ad_core.simulators.nonlinear_solver import SimulationReport

        states, report = [], SimulationReport()
        elapsed = self._elapsed_offset
        full_schedule_fp = (
            schedule_fingerprint(self._checkpoint_schedule)
            if self.checkpoint_every_report else None
        )
        iterator = self.nonlinear_solver.iter_schedule(
            self.model, self.initial_state, self.schedule,
        )
        cancelled = False
        for index, (state, step) in enumerate(iterator):
            states.append(state)
            report.steps.append(step)
            accepted = bool(step.converged)
            if self.result_store is not None:
                self.result_store.commit_state(
                    self.run_id, index, state, accepted=accepted,
                )
            if accepted:
                elapsed += float(step.dt)
            if accepted and self.checkpoint_every_report:
                completed = self._report_offset + index + 1
                checkpoint_id = CheckpointId(f"checkpoint_{completed:08d}")
                record = CheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    source_run_id=self.run_id,
                    completed_report_steps=completed,
                    accepted_elapsed_time=elapsed,
                    state=state,
                    schedule_fingerprint=full_schedule_fp,
                    model_fingerprint=default_model_fingerprint(self.model),
                    solver_fingerprint=default_solver_fingerprint(self.nonlinear_solver),
                    code_fingerprint=self.code_fingerprint,
                    control_index=(
                        int(self._checkpoint_schedule.step_control[completed-1])
                        if hasattr(self._checkpoint_schedule, "step_control") else None
                    ),
                )
                self.result_store.commit_checkpoint(
                    self.run_id, checkpoint_id, record.to_dict(),
                )
                evidence.checkpoints.append({
                    "checkpoint_id": str(checkpoint_id),
                    "completed_report_steps": completed,
                    "accepted_elapsed_time": elapsed,
                })
                stream.emit(
                    EventKind.CHECKPOINT_COMMITTED,
                    payload={"checkpoint_id": str(checkpoint_id)},
                )
            if accepted:
                previous = self.initial_state if index == 0 else states[index-1]
                self._record_step_evidence(
                    previous, state, step, evidence, stream,
                )
            if self.cancellation.requested:
                cancelled = True
                iterator.close()
                break
            if not accepted:
                break
        return states, report, cancelled

    def _record_step_evidence(self, previous, state, step, evidence, stream):
        if self.conservation_adapter is not None:
            boundary = (
                self.boundary_rate_provider(previous, state, float(step.dt))
                if self.boundary_rate_provider is not None else None
            )
            sources = (
                self.source_rate_provider(previous, state, float(step.dt))
                if self.source_rate_provider is not None else None
            )
            report = conservation_report(
                self.conservation_adapter, previous, state, float(step.dt),
                boundary_rates=boundary, source_rates=sources,
            )
            evidence.conservation.setdefault("steps", []).append(report.to_dict())
        if self.invariant_suite is not None:
            try:
                results = self.invariant_suite.evaluate(state, self.evidence_profile)
            except InvariantViolation as error:
                results = error.failures
                for result in results:
                    stream.emit(
                        EventKind.INVARIANT_VIOLATED,
                        payload=result.to_dict(),
                    )
                evidence.invariants.extend(result.to_dict() for result in results)
                raise
            evidence.invariants.extend(result.to_dict() for result in results)
            for result in results:
                if not result.passed:
                    stream.emit(
                        EventKind.INVARIANT_VIOLATED,
                        payload=result.to_dict(),
                    )

    def _live_observer(self, stream, evidence, previous_observer):
        kinds = {
            "report_step_started": EventKind.REPORT_STEP_STARTED,
            "ministep_started": EventKind.MINISTEP_STARTED,
            "nonlinear_iteration_completed": EventKind.NONLINEAR_ITERATION_COMPLETED,
            "linear_solve_completed": EventKind.LINEAR_SOLVE_COMPLETED,
            "linear_solve_failed": EventKind.LINEAR_SOLVE_COMPLETED,
            "timestep_cut": EventKind.TIMESTEP_CUT,
            "step_accepted": EventKind.STEP_ACCEPTED,
            "step_rejected": EventKind.STEP_REJECTED,
        }

        def observe(kind, payload):
            if previous_observer is not None:
                try:
                    previous_observer(kind, dict(payload))
                except Exception as error:
                    # Preserve isolation when composing observers. The native
                    # solver also isolates failures of this combined callback.
                    self.nonlinear_solver.observer_failures.append({
                        "event": kind, "type": type(error).__name__,
                        "message": str(error),
                    })
            safe = _finite_payload(payload)
            local_report = safe.get("report_step_index")
            report_index = (
                self._report_offset + int(local_report)
                if local_report is not None else None
            )
            attempt_index = safe.get("attempt_index")
            ministep_index = safe.get("ministep_index")
            report_id = (
                ReportStepId(f"{stream.run_id}:report:{report_index}")
                if report_index is not None else None
            )
            attempt_id = (
                AttemptId(f"{stream.run_id}:attempt:{report_index}:{int(attempt_index)}")
                if report_index is not None and attempt_index is not None else None
            )
            ministep_id = (
                MinistepId(f"{stream.run_id}:ministep:{report_index}:{int(ministep_index)}")
                if report_index is not None and ministep_index is not None else None
            )
            stream.emit(
                kinds[kind], report_step_id=report_id,
                ministep_id=ministep_id, attempt_id=attempt_id,
                payload={"live": True, "solver_event": kind, **safe},
            )
            if kind == "nonlinear_iteration_completed":
                evidence.convergence.setdefault("live_iterations", []).append(safe)
        return observe

    @staticmethod
    def _translate_report(
        report, evidence, stream: _EventSequence, *, include_solver_events=True,
        report_offset=0,
    ) -> None:
        for index, step in enumerate(getattr(report, "steps", ())):
            global_index = report_offset + index
            report_id = ReportStepId(f"{stream.run_id}:report:{global_index}")
            attempt_id = AttemptId(f"{stream.run_id}:attempt:{global_index}:0")
            if include_solver_events:
                stream.emit(
                    EventKind.REPORT_STEP_STARTED,
                    report_step_id=report_id,
                    attempt_id=attempt_id,
                    payload={"index": index, "dt": float(step.dt)},
                )
            if include_solver_events:
                for iteration, residuals in enumerate(getattr(step, "residuals", ())):
                    stream.emit(
                        EventKind.NONLINEAR_ITERATION_COMPLETED,
                        report_step_id=report_id,
                        attempt_id=attempt_id,
                        payload={"iteration": iteration, "residuals": residuals},
                    )
            if step.cuts and include_solver_events:
                stream.emit(
                    EventKind.TIMESTEP_CUT,
                    report_step_id=report_id,
                    attempt_id=attempt_id,
                    payload={"cuts": int(step.cuts)},
                )
            for mini_index, dt in enumerate(getattr(step, "ministeps", ()) if include_solver_events else ()):
                mini_id = MinistepId(f"{stream.run_id}:ministep:{index}:{mini_index}")
                stream.emit(
                    EventKind.MINISTEP_STARTED,
                    report_step_id=report_id,
                    ministep_id=mini_id,
                    attempt_id=attempt_id,
                    payload={"index": mini_index, "dt": float(dt), "retrospective": True},
                )
            if include_solver_events:
                kind = EventKind.STEP_ACCEPTED if step.converged else EventKind.STEP_REJECTED
                stream.emit(
                    kind,
                    report_step_id=report_id,
                    attempt_id=attempt_id,
                    payload={"iterations": int(step.iterations), "cuts": int(step.cuts)},
                )
            evidence.timestep_reports.append({
                "report_step_id": str(report_id),
                "dt": float(step.dt),
                "converged": bool(step.converged),
                "iterations": int(step.iterations),
                "cuts": int(step.cuts),
                "line_searches": int(getattr(step, "line_searches", 0)),
                "ministeps": [float(value) for value in getattr(step, "ministeps", ())],
                "residuals": _finite_payload(list(getattr(step, "residuals", ()))),
            })


def _finite_payload(value):
    """Make live diagnostics strict-JSON safe without hiding non-finites."""
    if isinstance(value, dict):
        return {str(key): _finite_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_payload(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


__all__ = ["SimulationResult", "SimulationSession", "SimulationStatus"]
