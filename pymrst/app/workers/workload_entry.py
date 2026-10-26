"""Private isolated workload-qualification process entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
from time import monotonic, process_time

import numpy as np


_cancelled = False
_progress_offset = 0.0
_progress_span = 1.0
_progress_label = ""
_telemetry_phase = "startup"
_telemetry_run = 0


def _cancel(*_args) -> None:
    global _cancelled
    _cancelled = True


def _emit(event: str, **payload) -> None:
    if event == "progress":
        inner = min(max(float(payload.get("progress", 0.0)) / 0.95, 0.0), 1.0)
        payload["progress"] = _progress_offset + _progress_span * inner
        if _progress_label:
            payload["message"] = f"{_progress_label}: {payload.get('message', 'running')}"
    print(json.dumps({"protocol_version": 1, "event": event, **payload},
                     sort_keys=True, separators=(",", ":")), flush=True)


def _raise_if_cancelled() -> None:
    if _cancelled:
        raise InterruptedError("workload qualification cancelled")


_PROFILES = {
    "smoke": {"cells": 5_000, "repeats": 3, "solve_cells": 2_000},
    "engineering": {"cells": 100_000, "repeats": 8, "solve_cells": 30_000},
    "field_scale": {"cells": 750_000, "repeats": 12, "solve_cells": 100_000},
    "endurance": {"cells": 250_000, "repeats": 100, "solve_cells": 50_000},
}

_SPE1_STEPS = {"smoke": 1, "engineering": 30}

_MANDEL_PROFILES = {
    "smoke": {"nx": 12, "nz": 4, "substeps": 4, "maximum_error": 0.06},
    "engineering": {"nx": 40, "nz": 4, "substeps": 20, "maximum_error": 0.01},
}


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class _ResourceTelemetry:
    """Bounded in-process CPU, memory and I/O time-series sampler."""

    def __init__(self, interval_seconds: float = 0.25, maximum_samples: int = 20_000):
        self.interval_seconds = float(interval_seconds)
        self.maximum_samples = int(maximum_samples)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._previous_wall: float | None = None
        self._previous_cpu: float | None = None
        self._process = None
        self.provider = "stdlib-resource"
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            self.provider = "psutil"
        except (ImportError, OSError):
            pass

    def _snapshot(self) -> dict:
        global _telemetry_phase, _telemetry_run
        now = monotonic()
        rss, cpu, read_bytes, write_bytes, threads = (
            _maximum_rss_bytes(), process_time(), None, None, None
        )
        if self._process is not None:
            try:
                memory = self._process.memory_info()
                times = self._process.cpu_times()
                rss = int(memory.rss)
                cpu = float(times.user + times.system)
                threads = int(self._process.num_threads())
            except (AttributeError, OSError, RuntimeError):
                self._process = None
                self.provider = "stdlib-resource"
            else:
                try:
                    io = self._process.io_counters()
                    read_bytes, write_bytes = int(io.read_bytes), int(io.write_bytes)
                except (AttributeError, OSError, RuntimeError):
                    # macOS exposes RSS/CPU/thread information through psutil,
                    # but not per-process I/O counters. Preserve every metric
                    # the platform does provide instead of discarding them all.
                    read_bytes, write_bytes = None, None
        delta_wall = None if self._previous_wall is None else now - self._previous_wall
        delta_cpu = None if self._previous_cpu is None else cpu - self._previous_cpu
        utilization = (
            None if delta_wall is None or delta_wall <= 0 or delta_cpu is None
            else max(0.0, 100.0 * delta_cpu / delta_wall)
        )
        self._previous_wall, self._previous_cpu = now, cpu
        return {
            "relative_seconds": max(0.0, now - self._started),
            "rss_bytes": rss, "cpu_seconds": cpu,
            "cpu_utilization_percent": utilization,
            "read_bytes": read_bytes, "write_bytes": write_bytes,
            "thread_count": threads, "phase": _telemetry_phase,
            "run": _telemetry_run,
        }

    def _sample(self) -> None:
        if len(self.samples) < self.maximum_samples:
            self.samples.append(self._snapshot())

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._started = monotonic()
        self._sample()
        self._thread = threading.Thread(
            target=self._loop, name="pymrst-workload-telemetry", daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))
        self._sample()
        rss_peak = max(self.samples, key=lambda item: item["rss_bytes"])
        cpu_samples = [item for item in self.samples
                       if item["cpu_utilization_percent"] is not None]
        cpu_peak = max(cpu_samples, key=lambda item: item["cpu_utilization_percent"],
                       default=None)
        first, last = self.samples[0], self.samples[-1]
        return {
            "provider": self.provider,
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples),
            "truncated": len(self.samples) >= self.maximum_samples,
            "samples": self.samples,
            "summary": {
                "sampled_peak_rss_bytes": int(rss_peak["rss_bytes"]),
                "sampled_peak_rss_at_seconds": float(rss_peak["relative_seconds"]),
                "sampled_peak_rss_phase": rss_peak["phase"],
                "sampled_peak_rss_run": int(rss_peak["run"]),
                "peak_cpu_utilization_percent": (
                    None if cpu_peak is None
                    else float(cpu_peak["cpu_utilization_percent"])
                ),
                "peak_cpu_at_seconds": (
                    None if cpu_peak is None else float(cpu_peak["relative_seconds"])
                ),
                "read_bytes_delta": (
                    None if first["read_bytes"] is None or last["read_bytes"] is None
                    else max(0, int(last["read_bytes"] - first["read_bytes"]))
                ),
                "write_bytes_delta": (
                    None if first["write_bytes"] is None or last["write_bytes"] is None
                    else max(0, int(last["write_bytes"] - first["write_bytes"]))
                ),
                "maximum_thread_count": max(
                    (item["thread_count"] for item in self.samples
                     if item["thread_count"] is not None), default=None,
                ),
            },
        }


def _numerical_core(tier: str, seed: int) -> dict:
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    profile = _PROFILES[tier]
    cells, repeats = profile["cells"], profile["repeats"]
    rng = np.random.default_rng(seed)
    pressure = 20e6 + rng.standard_normal(cells) * 2e5
    transmissibility = np.exp(rng.normal(-1.0, 0.35, cells - 1))
    accumulation = np.zeros(cells, dtype=np.float64)
    checksum = hashlib.sha256()
    for repeat in range(repeats):
        _raise_if_cancelled()
        flux = transmissibility * (pressure[:-1] - pressure[1:])
        accumulation.fill(0.0)
        np.add.at(accumulation, np.arange(cells - 1), -flux)
        np.add.at(accumulation, np.arange(1, cells), flux)
        pressure += accumulation * (1e-10 / max(float(np.abs(accumulation).max()), 1.0))
        checksum.update(np.asarray((pressure.mean(), flux.mean(), flux.std())).tobytes())
        _emit("progress", progress=0.75 * (repeat + 1) / repeats,
              message=f"Vectorized assembly {repeat + 1}/{repeats}")
    solve_cells = profile["solve_cells"]
    diagonal = np.full(solve_cells, 2.1)
    matrix = diags((-np.ones(solve_cells - 1), diagonal,
                    -np.ones(solve_cells - 1)), (-1, 0, 1), format="csc")
    rhs = np.sin(np.linspace(0, np.pi, solve_cells))
    solution = spsolve(matrix, rhs)
    residual = float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))
    checksum.update(solution[::max(1, solve_cells // 512)].tobytes())
    _emit("progress", progress=0.95, message="Sparse linear solve complete")
    return {
        "cells": cells, "connections": cells - 1, "assembly_repeats": repeats,
        "linear_system_size": solve_cells, "linear_residual_inf": residual,
        "operations": int(repeats * (cells - 1) + solve_cells),
        "result_checksum": checksum.hexdigest(),
    }


def _state_io(tier: str, seed: int, output: Path) -> dict:
    profile = _PROFILES[tier]
    cells, repeats = profile["cells"], max(1, profile["repeats"] // 2)
    rng = np.random.default_rng(seed)
    pressure = 20e6 + rng.standard_normal(cells) * 1e5
    saturation = rng.dirichlet((2.0, 3.0, 1.0), size=cells)
    payload = output / "state-throughput.npz"
    total_bytes, checksum = 0, hashlib.sha256()
    for repeat in range(repeats):
        _raise_if_cancelled()
        np.savez_compressed(payload, pressure=pressure, saturation=saturation)
        total_bytes += payload.stat().st_size
        with np.load(payload, allow_pickle=False) as state:
            checksum.update(state["pressure"][::max(1, cells // 512)].tobytes())
            if state["saturation"].shape != (cells, 3):
                raise RuntimeError("state round-trip shape mismatch")
        _emit("progress", progress=0.95 * (repeat + 1) / repeats,
              message=f"State round trip {repeat + 1}/{repeats}")
    payload.unlink(missing_ok=True)
    return {
        "cells": cells, "state_round_trips": repeats,
        "io_bytes": total_bytes, "operations": int(cells * repeats),
        "result_checksum": checksum.hexdigest(),
    }


def _spe1_blackoil(tier: str, _seed: int) -> dict:
    """Execute production SPE1 and return solver and engineering invariants."""
    from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
        _collect_history, setup_spe1_tutorial,
    )
    from pymrst.modules.ad_core import NonLinearSolver

    if tier not in _SPE1_STEPS:
        raise ValueError(f"SPE1 black-oil does not support tier {tier!r}")
    configured = os.environ.get("PYMRST_QUALIFICATION_SPE1_DECK", "").strip()
    if not configured:
        raise FileNotFoundError("the service did not resolve BENCH_SPE1.DATA")
    deck_path = Path(configured).resolve()
    if not deck_path.is_file():
        raise FileNotFoundError("the service-resolved BENCH_SPE1.DATA path is unavailable")
    steps = _SPE1_STEPS[tier]
    deck, model, state = setup_spe1_tutorial(deck_path)
    timesteps = np.asarray(deck["tsteps"], dtype=float)[:steps]
    if timesteps.size != steps:
        raise ValueError(f"SPE1 schedule contains only {timesteps.size} requested steps")
    solver = NonLinearSolver(
        max_iterations=30, max_timestep_cuts=6, use_line_search=True,
        verbose=False, progress="off",
    )
    states, reports, elapsed = [], [], 0.0
    for index, dt in enumerate(timesteps):
        _raise_if_cancelled()
        state, report = solver.solve_timestep(model, state, float(dt))
        if not report.converged:
            raise RuntimeError(f"SPE1 report step {index + 1} did not converge")
        states.append(state.copy()); reports.append(report); elapsed += float(dt)
        _emit("progress", progress=0.95 * (index + 1) / steps,
              message=f"SPE1 report step {index + 1}/{steps} accepted")
    history = _collect_history(states)
    pressure = np.asarray(history["pressure"], dtype=np.float64)
    sw = np.asarray(history["sW"], dtype=np.float64)
    sg = np.asarray(history["sG"], dtype=np.float64)
    so = 1.0 - sw - sg
    bhp = np.asarray(history["bhp"], dtype=np.float64)
    rates = np.concatenate(tuple(
        np.asarray(history[name], dtype=np.float64).reshape(steps, -1)
        for name in ("qWs", "qOs", "qGs")
    ), axis=1)
    final_residuals = [dict(report.residuals[-1]) if report.residuals else {}
                       for report in reports]
    residual_maxima = {
        name: max((float(values.get(name, 0.0)) for values in final_residuals), default=0.0)
        for name in ("CNV", "MB", "well", "control")
    }
    finite_state = all(np.all(np.isfinite(value)) for value in
                       (pressure, sw, so, sg, bhp, rates))
    phase_values = np.column_stack((sw.ravel(), so.ravel(), sg.ravel()))
    saturation_tolerance = 1e-8
    saturation_bounded = bool(
        phase_values.min() >= -saturation_tolerance
        and phase_values.max() <= 1.0 + saturation_tolerance
    )
    checksum = hashlib.sha256()
    for value in (pressure[-1], sw[-1], sg[-1], bhp[-1], rates[-1]):
        checksum.update(np.ascontiguousarray(value).tobytes())
    return {
        "case": "SPE1", "cells": int(model.G.cells.num),
        "wells": len(model.wells), "report_steps": steps,
        "simulated_time_seconds": elapsed,
        "newton_iterations": int(sum(report.iterations for report in reports)),
        "timestep_cuts": int(sum(report.cuts for report in reports)),
        "line_searches": int(sum(report.line_searches for report in reports)),
        "linear_iterations": int(sum(report.total_linear_iterations for report in reports)),
        "solver_wall_seconds": float(sum(report.wall_seconds for report in reports)),
        "assembly_seconds": float(sum(report.assembly_seconds for report in reports)),
        "final_pressure_min": float(pressure[-1].min()),
        "final_pressure_max": float(pressure[-1].max()),
        "final_pressure_mean": float(pressure[-1].mean()),
        "final_bhp": bhp[-1].tolist(),
        "maximum_final_residuals": residual_maxima,
        "minimum_saturation": float(phase_values.min()),
        "maximum_saturation": float(phase_values.max()),
        "maximum_saturation_sum_error": float(np.max(np.abs(sw + so + sg - 1.0))),
        "operations": int(sum(report.iterations for report in reports)),
        "deck_sha256": hashlib.sha256(deck_path.read_bytes()).hexdigest(),
        "result_checksum": checksum.hexdigest(),
        "correctness": {
            "all_report_steps_completed": len(reports) == steps,
            "all_report_steps_converged": all(report.converged for report in reports),
            "finite_reservoir_and_well_state": bool(finite_state),
            "positive_pressure": bool(np.all(pressure > 0.0)),
            "saturations_bounded": saturation_bounded,
            "saturations_sum_to_one": bool(np.max(np.abs(sw + so + sg - 1.0)) <= 1e-12),
            "final_cnv_within_tolerance": residual_maxima["CNV"] <= float(model.tol_cnv),
            "final_mb_within_tolerance": residual_maxima["MB"] <= float(model.tol_mb),
            "final_well_within_tolerance": residual_maxima["well"] <= float(model.tol_well),
            "final_control_within_tolerance": residual_maxima["control"] <= float(model.tol_well),
        },
    }


def _mandel_poromechanics(tier: str, _seed: int) -> dict:
    """Exercise the coupled flow/VEM kernel against Mandel's exact solution."""
    from pymrst.modules.ad_mechanics.mandel import MandelCase, mandel_pressure

    if tier not in _MANDEL_PROFILES:
        raise ValueError(f"Mandel poromechanics does not support tier {tier!r}")
    profile = _MANDEL_PROFILES[tier]
    times = np.asarray((0.01, 0.05, 0.1, 0.2, 0.5), dtype=float)
    case = MandelCase(nx=profile["nx"], nz=profile["nz"])
    state, plate = case.initial_state()
    x = np.asarray(case.G.cells.centroids[:, 0], dtype=float) / case.a
    centre = int(np.argmin(x))
    p0 = float(state["pressure"][centre])
    if not np.isfinite(p0) or p0 <= 0.0:
        raise RuntimeError("Mandel undrained pressure normalization is invalid")
    profiles, states, plate_history = [], [], []
    previous, completed = 0.0, 0
    total_steps = int(times.size * profile["substeps"])
    for target in times:
        dt = (float(target) - previous) * case.a**2 / case.cv / profile["substeps"]
        for _ in range(profile["substeps"]):
            _raise_if_cancelled()
            state, plate = case.solve_step(state, dt, plate)
            completed += 1
        states.append(state.copy()); plate_history.append(float(plate))
        profiles.append(np.asarray(state["pressure"], dtype=float) / p0)
        _emit("progress", progress=0.95 * completed / total_steps,
              message=f"Mandel time {float(target):g} accepted ({completed}/{total_steps} coupled steps)")
        previous = float(target)
    simulated = np.asarray(profiles, dtype=float)
    exact = mandel_pressure(x, times, case.nu, case.params["nu_u"])
    error = np.abs(simulated - exact)
    centre_history = simulated[:, centre]
    force_error = abs(case.plate_force(states[-1]) - case.load) / case.load
    z_spread = 0.0
    x_coordinates = np.asarray(case.G.cells.centroids[:, 0], dtype=float)
    for coordinate in np.unique(np.round(x_coordinates, 10)):
        values = simulated[:, np.isclose(x_coordinates, coordinate)]
        z_spread = max(z_spread, float(np.max(np.ptp(values, axis=1))))
    finite = all(np.all(np.isfinite(value)) for value in (
        simulated, exact, states[-1]["pressure"], states[-1]["xd"], states[-1]["u"],
        np.asarray(plate_history),
    ))
    maximum_error = float(error.max())
    peak = float(centre_history.max())
    checksum = hashlib.sha256()
    for value in (simulated, states[-1]["pressure"], states[-1]["xd"],
                  states[-1]["u"], np.asarray(plate_history)):
        checksum.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return {
        "case": "Mandel", "cells": int(case.G.cells.num),
        "nodes": int(case.G.nodes.num), "faces": int(case.G.faces.num),
        "mechanical_free_dofs": int(case.mech.n_free),
        "dimensionless_report_times": times.tolist(),
        "accepted_coupled_steps": total_steps,
        "maximum_pressure_relative_error": maximum_error,
        "pressure_l2_error": float(np.sqrt(np.mean((simulated - exact) ** 2))),
        "centre_peak_pressure_ratio": peak,
        "final_centre_pressure_ratio": float(centre_history[-1]),
        "maximum_column_pressure_spread": z_spread,
        "final_plate_force_relative_error": float(force_error),
        "initial_pressure_normalization_pa": p0,
        "final_plate_displacement_m": float(plate_history[-1]),
        "operations": total_steps, "result_checksum": checksum.hexdigest(),
        "correctness": {
            "finite_coupled_state": bool(finite),
            "positive_undrained_pressure": bool(p0 > 0.0),
            "mandel_cryer_overshoot_reproduced": bool(peak >= 1.05),
            "pressure_profile_matches_analytic": bool(
                maximum_error <= profile["maximum_error"]
            ),
            "pressure_uniform_through_thickness": bool(z_spread <= 1e-6),
            "constant_plate_load_balanced": bool(force_error <= 1e-6),
            # Reservoir coordinates use positive depth, so downward plate
            # motion under compression is positive in this model.
            "compression_displacement_is_depth_positive": bool(plate_history[-1] > 0.0),
            "pressure_is_draining_after_peak": bool(centre_history[-1] < peak),
            "normalized_pressure_nonnegative": bool(simulated.min() >= -1e-8),
        },
    }


def _dispatch(workload_id: str, tier: str, seed: int, output: Path) -> dict:
    if workload_id == "numerical-core":
        return _numerical_core(tier, seed)
    if workload_id == "state-io":
        return _state_io(tier, seed, output)
    if workload_id == "spe1-blackoil":
        return _spe1_blackoil(tier, seed)
    if workload_id == "mandel-poromechanics":
        return _mandel_poromechanics(tier, seed)
    raise ValueError(f"unknown workload: {workload_id}")


def _statistics(values) -> dict[str, float | int | list[float]]:
    samples = np.asarray(values, dtype=np.float64)
    median = float(np.median(samples))
    return {
        "count": int(samples.size), "median": median,
        "mad": float(np.median(np.abs(samples - median))),
        "minimum": float(samples.min()), "maximum": float(samples.max()),
        "p10": float(np.percentile(samples, 10)),
        "p90": float(np.percentile(samples, 90)),
        "samples": samples.tolist(),
    }


def _run_repeated(request: dict, output: Path) -> dict:
    global _progress_label, _progress_offset, _progress_span
    global _telemetry_phase, _telemetry_run
    workload_id, tier, seed = request["workload_id"], request["tier"], int(request["seed"])
    warmups = int(request.get("warmup_runs", 1))
    measured = int(request.get("measured_runs", 3))
    if not 0 <= warmups <= 5 or not 1 <= measured <= 9 or measured % 2 == 0:
        raise ValueError("invalid warmup/measured repetition policy")
    total = warmups + measured
    measured_payloads, warmup_times = [], []
    for index in range(total):
        _raise_if_cancelled()
        is_warmup = index < warmups
        ordinal = index + 1 if is_warmup else index - warmups + 1
        _progress_offset, _progress_span = index / total, 1.0 / total
        _progress_label = (
            f"Warmup {ordinal}/{warmups}" if is_warmup
            else f"Measured run {ordinal}/{measured}"
        )
        _telemetry_phase = "warmup" if is_warmup else "measured"
        _telemetry_run = ordinal
        wall_started, cpu_started = monotonic(), process_time()
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        payload = _dispatch(workload_id, tier, seed, output)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        timing = {
            "wall_seconds": monotonic() - wall_started,
            "cpu_seconds": process_time() - cpu_started,
            "user_cpu_seconds": usage.ru_utime - usage_before.ru_utime,
            "system_cpu_seconds": usage.ru_stime - usage_before.ru_stime,
        }
        _emit("progress", progress=0.95,
              message="completed; excluded from statistics" if is_warmup else "completed")
        if is_warmup:
            warmup_times.append(timing["wall_seconds"])
        else:
            payload.update(timing); measured_payloads.append(payload)
    _progress_label, _progress_offset, _progress_span = "", 0.0, 1.0
    _telemetry_phase, _telemetry_run = "aggregation", 0
    representative = dict(measured_payloads[-1])
    statistic_names = (
        "wall_seconds", "cpu_seconds", "user_cpu_seconds", "system_cpu_seconds",
        "solver_wall_seconds", "assembly_seconds",
    )
    statistics = {
        name: _statistics([payload[name] for payload in measured_payloads])
        for name in statistic_names if all(name in payload for payload in measured_payloads)
    }
    for name in ("wall_seconds", "cpu_seconds", "user_cpu_seconds", "system_cpu_seconds"):
        representative[name] = statistics[name]["median"]
    checksums = [str(payload.get("result_checksum", "")) for payload in measured_payloads]
    deterministic = bool(checksums[0] and len(set(checksums)) == 1)
    correctness = dict(representative.get("correctness") or {})
    correctness["deterministic_across_measured_runs"] = deterministic
    representative.update({
        "warmup_runs": warmups, "measured_runs": measured,
        "warmup_wall_seconds": warmup_times,
        "statistics": statistics, "measured_result_checksums": checksums,
        "correctness": correctness, "peak_rss_bytes": _maximum_rss_bytes(),
    })
    return representative


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    request_path, output = Path(arguments.request).resolve(), Path(arguments.output).resolve()
    if not request_path.is_file() or request_path.parent != output.parent:
        raise ValueError("workload request and output must share a qualification directory")
    output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _cancel)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _cancel)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    workload_id, tier, seed = request["workload_id"], request["tier"], int(request["seed"])
    if tier not in _PROFILES:
        raise ValueError(f"unknown workload tier: {tier}")
    qualification_started = monotonic()
    _emit("started", process_id=os.getpid())
    try:
        telemetry_sampler = _ResourceTelemetry()
        telemetry_sampler.start()
        try:
            metrics = _run_repeated(request, output)
        finally:
            telemetry = telemetry_sampler.stop()
        _raise_if_cancelled()
        metrics["qualification_wall_seconds"] = monotonic() - qualification_started
        telemetry_samples = telemetry.pop("samples")
        telemetry_document = {
            "schema_version": 1, "provider": telemetry["provider"],
            "sampling_interval_seconds": telemetry["sampling_interval_seconds"],
            "samples": telemetry_samples,
        }
        telemetry_encoded = (json.dumps(
            telemetry_document, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ) + "\n").encode()
        telemetry_path = output / "resource-telemetry.json"
        telemetry_temporary = output / "resource-telemetry.tmp"
        telemetry_temporary.write_bytes(telemetry_encoded)
        os.replace(telemetry_temporary, telemetry_path)
        telemetry.update({
            "artifact_relative_path": telemetry_path.relative_to(output.parent).as_posix(),
            "artifact_sha256": hashlib.sha256(telemetry_encoded).hexdigest(),
            "artifact_bytes": len(telemetry_encoded),
        })
        metrics["telemetry"] = telemetry
        metrics["peak_rss_bytes"] = max(
            int(metrics["peak_rss_bytes"]),
            int(telemetry["summary"]["sampled_peak_rss_bytes"]),
        )
        sample_times = [item["relative_seconds"] for item in telemetry_samples]
        correctness = dict(metrics.get("correctness") or {})
        correctness.update({
            "resource_telemetry_collected": telemetry["sample_count"] >= 2,
            "resource_telemetry_time_monotonic": all(
                right >= left for left, right in zip(sample_times, sample_times[1:])
            ),
            "sampled_rss_not_above_process_peak": (
                telemetry["summary"]["sampled_peak_rss_bytes"]
                <= metrics["peak_rss_bytes"]
            ),
        })
        metrics["correctness"] = correctness
        _emit("result", metrics=metrics)
        return 0
    except InterruptedError as error:
        _emit("cancelled", message=str(error))
        return 130
    except BaseException as error:
        _emit("failure", error_type=type(error).__name__, message=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
