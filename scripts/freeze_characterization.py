"""Freeze and compare Foundation 0.3 characterization baselines.

    python3 scripts/freeze_characterization.py --freeze          # record all
    python3 scripts/freeze_characterization.py --check           # compare all
    python3 scripts/freeze_characterization.py --check p64_water # one case

A baseline answers one question: *did this change alter behaviour?* It is not a
correctness check. Correctness is what `examples/validation/validate_phase*.py`
already establishes against MATLAB; this records what PyMRST does **now**, so a
later change can be shown to preserve it.

**States are not enough.** Two runs can agree on every final value and disagree
on iteration counts, timestep cuts and achieved residuals. An integration that
changes solver decisions has changed behaviour even when the answer matches,
and that is precisely what Milestone 1.3's gate asks about. So the baseline
records `attempts`, `iterations`, `cuts` and `converged` alongside the states,
and `--check` reports a decision difference as loudly as a numerical one.

**Durable data is JSON, never a pickle.** ADR 0002 forbids arbitrary Python
objects in persistent evidence, and a pickle would also silently rot as classes
move. Everything written here is JSON-compatible, so a baseline stays readable
when the code that produced it no longer imports.

Tolerances are **relative with `atol=0`**. Transmissibilities and several state
fields are around 1e-14 in SI, where `numpy`'s default `atol=1e-8` calls any two
values equal; a baseline using it would pass whatever the code did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "docs" / "research_studio" / "characterization"
RTOL = 1e-10


# --------------------------------------------------------------------------
# case registry

def _case_p64_water():
    """Single-phase water, fixed-stress splitting. MRST-validated as phase 64."""
    from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
    return run_2d_case(fluid_model="water", method="fixed_stress",
                       nx=20, ny=4, nsteps=3)


def _case_p65_oilwater():
    """Immiscible oil-water, fixed-stress splitting. Phase 65."""
    from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
    return run_2d_case(fluid_model="oil_water", method="fixed_stress",
                       nx=20, ny=4, nsteps=3)


def _case_norne_mini():
    """Norne field geomechanics, mini subset (605 cells), fully coupled.

    The second case Milestone 1.9's gate names, and the reason it was still
    open. Nothing here needed adding to the runner: it already records
    `model.acceptance_reports`, one dict per step carrying `Converged`,
    `Iterations` and - in the fully coupled branch - the per-iteration
    `Residuals` the classifier reads. They hang off the model rather than
    appearing in the returned `(g, model, states)` tuple, so the harness could
    not see them; wrapping the tuple is this function's whole job.

    Kept to the mini subset deliberately. The full 7392-cell model is the same
    physics and takes long enough to make the whole freeze set unpleasant to
    re-run, and a characterization case earns its place by being run often.
    """
    from pymrst.modules.ad_mechanics.examples.run_norne_example import (
        run_norne_example,
    )
    g, model, states = run_norne_example(
        "mini Norne", method="fully coupled", fluid_model="oil water",
        n_steps=3, check=False,
    )
    # Normalized to the lowercase shape `_reports_of` produces for solver
    # reports. The runner writes MRST's capitalized keys, and
    # `_residual_histories` walks for `residuals`; left as `Residuals` the
    # history is present, recorded, and silently never classified - which is
    # indistinguishable from having no history at all.
    reports = [
        {"step": i,
         "converged": bool(entry.get("Converged", True)),
         "iterations": int(entry.get("Iterations", 0)),
         "cuts": int(entry.get("Cuts", 0)),
         "residuals": [dict(r) for r in entry.get("Residuals", ())]}
        for i, entry in enumerate(model.acceptance_reports)
    ]
    return SimpleNamespace(
        grid=g, model=model, states=list(states),
        dts=np.asarray(model.acceptance_timesteps, dtype=float),
        reports=reports,
    )


def _case_p64_coupled():
    """The same water case, fully coupled rather than split.

    Included because the pair is what makes a formulation comparison possible
    later (Milestone 2.3); on its own each is just a run.
    """
    from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case
    return run_2d_case(fluid_model="water", method="fully_coupled",
                       nx=20, ny=4, nsteps=3)


def _case_spe1_compositional():
    """SPE1 in compositional (natural) variables. MRST-validated as phase 114."""
    from pymrst.modules.compositional.examples.run_compositional_spe1 import (
        run_compositional_spe1,
    )
    return run_compositional_spe1(nsteps=2)


def _case_blackoil_one_phase():
    """Single-phase black-oil tutorial, both the open and closed well settings.

    Named honestly: this is the black-oil *model family*, not SPE1. A black-oil
    SPE1 baseline still belongs in this registry; the deck path used by the
    validators is the obvious source and is not wired up here yet.
    """
    from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_one_phase import (
        run_blackoil_tutorial_one_phase,
    )
    return run_blackoil_tutorial_one_phase()


def _case_adjoint_uplift():
    """Adjoint gradient of a surface-uplift objective, with its FD cross-check.

    The plan classifies the three `simulate_forward` variants as intentionally
    specialized: they must not be forced through a forward-only session kernel.
    Freezing one of them is what makes that boundary checkable rather than
    merely asserted.
    """
    from pymrst.modules.ad_mechanics.examples.run_adjoint_example import (
        run_adjoint_example,
    )
    return run_adjoint_example()


def _case_packed_compositional():
    """A packed problem: `simulate_packed_problem` with on-disk output handlers.

    `runtime_entrypoints.csv` classifies this as a specialized session adapter
    - it adds persistence and restart around a problem-owned simulator setup,
    so it is the one entry-point class a forward-only session kernel cannot
    simply absorb. Freezing it is what makes that boundary checkable.

    It runs in a temporary directory: the packed runner writes output handlers
    to the working directory, and a baseline that leaves artifacts in the repo
    would make its own second run differ from its first.
    """
    import os
    import tempfile
    from pymrst.modules.compositional.examples.compositional_validation_simple import (
        run_compositional_validation_packed,
    )

    previous = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="pymrst_characterization_")
    try:
        os.chdir(scratch)
        return run_compositional_validation_packed(nstep=2)
    finally:
        os.chdir(previous)


CASES = {
    "p64_water": (_case_p64_water, "examples/validation/ref_p64_run2d_water_fixedstress.mat"),
    "p65_oilwater": (_case_p65_oilwater, "examples/validation/ref_p65_run2d_oilwater_fixedstress.mat"),
    "p64_coupled": (_case_p64_coupled, ""),
    "spe1_compositional": (_case_spe1_compositional,
                           "examples/validation/ref_p114_spe1_native_compositional.mat"),
    "blackoil_one_phase": (_case_blackoil_one_phase, ""),
    "adjoint_uplift": (_case_adjoint_uplift, ""),
    "packed_compositional": (_case_packed_compositional, ""),
    "norne_mini": (_case_norne_mini, ""),
}


# --------------------------------------------------------------------------
# recording

def _state_fields(state):
    """Every numeric field of a state, as sorted plain lists.

    `State` is dict-like: its data lives in keys, not attributes. A first
    version read `vars(state)` and recorded **nothing**, so the baseline
    compared an empty dict to an empty dict and reported "ok" whatever the
    solver produced. That is the vacuous-baseline failure this package exists
    to avoid, so both mappings and attributes are read here, and `record`
    refuses to write a case with no state fields at all.
    """
    out = {}
    items = {}
    if hasattr(state, "keys"):
        items.update({k: state[k] for k in state.keys()})
    items.update({k: v for k, v in vars(state).items()} if hasattr(state, "__dict__") else {})

    for name, value in items.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[str(name)] = np.asarray(value, dtype=float).ravel().tolist()
        elif isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[str(name)] = float(value)
    return dict(sorted(out.items()))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    # Reports arrive as dataclasses and small objects as well as dicts. Their
    # fields are the solver decisions, so they are unpacked rather than
    # stringified: `str(report)` would record an address and compare equal to
    # nothing on the next run.
    if hasattr(value, "__dict__") and vars(value):
        return {str(k): _jsonable(v) for k, v in sorted(vars(value).items())}
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    return str(value)


def _states_of(result):
    """States from a result, whatever shape it comes in.

    The runners do not share a return type: some expose `.states`, the
    black-oil tutorial returns a dict of per-setting results, and the
    compositional runner names them `compositional_states`.
    """
    for attr in ("states", "compositional_states"):
        value = getattr(result, attr, None)
        if value is not None and len(value):
            return list(value)
    if isinstance(result, dict):
        # A packed run returns a plain dict whose "states" is a list of
        # per-problem state lists; the tutorial returns a dict of result
        # objects. Both flatten to a list of states.
        nested = result.get("states") if "states" in result else None
        if nested:
            merged = []
            for entry in nested:
                merged.extend(list(entry) if isinstance(entry, (list, tuple)) else [entry])
            if merged:
                return merged
        merged = []
        for key in sorted(result):
            merged.extend(getattr(result[key], "states", []) or [])
        if merged:
            return merged
    raise RuntimeError(f"no states found on {type(result).__name__}")


def _model_of(result):
    """The model an adapter should be built against.

    Same problem as `_states_of`: the poromechanics runners expose `.model`,
    while the compositional one keeps both the black-oil model it was converted
    from and the converted `compositional_model`. The converted one is the one
    whose components the adapter conserves.
    """
    for attr in ("compositional_model", "model"):
        model = getattr(result, attr, None)
        if model is not None:
            return model
    raise RuntimeError(f"no model found on {type(result).__name__}")


def _schedule_of(result):
    """Step lengths between the states `_states_of` returns.

    `.dts` is already a list of step lengths. `.time` is cumulative, so it has
    to be differenced - and the first state is the one at `time[0]`, not an
    initial condition at zero, so differencing yields one interval fewer than
    there are states, which is exactly the number of intervals between them.
    """
    dts = getattr(result, "dts", None)
    if dts is not None:
        return np.asarray(dts, dtype=float)
    time = getattr(result, "time", None)
    if time is not None:
        return np.diff(np.asarray(time, dtype=float).reshape(-1))
    raise RuntimeError(f"no schedule found on {type(result).__name__}")


def _reports_of(result):
    # A solver report first: it carries the per-iteration residual history,
    # where `iterations` is only a count. `run_compositional_spe1` now exposes
    # one, which is what lets --classify reach phase 114 - a case Milestone
    # 1.9's gate names explicitly.
    report = getattr(result, "report", None)
    steps = getattr(report, "steps", None)
    if steps:
        return [
            {"step": i, "converged": bool(getattr(st, "converged", True)),
             "iterations": int(getattr(st, "iterations", 0)),
             "residuals": [dict(r) for r in getattr(st, "residuals", [])]}
            for i, st in enumerate(steps)
        ]
    for attr in ("reports", "substeps", "iterations"):
        value = getattr(result, attr, None)
        # `len()` rather than truthiness: several of these are numpy arrays,
        # where `if value:` raises on anything longer than one element.
        if value is not None and len(value):
            return list(value)
    if isinstance(result, dict):
        nested = result.get("reports") if "reports" in result else None
        if nested:
            merged = []
            for entry in nested:
                merged.extend(list(entry) if isinstance(entry, (list, tuple)) else [entry])
            if merged:
                return merged
        merged = []
        for key in sorted(result):
            report = getattr(result[key], "report", None)
            if report:
                merged.append({"setting": key, "report": report})
        if merged:
            return merged
    return []


def _jacobian_signature(result):
    """Shape, nonzero count and a hash of the Jacobian's *structure*.

    The F0.3 addendum's highest-value item. Every existing check compares
    values: states, decisions, residuals. A sparse-assembly change that alters
    the sparsity *pattern* rather than the numbers - a different insertion
    order, a dropped explicit zero, a block assembled in another orientation -
    passes all of them and changes the matrix the solver factors.

    Hashed over `indices` and `indptr` only, never `data`, so this is
    orthogonal to the numerical comparison rather than a duplicate of it.
    Indices are sorted first, because two CSR matrices with identical
    structure can hold their column indices in different orders within a row
    and are the same matrix.

    Records *why* it is unavailable rather than a bare None, because the
    reasons differ and matter: a fixed-stress model raises NotImplementedError
    because it solves flow and mechanics in separate systems and has no single
    Jacobian to sign, while a dict-valued case has no model at all. A silent
    None makes those look identical, and makes a genuine regression - a model
    that used to assemble and now does not - look like a case that never did.
    """
    try:
        from pymrst.utils.adi import combine_equations
        model = _model_of(result)
        states = _states_of(result)
        schedule = _schedule_of(result)
        equations = model.get_equations(states[0], states[min(1, len(states)-1)],
                                        float(np.asarray(schedule).ravel()[0]))
        jacobian = combine_equations(*equations).jac[0].tocsr()
        jacobian.sort_indices()
        digest = hashlib.sha256()
        digest.update(np.asarray(jacobian.indices, dtype=np.int64).tobytes())
        digest.update(np.asarray(jacobian.indptr, dtype=np.int64).tobytes())
        return {
            "rows": int(jacobian.shape[0]),
            "columns": int(jacobian.shape[1]),
            "nonzeros": int(jacobian.nnz),
            "structure_sha256_16": digest.hexdigest()[:16],
        }
    except Exception as error:
        return {"unavailable": f"{type(error).__name__}: {str(error)[:80]}"
                               or type(error).__name__}


def _cells_of(result):
    candidates = [result, getattr(result, "model", None),
                  getattr(result, "compositional_model", None),
                  getattr(result, "blackoil_model", None)]
    if isinstance(result, dict):
        candidates += [result[k] for k in sorted(result)]
        candidates += [getattr(result[k], "model", None) for k in sorted(result)]
    for owner in candidates:
        grid = getattr(owner, "G", None)
        if grid is not None and hasattr(grid, "cells"):
            return int(grid.cells.num)
    return 0


def _metrics_of(result):
    """Named scalar outputs worth pinning, beyond states and solver decisions.

    The adjoint case records no per-step reports, so without this its
    `decisions` list is empty and that half of the comparison comes to
    `[] == []`. Its gradient and the finite-difference agreement are the
    quantities that actually characterize it.
    """
    out = {}
    for name in ("gradient_relative_error", "objective", "node", "direction"):
        value = getattr(result, name, None)
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[name] = float(value)
    for name in ("adjoint", "finite_difference_gradient", "controls", "dts"):
        value = getattr(result, name, None)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[name] = np.asarray(value, dtype=float).ravel().tolist()
    return dict(sorted(out.items()))


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:                                    # pragma: no cover
        return ""


def record(name: str) -> dict:
    """Run a case and return its baseline record."""
    builder, reference = CASES[name]
    started = time.perf_counter()
    result = builder()
    return _record_from(name, result, reference, time.perf_counter() - started)


def _record_from(name, result, reference, runtime) -> dict:
    states = _states_of(result)
    # First accepted step and the last one. The midpoint matters because a
    # divergence that only shows at the end is hard to locate in time.
    selected = {"first": 0, "middle": len(states) // 2, "final": len(states) - 1}

    recorded_states = {label: _state_fields(states[i]) for label, i in selected.items()}
    empty = [label for label, fields in recorded_states.items() if not fields]
    if empty:
        raise RuntimeError(
            f"{name}: recorded no state fields for {', '.join(empty)}. A baseline "
            f"that stores nothing compares nothing and reports success whatever "
            f"the code does; refusing to write it.")

    decisions = [_jsonable(r) for r in _reports_of(result)]
    metrics = _metrics_of(result)
    if not decisions and not metrics:
        raise RuntimeError(
            f"{name}: no solver decisions and no metrics recorded. States alone "
            f"cannot show that an integration changed how the solver behaved, "
            f"which is what this baseline is for; refusing to write it.")

    return {
        "case": name,
        "command": f"python3 scripts/freeze_characterization.py --check {name}",
        "provenance": {
            "commit": _commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matlab_reference": reference,
            "matlab_reference_sha256_16": _file_hash(ROOT / reference) if reference else "",
        },
        "shape": {
            "cells": _cells_of(result),
            "states": len(states),
            "reports": len(_reports_of(result)),
            "jacobian": _jacobian_signature(result),
        },
        # Solver decisions. Recorded separately from the states because a
        # change here is a behaviour change even when the states match.
        "decisions": decisions,
        "states": recorded_states,
        "metrics": metrics,
        "timing": {"seconds": round(runtime, 3)},
    }


# --------------------------------------------------------------------------
# observation: Milestone 1.3's gate

class _Recorder:
    """Counts events without touching anything the solver reads."""

    def __init__(self):
        self.events = []

    def __call__(self, kind, payload):
        self.events.append(kind)


def _attach_observer(builder, recorder):
    """Run `builder` with `recorder` attached to every instrumented owner.

    Two owners now carry the hook, not one: `NonLinearSolver` and the
    fixed-stress splitting loop, which keeps its own Newton iteration and so
    was invisible to the canonical instrumentation. Both are patched, because a
    case may use either.
    """
    import sys as _sys
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_mechanics.models.splitting import fixed_stress_models

    original_init = NonLinearSolver.__init__
    original_fixed = fixed_stress_models.simulate_fixed_stress_schedule

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("observer", recorder)
        original_init(self, *args, **kwargs)

    def patched_fixed(*args, **kwargs):
        kwargs.setdefault("observer", recorder)
        return original_fixed(*args, **kwargs)

    NonLinearSolver.__init__ = patched_init
    fixed_stress_models.simulate_fixed_stress_schedule = patched_fixed
    # `run_2d_case` holds its own module-scope reference to the splitting loop.
    # Reached through `sys.modules`, not `import ... as`: the package re-exports
    # the *function* `run_2d_case`, so the attribute lookup binds a function and
    # every patch lands on it silently. Fourth appearance of that trap here,
    # after phase_stability_test, convert_rs_rv_to_k_value and
    # upscaling/coarsegrid.
    run_2d = _sys.modules.get("pymrst.modules.ad_mechanics.examples.run_2d_case")
    original_alias = getattr(run_2d, "simulate_fixed_stress_schedule", None)
    if original_alias is not None:
        run_2d.simulate_fixed_stress_schedule = patched_fixed

    # The canonical `simulate_packed_problem` is deliberately NOT patched here.
    # `packed_compositional` does not use it: the compositional example imports
    # both its packer and its runner from `examples/_shared`, which has its own
    # in-memory `PackedSimulationProblem` - a different type with a different
    # field set, documented as a small analogue of MRST's disk-backed job.
    # F0.1 classifies that runner as a deprecated duplicate.
    #
    # An earlier attempt patched ad_core's runner into the example's namespace,
    # which handed `_shared`'s problem type to a runner that isinstance-checks
    # ad_core's, missing the scalar branch. That was the patch's fault, not a
    # defect in either module.
    try:
        return builder()
    finally:
        NonLinearSolver.__init__ = original_init
        fixed_stress_models.simulate_fixed_stress_schedule = original_fixed
        if original_alias is not None:
            run_2d.simulate_fixed_stress_schedule = original_alias


def _observed(builder):
    """Run `builder` with a recorder attached to every `NonLinearSolver`.

    Attached by wrapping `__init__`, because the hook is per-instance and a
    case constructs its own solvers. Returns `(result, recorder)`.
    """
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    recorder = _Recorder()
    return _attach_observer(builder, recorder), recorder


def check_observed(names) -> int:
    """Does attaching an observer change any recorded behaviour?

    This is what Milestone 1.3 asks. The comparison is against the baseline
    frozen *without* an observer, so any difference in states, solver decisions
    or metrics means observation is not observational.

    The run must also prove the observer fired. Without that check a failure to
    attach it would make every case pass, reporting a property that was never
    tested - the same vacuity that made the first version of this harness
    record no state fields at all.
    """
    bad = 0
    uninstrumented = []
    for name in names:
        path = baseline_path(name)
        if not path.exists():
            print(f"SKIP {name}: no baseline (run --freeze)")
            continue

        builder, reference = CASES[name]
        result, recorder = _observed(builder)
        if not recorder.events:
            # Not a harness fault. `NonLinearSolver` is the *canonical*
            # numerical owner, and some paths deliberately are not it: the
            # fixed-stress splitting runs its own Newton loop and constructs no
            # `NonLinearSolver` at all, and neither does the packed runner. So
            # M1.3's gate simply does not reach these cases yet, and saying so
            # is the point - reporting "ok" here would claim evidence that was
            # never gathered.
            uninstrumented.append(name)
            print(f"n/a  {name}: constructs no instrumented owner, so the hook "
                  f"cannot be evidenced on this path")
            continue

        current = _record_from(name, result, reference, 0.0)
        failures = compare(current, json.loads(path.read_text()))
        if failures:
            bad = 1
            print(f"FAIL {name}: observation changed behaviour "
                  f"({len(recorder.events)} events)")
            for f in failures[:8]:
                print(f"       {f}")
        else:
            kinds = len(set(recorder.events))
            print(f"ok   {name}: unchanged under {len(recorder.events)} events "
                  f"of {kinds} kinds")

    if uninstrumented:
        print(f"\n{len(uninstrumented)} of {len(names)} cases run no instrumented "
              f"solver: {', '.join(uninstrumented)}.")
        print("Two owners carry the hook: NonLinearSolver and the fixed-stress "
              "splitting loop. packed_compositional runs neither - it uses "
              "compositional/examples/_shared.simulate_packed_problem, which "
              "F0.1 classifies as a deprecated duplicate of the canonical "
              "runner. The canonical one is instrumented; no characterization "
              "case exercises it yet.")
    return bad


# --------------------------------------------------------------------------
# invariants: Milestone 1.8's gate

def check_invariants(names) -> int:
    """Do the invariants hold on real states, and do the profiles differ?

    Milestone 1.8 asks two things that are easy to conflate. The suite must
    hold on real data, and it must *distinguish floating-point noise from
    physical impossibility* - a saturation of -1e-15 is arithmetic, a
    saturation of -0.3 is a broken model, and an invariant that treats them
    alike is either useless or unusable.

    The profiles are checked on the same states rather than asserted from the
    source: `observe` reports and continues, `strict` raises. If both behaved
    identically the distinction would be decorative, so the difference itself
    is the assertion.
    """
    from pymrst.research_studio.evidence.invariants import (
        InvariantProfile, InvariantSuite, InvariantViolation, bounded_field,
        finite_field,
    )

    bad = 0
    for name in names:
        builder, _ = CASES[name]
        states = _states_of(builder())
        final = states[-1]
        fields = _state_fields(final)

        providers = [finite_field(f) for f in sorted(fields)]
        saturation = [f for f in fields if f in ("sW", "sG", "sO", "s")]
        providers += [bounded_field(f, 0.0, 1.0) for f in saturation]

        suite = InvariantSuite(providers)
        results = suite.evaluate(final, InvariantProfile.OBSERVE)
        failed = [r for r in results if not r.passed]
        if failed:
            bad = 1
            print(f"FAIL {name}: {len(failed)} invariant(s) violated on real data")
            for r in failed[:4]:
                print(f"       {r.message}")
            continue
        print(f"ok   {name}: {len(results)} invariants hold"
              + (f", {len(saturation)} saturation bound(s)" if saturation else ""))

    # Noise must not be treated as impossibility, and impossibility must not be
    # written off as noise. Both directions, on the same invariant.
    noise = {"sW": np.array([0.5, -1e-15, 1.0 + 1e-15])}
    broken = {"sW": np.array([0.5, -0.3, 1.4])}
    bound = bounded_field("sW", 0.0, 1.0, tolerance=1e-10)

    if not bound(noise).passed:
        bad = 1
        print("FAIL noise control: 1e-15 outside [0,1] reported as a violation")
    else:
        print("ok   noise control: 1e-15 outside [0,1] is not a violation")

    result = bound(broken)
    if result.passed:
        bad = 1
        print("FAIL impossibility control: a saturation of -0.3 passed")
    else:
        print(f"ok   impossibility control: caught, max violation "
              f"{result.maximum_violation:.3g} at cells {list(result.locations)}")

    # observe vs strict must actually differ.
    suite = InvariantSuite([bound])
    observed = suite.evaluate(broken, InvariantProfile.OBSERVE)
    if observed and not observed[0].passed:
        print("ok   observe profile: reports the violation and continues")
    else:
        bad = 1
        print("FAIL observe profile: did not report the violation")
    try:
        suite.evaluate(broken, InvariantProfile.STRICT)
    except InvariantViolation as error:
        print(f"ok   strict profile: raised ({len(error.failures)} failure(s))")
    else:
        bad = 1
        print("FAIL strict profile: did not raise on an impossible state")

    # The enum declares three profiles; `evaluate` special-cases only STRICT,
    # so REFERENCE is indistinguishable from OBSERVE. Reported rather than
    # asserted either way: which behaviour it should have is a design decision,
    # and silently testing it as an alias would bless one answer.
    try:
        suite.evaluate(broken, InvariantProfile.REFERENCE)
        print("n/a  reference profile: declared but behaves as observe; the "
              "plan names three distinct profiles and two exist")
    except InvariantViolation:
        print("ok   reference profile: raises like strict")
    return bad


# --------------------------------------------------------------------------
# conservation: Milestone 1.7's gate

# Which adapter belongs to which case. Chosen by model family, because that is
# what an adapter is written against - the plan is explicit that conservation
# semantics must be declared per model rather than guessed globally.
# All three of these are poromechanics models, so they take the poromechanics
# adapter, which delegates to the right fluid basis and - importantly - keeps
# the coupled model's own PoreVolume state function authoritative. Pointing
# them at the bare fluid adapters instead was a first attempt here, and it
# failed at 5.3e-07: mechanical dilation changes the pore volume, and a fluid
# adapter that does not see it reports the difference as lost mass.
ADAPTERS = {
    "p64_water": ("PoromechanicsConservationAdapter", "water, fixed-stress"),
    "p64_coupled": ("PoromechanicsConservationAdapter", "water, fully coupled"),
    # A third family, and the first here that is not poromechanics. The
    # converted black-oil model is a wrapper the general compositional adapter
    # cannot read - it exposes `mixture` rather than `mix` and has no
    # `_flow_props0` - so it gets an adapter written against its own two
    # pseudo-components.
    "spe1_compositional": ("BlackOilPseudoComponentConservationAdapter",
                           "PseudoOil/PseudoGas mass"),
    # The oil-water poromechanics family, which p65_oilwater was supposed to
    # cover and cannot. Same adapter route p65 takes - PoromechanicsConservation
    # -> OilWaterConservation - but on a case whose pore volume stays positive
    # in all 605 cells and whose pressures stay between 220 and 610 bar. The
    # water component carries the check: injection drives the inventory from 0
    # to 160,000 surface m3 over four steps, and 50,000 m3 of it per step
    # balances to about 0.015.
    "norne_mini": ("PoromechanicsConservationAdapter", "oil-water, fully coupled"),
    # p65_oilwater is deliberately absent, and the reason recorded here before
    # was wrong. It said the blocker was that the two fluid adapters disagree
    # about where a fluid lives: OilWaterConservationAdapter read
    # `self.model.fluid.b_w`, while its sibling WaterConservationAdapter
    # resolves `raw_fluid` first. That disagreement was real - this model has
    # both, a `PhaseFluid` under `fluid` and an `ADFluid` under `raw_fluid`,
    # and only the second has `b_w` - and it has since been fixed. It was not
    # what blocked the case. With both adapters agreeing, this one runs and
    # still reports 1.569 on its first step: the same "normalized imbalance of
    # 1.6" once recorded here as evidence for the fluid-lookup theory, which
    # was in fact measured against a pore volume that is negative.
    #
    # The case's own PoreVolume state function returns a negative volume in 74
    # of 80 cells at the *initial* state, and 27-42 in the later ones against
    # an undeformed 3.75 - eleven-fold dilation. Pressure reaches 3.2e10 Pa.
    # Nothing conserves against a negative inventory, so any number this
    # produced would be an artifact of the case rather than a measurement of
    # the adapter.
    #
    # None of that is a porting defect: against
    # ref_p65_run2d_oilwater_fixedstress.mat this run matches MRST to 1.7e-09
    # on pressure and 7.7e-10 on saturation for both recorded steps. It is what
    # MRST's example computes. `mrst_oilwater_bug=False` - PyMRST's correction
    # to MRST's missing cell volume on the Biot term - lowers the pressures
    # about threefold and does not make them physical, so the extreme strain is
    # the configuration, not that bug.
    #
    # So this stays out until the case is reconfigured into a regime where a
    # pore volume stays positive.
}


# Why a case is absent from ADAPTERS, when "no adapter for this model family"
# would be untrue. The default message was actively misleading for
# blackoil_one_phase: its model is a single-phase `WaterModel` and
# `WaterConservationAdapter` fits it exactly.
NOT_CONSERVED = {
    "blackoil_one_phase": (
        "runs two independent settings (open and closed) whose states "
        "`_states_of` concatenates, so consecutive states straddle a seam "
        "where the mean pressure jumps 1.74e7 -> 1.99e7 Pa; the adapter would "
        "fit, the state sequence does not"
    ),
    "adjoint_uplift": (
        "an adjoint gradient with its finite-difference cross-check, not a "
        "forward schedule with a conserved inventory"
    ),
    "p65_oilwater": (
        "its own PoreVolume is negative in 74 of 80 cells; see the note in "
        "ADAPTERS"
    ),
    "packed_compositional": (
        "F0.1 classifies its runner as a deprecated duplicate"
    ),
}


def _perf_timing(run, repeats):
    """Best-of-`repeats` wall time, with the spread that says whether it means
    anything.

    Minimum rather than mean: interference on a shared machine is one-sided -
    other work can only make a run slower - so the fastest observation is the
    closest to the cost being measured. The spread is printed alongside, because
    a difference smaller than it is noise being read as a result.
    """
    run()                       # warm-up, discarded: the first call pays for
                                # imports, caches and first-touch allocation,
                                # and on the smallest case that alone was 65%
                                # of the spread.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter()-start)
    return min(samples), max(samples)-min(samples)


def check_perf(names, repeats: int = 3) -> int:
    """Measure the six runtime/evidence costs section 16 lists.

    Section 16 says to set budgets *after* Foundation 0 measurement, so this
    measures and records rather than asserting a threshold - the stance
    `--conserve` takes, for the same reason: inventing a budget here would
    decide the question instead of answering it.

    Observed and native runs go through **the same builder**, with the observer
    attached by patching the owners' constructors exactly as `--observed` does.
    The first version of this instead ran the case natively and then re-ran it
    through a freshly built `NonLinearSolver`, which on Norne reported
    observation making the run 60% *faster* - it had swapped that case's
    Schur-preconditioned solver for a generic one and measured the difference
    between two solvers.

    Only a gross failure fails: a non-finite time, or evidence costing more than
    100% on top of native, which no measurement noise explains.
    """
    import tempfile
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.research_studio import HDF5ResultStore, SimulationSession

    bad = 0
    for name in names:
        builder, _ = CASES[name]
        native, native_spread = _perf_timing(builder, repeats)

        recorder = _Recorder()
        observed, observed_spread = _perf_timing(
            lambda: _attach_observer(builder, recorder), repeats)
        events = len(recorder.events)//max(repeats, 1)

        peak = _perf_peak_memory(builder)

        store_cost = float("nan")
        store_bytes = 0
        # Restricted to cases a session can drive at all; the rest report n/a
        # rather than a number measured against a different code path.
        driver = _driver_for(name) if name in RESUMABLE else None
        if driver is not None:
            case = builder()
            model, state0 = _model_of(case), _states_of(case)[0]
            schedule = _schedule_of(case)
            scratch = Path(tempfile.mkdtemp(prefix="pymrst_perf_"))
            counter = {"n": 0}

            def session_run():
                counter["n"] += 1
                path = scratch/f"results_{counter['n']}.h5"
                SimulationSession(
                    model=model, initial_state=state0, schedule=schedule,
                    nonlinear_solver=NonLinearSolver(max_iterations=100),
                    run_id=f"run_perf_{name}_{counter['n']}",
                    result_store=HDF5ResultStore(path),
                    checkpoint_every_report=True,
                    code_fingerprint="perf",
                ).run()
                return path

            profile_cost = float("nan")
            try:
                # Driven by the *same* solver the session drives, so the only
                # difference measured is the session and its store. Using the
                # case's own driver here made Norne's store look 61% faster
                # than no store at all - the identical solver-substitution
                # mistake this function's docstring describes, made a second
                # time in a second place.
                bare, _ = _perf_timing(
                    lambda: NonLinearSolver(max_iterations=100)
                    .simulate_schedule(model, state0, schedule), repeats)
                stored, _ = _perf_timing(session_run, repeats)
                store_cost = (stored-bare)/bare
                written = sorted(scratch.glob("results_*.h5"))
                store_bytes = written[-1].stat().st_size if written else 0
                # Section 16 asks for observe/strict profile overhead, and the
                # session evaluates both the invariant suite and the
                # conservation adapter once per accepted state - so they sit on
                # the critical path, not beside it. Measuring the session
                # without them, as this did at first, measures the one
                # configuration that avoids the cost.
                # Marginal over the same session *with* its store, not over
                # the bare solve: the profile adds invariants and conservation
                # on top of everything the store session already does.
                # Measured against `bare` it read +46% where the store alone
                # read +58% - the profiled run apparently cheaper than a strict
                # subset of itself, which is two timing series compared to the
                # wrong baseline rather than a real result.
                profile_cost = _perf_profile_cost(
                    name, model, state0, schedule, scratch, repeats, stored)
            except Exception as error:
                print(f"     ({name}: store path unavailable - "
                      f"{type(error).__name__}: {error})")

        # Section 16's sixth item, measured directly rather than inferred from
        # the store's total size: one checkpoint's own bytes and the wall time
        # to commit it. Restart cadence is chosen against these two numbers -
        # a checkpoint worth writing every report step and one worth writing
        # hourly are different objects.
        commit_ms = checkpoint_kb = float("nan")
        if driver is not None:
            try:
                commit_ms, checkpoint_kb = _perf_checkpoint(
                    name, model, state0, schedule, repeats)
            except Exception as error:
                print(f"     ({name}: checkpoint cost unavailable - "
                      f"{type(error).__name__}: {error})")

        if not np.isfinite(native) or not np.isfinite(observed) or native <= 0:
            bad = 1
            print(f"FAIL {name}: a timing was not finite")
            continue
        observe_cost = (observed-native)/native
        worst = max(observe_cost, store_cost if np.isfinite(store_cost) else 0.0)
        if worst > 1.0:
            bad = 1
            print(f"FAIL {name}: evidence costs {worst:+.0%} over native, "
                  f"beyond what measurement noise explains")
            continue
        store_text = ("n/a" if not np.isfinite(store_cost)
                      else f"{store_cost:+.0%} ({store_bytes/1024:.0f} kB)")
        profile_text = ("n/a" if not np.isfinite(profile_cost)
                        else f"{profile_cost:+.0%}")
        checkpoint_text = ("n/a" if not np.isfinite(commit_ms)
                           else f"{checkpoint_kb:.0f} kB in {commit_ms:.1f} ms")
        print(f"ok   {name}: native {native*1e3:.0f} ms "
              f"(spread {100*native_spread/native:.0f}%); observe "
              f"{observe_cost:+.0%}; store+checkpoint {store_text}; "
              f"observe profile {profile_text}; checkpoint {checkpoint_text}; "
              f"{events} events; peak {peak/1e6:.0f} MB [measured, not asserted]")
    return bad


def _perf_profile_cost(name, model, state0, schedule, scratch, repeats, baseline):
    """Cost of the observe profile: invariants and conservation, per state.

    Returns NaN when the case has no conservation adapter mapped, rather than
    reporting an invariants-only figure under a name that promises both.

    `baseline` is the session-with-store time, so what comes back is the
    marginal cost of evaluating invariants and conservation once per accepted
    state - which is what section 16 means by profile overhead.
    """
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.research_studio import (
        HDF5ResultStore, InvariantSuite, SimulationSession, bounded_field,
        finite_field,
    )
    from pymrst.research_studio.evidence import conservation as C

    if name not in ADAPTERS:
        return float("nan")
    adapter = getattr(C, ADAPTERS[name][0])(model)
    suite = InvariantSuite([finite_field("pressure")]
                           + ([bounded_field("sW", 0.0, 1.0)]
                              if "sW" in state0 else []))
    counter = {"n": 0}

    def run_profiled():
        counter["n"] += 1
        SimulationSession(
            model=model, initial_state=state0, schedule=schedule,
            nonlinear_solver=NonLinearSolver(max_iterations=100),
            run_id=f"run_prof_{name}_{counter['n']}",
            result_store=HDF5ResultStore(scratch/f"prof_{counter['n']}.h5"),
            checkpoint_every_report=True, code_fingerprint="perf",
            conservation_adapter=adapter, invariant_suite=suite,
            evidence_profile="observe",
        ).run()

    profiled, _ = _perf_timing(run_profiled, repeats)
    return (profiled-baseline)/baseline


def _perf_checkpoint(name, model, state0, schedule, repeats):
    """One checkpoint's size in kB and its commit time in ms.

    Sized from the record actually committed, not from the file: an HDF5 file
    carries its own structure, and the question a restart cadence is chosen
    against is what one checkpoint costs, not what the container weighs.
    """
    import tempfile
    from pymrst.research_studio import HDF5ResultStore
    from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId
    from pymrst.research_studio.storage.checkpoint import (
        CheckpointRecord, default_model_fingerprint, default_solver_fingerprint,
        schedule_fingerprint,
    )
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    solver = NonLinearSolver(max_iterations=100)
    record = _checkpoint_at("perf", name, 1, schedule, state0, model, solver)
    payload = record.to_dict()
    kilobytes = len(json.dumps(payload, default=lambda v: np.asarray(v).tolist()))/1024

    scratch = Path(tempfile.mkdtemp(prefix="pymrst_ckpt_"))
    store = HDF5ResultStore(scratch/"ckpt.h5")
    run = RunId(f"run_ckpt_{name}")
    store.begin_run(run, {"case": name})
    def commit(counter=[0]):
        counter[0] += 1
        store.commit_checkpoint(run, CheckpointId(f"c{counter[0]}"), payload)
    best, _ = _perf_timing(commit, repeats)
    return best*1e3, kilobytes


def _perf_peak_memory(builder):
    import tracemalloc
    tracemalloc.start()
    try:
        builder()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def check_conserve(names, gross: float = 1e-2) -> int:
    """Measure the conservation imbalance the adapters report.

    Milestone 1.7's gate asks that the primary families' adapters pass, but
    "pass" needs a tolerance, and section 16 is explicit that budgets are set
    *after* Foundation 0 measurement rather than guessed beforehand. Inventing
    one here would decide the question instead of answering it.

    So this measures and records. Two numbers make the point: on p64_water the
    adapter reports a normalized imbalance of 3.6e-06 while the flow solve
    reports a material-balance residual of 5.4e-09. Those are normalized
    differently and are not directly comparable - which is exactly why the
    figure needs pinning and then explaining, not asserting against a threshold
    picked to make it pass.

    Only a gross failure fails here: a non-finite imbalance, or one above 1%,
    which no normalization difference explains.
    """
    from pymrst.research_studio.evidence import conservation as C

    bad = 0
    for name in names:
        if name not in ADAPTERS:
            print(f"n/a  {name}: {NOT_CONSERVED.get(name, 'no adapter mapped to this model family')}")
            continue
        adapter_name, description = ADAPTERS[name]

        builder, _ = CASES[name]
        case = builder()
        states = _states_of(case)
        schedule = _schedule_of(case)
        adapter = getattr(C, adapter_name)(_model_of(case))

        worst, reports = 0.0, []
        for step, dt in enumerate(schedule):
            report = C.conservation_report(adapter, states[step], states[step + 1],
                                           float(dt))
            worst = max(worst, report.maximum_normalized_imbalance)
            reports.append(report)

        if not np.isfinite(worst):
            bad = 1
            print(f"FAIL {name}: imbalance is not finite")
        elif worst > gross:
            bad = 1
            print(f"FAIL {name} ({description}): worst normalized imbalance "
                  f"{worst:.3e} exceeds {gross:g}, which no normalization "
                  f"difference explains")
        else:
            print(f"ok   {name} ({description}): worst normalized imbalance "
                  f"{worst:.3e} over {len(schedule)} steps, "
                  f"{len(adapter.component_names)} component(s) "
                  f"[measured, not asserted]")
    return bad


# --------------------------------------------------------------------------
# failure bundles: Milestone 1.10's gate

def _checkpoint_at(checkpoint_id, name, cut, schedule, state, model, solver):
    """The mid-schedule checkpoint `--resume` and `--bundle` both build.

    Extracted because the two call sites held byte-identical copies of it, and
    that cost real time: a source mutation written to falsify `--resume`
    matched the *bundle* copy first, so `--resume` kept passing and the check
    looked vacuous for all five cases. It was not - the same mutation applied
    here fails every one. Two identical lines in two functions is a hazard for
    anything that edits code by matching text, this file's own falsification
    workflow included.
    """
    from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId
    from pymrst.research_studio.storage.checkpoint import (
        CheckpointRecord, default_model_fingerprint, default_solver_fingerprint,
        schedule_fingerprint,
    )
    return CheckpointRecord(
        checkpoint_id=CheckpointId(checkpoint_id),
        source_run_id=RunId(f"run_{name}"),
        completed_report_steps=cut,
        accepted_elapsed_time=float(np.sum(schedule[:cut])),
        # Arrays, not lists. The HDF5 store round-trips numpy arrays as
        # arrays, so `.tolist()` here would test a shape the real persistence
        # path never produces.
        state={k: np.asarray(v).copy() for k, v in state.items()},
        schedule_fingerprint=schedule_fingerprint(schedule),
        model_fingerprint=default_model_fingerprint(model),
        solver_fingerprint=default_solver_fingerprint(solver),
        code_fingerprint="characterization",
    )


def check_bundle(names) -> int:
    """Does a bundle reproduce the failing step without the earlier schedule?

    That is Milestone 1.10's stated requirement, and the existing unit test
    answers it for a toy model whose solver adds `dt` to a scalar. This writes
    a bundle from a real poromechanics run, reloads it **from disk** in a fresh
    context, and replays only the remaining steps.

    The comparison is against the uninterrupted run's states at the same
    positions. Exact equality, for the same reason resume is exact: the replay
    performs the same arithmetic in the same order, so any difference means the
    bundle lost something.
    """
    import tempfile
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
    from pymrst.research_studio.evidence.manifest import capture_manifest
    from pymrst.research_studio.evidence.report import RunIdentity, SimulationEvidence
    from pymrst.research_studio.evidence.failure_bundle import (
        load_failure_bundle, prepare_failure_replay, write_failure_bundle,
    )
    from pymrst.research_studio.runtime.identifiers import (
        CaseId, CheckpointId, RunId,
    )
    from pymrst.research_studio.storage.checkpoint import (
        CheckpointRecord, default_model_fingerprint, default_solver_fingerprint,
        schedule_fingerprint,
    )

    bad = 0
    for name in names:
        if name not in RESUMABLE:
            print(f"n/a  {name}: not session-drivable, so no bundle to write")
            continue

        driver = _driver_for(name)
        if driver is None:
            print(f"n/a  {name}: no schedule driver identified")
            continue
        driver_name, run = driver

        builder, _ = CASES[name]
        case = builder()
        model, state0 = case.model, _states_of(case)[0]
        schedule = np.asarray(case.dts, dtype=float)
        cut = max(1, len(schedule) // 2)
        solver = NonLinearSolver(max_iterations=100)

        uninterrupted, _ = run(model, state0, schedule)
        prefix, _ = run(model, state0, schedule[:cut])

        scratch = Path(tempfile.mkdtemp(prefix="pymrst_bundle_"))
        source = scratch / "CASE.DATA"
        source.write_text("RUNSPEC\n", encoding="utf-8")
        manifest = capture_manifest(f"run_{name}", code_fingerprint="characterization",
                                    inputs=[source])
        checkpoint = _checkpoint_at("bundle", name, cut, schedule, prefix[-1],
                                    model, solver)
        path = write_failure_bundle(
            scratch / f"{name}.pymrst-failure", manifest=manifest,
            evidence=SimulationEvidence(RunIdentity(RunId(f"run_{name}"), CaseId(name))),
            checkpoint=checkpoint,
            failing_step={"report_step": cut, "reason": "characterization"},
        )

        # Independently loadable: read it back with nothing but the file.
        reloaded = load_failure_bundle(path)
        failures = []
        if reloaded.get("failing_step", {}).get("report_step") != cut:
            failures.append("the reloaded bundle lost its failing step")

        replay = prepare_failure_replay(
            path, model=model, nonlinear_solver=solver,
            full_schedule=schedule, code_fingerprint="characterization",
            # `state_factory` defaults to `dict`, which is fine for a toy model
            # and not for a real one: poromechanics needs a `State` with its
            # cache, and a plain dict dies on the first update.
            state_factory=type(state0),
        )
        if len(replay.remaining_schedule) != len(schedule) - cut:
            failures.append(f"remaining schedule {len(replay.remaining_schedule)}, "
                            f"expected {len(schedule) - cut}")
        # Replay through the case's own driver, for the same reason `--resume`
        # does: `replay.run()` would use the bundled solver, which is the wrong
        # method for a fixed-stress model.
        replayed, _ = run(model, replay.state, replay.remaining_schedule)

        for field, want in sorted(_state_fields(uninterrupted[-1]).items()):
            want = np.asarray(want, dtype=float)
            got = np.asarray(_state_fields(replayed[-1]).get(field, []), dtype=float)
            if want.shape != got.shape:
                failures.append(f"{field}: shape {want.shape} became {got.shape}")
            elif not np.array_equal(want, got):
                scale = np.maximum(np.abs(want), np.finfo(float).tiny)
                failures.append(f"{field}: worst relative difference "
                                f"{float(np.max(np.abs(want - got) / scale)):.3e}")
        if failures:
            bad = 1
            print(f"FAIL {name}: bundle replay differs from the original run")
            for f in failures[:6]:
                print(f"       {f}")
        else:
            size = path.stat().st_size
            print(f"ok   {name}: bundle ({size} bytes) reloaded from disk, "
                  f"replayed steps {cut}-{len(schedule)} exactly via "
                  f"{driver_name}, without running steps 0-{cut}")
    return bad


# --------------------------------------------------------------------------
# handler isolation: Milestone 1.2's gate

class _HostileObserver:
    """An observer that actively tries to interfere, and fails loudly.

    `--observed` proves a *well-behaved* recorder changes nothing, which is the
    easy half. The covenant's claim is stronger: a handler *cannot* change a
    numerical decision. That has to be tested against a handler which tries.

    So this one mutates every payload it is handed, replaces the numbers in it,
    and raises on some events. If the run afterwards is still bit-identical to
    the baseline, the isolation is real rather than merely conventional.

    It also records the *types* it was handed, because that is where the real
    guarantee lives. Observed in practice the payload surface is bools, ints,
    floats and a residuals dict - no arrays and no state object - so a handler
    has nothing live to corrupt in the first place. That is a stronger property
    than "mutations were ignored", and `check_hostile` asserts it, so the day
    someone passes a state array to an observer this fails rather than quietly
    widening the surface.
    """

    def __init__(self):
        self.events, self.mutations, self.raised = [], 0, 0
        self.payload_types: set[str] = set()

    def __call__(self, kind, payload):
        self.events.append(kind)
        try:
            for key, value in list(payload.items()):
                self.payload_types.add(type(value).__name__)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    payload[key] = float(value) * -1e6 + 1.0
                    self.mutations += 1
                elif isinstance(value, np.ndarray):
                    value.fill(np.nan)          # in-place: the worst case
                    self.mutations += 1
                elif isinstance(value, dict):
                    value.clear()
                    self.mutations += 1
            payload["injected_by_handler"] = "should never be read"
        except Exception:
            pass                                 # a frozen payload is a pass here
        if len(self.events) % 3 == 0:
            self.raised += 1
            raise RuntimeError("hostile handler")


def check_hostile(names) -> int:
    """Can a handler that tries to interfere change the answer?"""
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    bad = 0
    for name in names:
        path = baseline_path(name)
        if not path.exists():
            print(f"SKIP {name}: no baseline (run --freeze)")
            continue

        builder, reference = CASES[name]
        hostile = _HostileObserver()
        result = _attach_observer(builder, hostile)

        if not hostile.events:
            print(f"n/a  {name}: constructs no instrumented owner")
            continue

        failures = compare(_record_from(name, result, reference, 0.0),
                           json.loads(path.read_text()))

        # The surface itself is the guarantee: a handler cannot corrupt what it
        # is never given. Anything richer than a scalar or a plain container
        # here means live numerical state reached a handler.
        allowed = {"bool", "int", "float", "str", "dict", "NoneType"}
        leaked = sorted(hostile.payload_types - allowed)
        if leaked:
            bad = 1
            failures.append(f"payload exposes live objects to handlers: {leaked}")
        if failures:
            bad = 1
            print(f"FAIL {name}: a hostile handler changed the result")
            for f in failures[:6]:
                print(f"       {f}")
        else:
            print(f"ok   {name}: unchanged after {hostile.mutations} attempted "
                  f"mutations and {hostile.raised} exceptions across "
                  f"{len(hostile.events)} events")
    return bad


# --------------------------------------------------------------------------
# failure classification: Milestone 1.9's gate

def _residual_histories(node, out=None):
    """Every residual history buried anywhere in a decisions tree.

    The reports nest differently per model - a fixed-stress step holds
    `splitting.outer[i].flow.residuals`, a plain one holds them nearer the top
    - so this walks rather than assuming a shape.
    """
    out = [] if out is None else out
    if isinstance(node, dict):
        history = node.get("residuals")
        if isinstance(history, list) and history and isinstance(history[0], dict):
            out.append(history)
        for value in node.values():
            _residual_histories(value, out)
    elif isinstance(node, list):
        for value in node:
            _residual_histories(value, out)
    return out


def check_classify(names) -> int:
    """Does the failure classifier read converged runs as converged?

    Milestone 1.9's gate asks that known residual behaviour is classified
    correctly. Every frozen case here converged, so the classifier must report
    no failure for any of them: a classifier that cries divergence on a healthy
    run is worse than none, because every real failure then looks like noise.

    The negative half matters just as much and is checked alongside: histories
    built to stall, diverge and go non-finite must be classified as such. A
    classifier that returns UNKNOWN for everything would pass the first half
    perfectly.
    """
    from pymrst.research_studio.evidence.failure_analysis import (
        FailureKind, analyze_residual_history,
    )

    bad = 0
    for name in names:
        path = baseline_path(name)
        if not path.exists():
            print(f"SKIP {name}: no baseline (run --freeze)")
            continue
        histories = _residual_histories(json.loads(path.read_text())["decisions"])
        if not histories:
            print(f"n/a  {name}: its reports carry no residual history")
            continue

        misread = []
        for i, history in enumerate(histories):
            analysis = analyze_residual_history(history, converged=True)
            failing = [k for k in analysis.classifications
                       if k not in (FailureKind.UNKNOWN,)]
            if failing:
                misread.append(f"history {i}: {[str(k) for k in failing]}")
        if misread:
            bad = 1
            print(f"FAIL {name}: converged runs classified as failing")
            for m in misread[:5]:
                print(f"       {m}")
        else:
            print(f"ok   {name}: {len(histories)} converged histories, "
                  f"none misread as a failure")

    # The negative controls. Without them the whole check passes on a
    # classifier that says nothing about anything.
    #
    # One per kind the work package names, because "the classifier works" is
    # not a claim any single control supports: stall, divergence and non-finite
    # were covered while oscillation, chattering, the residual floor and linear
    # failure were not, and a classifier silent on those four would have passed.
    # Each history is built to be *specific* - the oscillating one improves by
    # 99% across its last window so it cannot also register as a stall, and the
    # max-iterations one decays cleanly so it triggers nothing else.
    synthetic = {
        "stall": ([{"CNV": 1e-3}, {"CNV": 9.9e-4}, {"CNV": 9.8e-4},
                   {"CNV": 9.75e-4}, {"CNV": 9.7e-4}, {"CNV": 9.69e-4}],
                  (), FailureKind.STALL),
        "divergence": ([{"CNV": 1e-6}, {"CNV": 1e-4}, {"CNV": 1e-1}, {"CNV": 1e3}],
                       (), FailureKind.DIVERGENCE),
        "nonfinite": ([{"CNV": 1e-3}, {"CNV": float("nan")}], (),
                      FailureKind.NONFINITE),
        # A linear solve that failed reports itself by name rather than by any
        # shape of the residual curve.
        "linear failure": ([{"CNV": 1e-3, "linear_failure": 1.0}], (),
                           FailureKind.LINEAR_FAILURE),
        # Flat to within 2% across the window: converging to a floor it will
        # not get below, which is a different diagnosis from slowing down.
        "residual floor": ([{"CNV": 1e-6}, {"CNV": 1.005e-6}, {"CNV": 0.995e-6},
                            {"CNV": 1.002e-6}, {"CNV": 1e-6}], (),
                           FailureKind.RESIDUAL_FLOOR),
        # Alternating by two orders of magnitude each iteration.
        "oscillation": ([{"CNV": 1e-2}, {"CNV": 1e-4}, {"CNV": 1e-2},
                         {"CNV": 1e-4}, {"CNV": 1e-2}, {"CNV": 1e-4}], (),
                        FailureKind.OSCILLATION),
        # Not in the residuals at all: a well flipping between rate and BHP
        # control every iteration, which is what `control_history` is for.
        "control chattering": ([], ("bhp", "rate", "bhp", "rate"),
                               FailureKind.CONTROL_CHATTERING),
        # Decaying cleanly and simply not finished. Nothing is wrong with this
        # curve, so every other detector must stay quiet.
        "max iterations": ([{"CNV": 1e-1}, {"CNV": 1e-3}, {"CNV": 1e-5}], (),
                           FailureKind.MAX_ITERATIONS),
    }
    for label, (history, controls, expected) in synthetic.items():
        analysis = analyze_residual_history(history, converged=False,
                                            control_history=controls)
        if expected in analysis.classifications:
            print(f"ok   negative control {label}: classified {expected}")
        else:
            bad = 1
            print(f"FAIL negative control {label}: expected {expected}, got "
                  f"{[str(k) for k in analysis.classifications]}")

    # The two reporting outputs the work package asks for by name, which no
    # classification check exercises: "worst owner/location reporting" and
    # "convergence-rate analysis". Both are numbers an engineer reads off a
    # failed run, and both can be wrong while every kind above is right.
    located = analyze_residual_history(
        [{"CNV": 1e-4, "MB": 1e-6},
         {"CNV": 1e-5, "MB": 7.5e-2},      # the worst value, at iteration 1
         {"CNV": 1e-6, "MB": 1e-7}],
        converged=False,
    )
    if (located.worst_residual, located.worst_iteration) == ("MB", 1) \
            and located.worst_value == 7.5e-2:
        print("ok   worst owner/location: MB at iteration 1, 7.500e-02")
    else:
        bad = 1
        print(f"FAIL worst owner/location: expected MB at iteration 1 "
              f"(7.5e-02), got {located.worst_residual} at "
              f"{located.worst_iteration} ({located.worst_value})")

    # Ten-fold reduction per iteration: the rate must come out at -1 decade
    # and the median ratio at 0.1. A sign error here reads a converging run as
    # a diverging one.
    rated = analyze_residual_history(
        [{"CNV": 10.0**(-k)} for k in range(6)], converged=True,
    )
    if (rated.estimated_log10_rate is not None
            and abs(rated.estimated_log10_rate+1.0) < 1e-9
            and rated.median_convergence_ratio is not None
            and abs(rated.median_convergence_ratio-0.1) < 1e-9):
        print(f"ok   convergence rate: {rated.estimated_log10_rate:.1f} decades "
              f"per iteration, median ratio "
              f"{rated.median_convergence_ratio:.3f}")
    else:
        bad = 1
        print(f"FAIL convergence rate: expected -1.0 decades and ratio 0.1, "
              f"got {rated.estimated_log10_rate} and "
              f"{rated.median_convergence_ratio}")

    # A real failure, not a synthetic one. Everything above is either a healthy
    # run or a hand-written array; the work package also asks for validation
    # against real failures, and those are the histories a classifier exists
    # for. Induced rather than retained, by starving a real Norne solve of
    # iterations - the physics and the residuals are the model's own.
    #
    # This is also the regression test for what it uncovered. `solve_timestep`
    # built a fresh StepReport on both its cut-success and failure exits and
    # left `residuals` empty, so a failed step reported no history and
    # classified as `unknown`: the one case failure analysis is for was the one
    # case with nothing to analyze.
    try:
        from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
        builder, _ = CASES["norne_mini"]
        case = builder()
        model, state0 = _model_of(case), _states_of(case)[0]
        native = case.model.acceptance_solver
        starved = NonLinearSolver(
            max_iterations=3, max_timestep_cuts=1,
            linear_solver=getattr(native, "linear_solver", None),
            use_line_search=getattr(native, "use_line_search", False),
            use_relaxation=getattr(native, "use_relaxation", False),
        )
        _, report = starved.solve_timestep(
            model, state0, float(_schedule_of(case)[0]),
        )
        history = [dict(r) for r in report.residuals]
        analysis = analyze_residual_history(history, converged=bool(report.converged))
        definite = [k for k in analysis.classifications if k is not FailureKind.UNKNOWN]
        if report.converged:
            bad = 1
            print("FAIL real failure: the starved solve converged, so this "
                  "control no longer exercises a failure")
        elif not history:
            bad = 1
            print("FAIL real failure: a failed step carried no residual "
                  "history, so nothing reaches the classifier")
        elif not definite:
            bad = 1
            print(f"FAIL real failure: {len(history)} iterations classified "
                  f"only as {[str(k) for k in analysis.classifications]}")
        else:
            print(f"ok   real failure (Norne, 3-iteration cap): "
                  f"{len(history)} real iterations classified "
                  f"{[str(k) for k in definite]}, worst {analysis.worst_residual} "
                  f"at iteration {analysis.worst_iteration}")
    except Exception as error:                      # pragma: no cover
        bad = 1
        print(f"FAIL real failure: could not induce one ({type(error).__name__}: {error})")
    return bad


# --------------------------------------------------------------------------
# resume equivalence: Milestone 1.6's gate

# Cases whose (model, initial state, timesteps) can be handed to
# `simulate_schedule` directly. The others build their schedule internally or
# run a splitting loop of their own, so a session cannot drive them.
def _driver_for(name):
    """The schedule driver a case actually uses, or None.

    Driving a fixed-stress model through `NonLinearSolver` would solve it fully
    coupled - a different method, so a resume test built that way would prove
    nothing about the path the case really runs. Each case is therefore
    resumed through its own driver.
    """
    from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver

    if name in ("p64_water", "p65_oilwater"):
        from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models import (
            simulate_fixed_stress_schedule,
        )
        return ("fixed-stress splitting",
                lambda model, state, sched: simulate_fixed_stress_schedule(
                    model, state, sched))
    if name in ("p64_coupled", "adjoint_uplift"):
        return ("NonLinearSolver.simulate_schedule",
                lambda model, state, sched: NonLinearSolver(
                    max_iterations=100).simulate_schedule(model, state, sched))
    if name == "norne_mini":
        # The runner's own solver, not a fresh one: its Schur linear solver,
        # iteration cap and line-search/relaxation switches are chosen by fluid
        # inside `run`, so a plain NonLinearSolver(max_iterations=100) resumes a
        # solve the case does not actually perform.
        #
        # Worth being exact about what that buys, because substituting the
        # generic solver here was tried and `--resume` still passed. It would:
        # `check_resume` drives the uninterrupted, prefix and resumed runs
        # through this same callable, so any driver makes the comparison
        # self-consistent. The check cannot detect a wrong driver, and no
        # arrangement of it could. Using the real one is therefore a statement
        # about what is being characterized, not a defence the check enforces.
        return ("run_norne_geomech solver.simulate_schedule",
                lambda model, state, sched: model.acceptance_solver
                .simulate_schedule(model, state, sched))
    return None


RESUMABLE = ("p64_coupled", "p64_water", "p65_oilwater", "adjoint_uplift",
             "norne_mini")


def check_resume(names) -> int:
    """Does resuming from a checkpoint reproduce an uninterrupted run?

    The existing unit test answers this for `AccumulatingModel`, where `x`
    advances by `dt`. That proves the schedule bookkeeping and nothing about a
    real model: an accumulator has no Newton loop to restart, no well solution
    to carry and no state fields that could be dropped. This runs the same
    question against real poromechanics, through the same `restart_payload`
    the session uses.

    Equality is exact. A resumed run is not an approximation of an
    uninterrupted one - it is the same arithmetic in the same order - so any
    difference at all means state was lost across the boundary.
    """
    from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId
    from pymrst.research_studio.storage.checkpoint import (
        CheckpointRecord, default_model_fingerprint, default_solver_fingerprint,
        restart_payload, schedule_fingerprint,
    )

    bad = 0
    for name in names:
        if name not in RESUMABLE:
            print(f"n/a  {name}: builds its schedule internally, so a session "
                  f"cannot drive it")
            continue

        driver = _driver_for(name)
        if driver is None:
            print(f"n/a  {name}: no schedule driver identified")
            continue
        driver_name, run = driver

        builder, _ = CASES[name]
        case = builder()
        model, state0 = case.model, _states_of(case)[0]
        schedule = np.asarray(case.dts, dtype=float)
        cut = max(1, len(schedule) // 2)

        uninterrupted, _ = run(model, state0, schedule)
        prefix, _ = run(model, state0, schedule[:cut])

        from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
        solver = NonLinearSolver(max_iterations=100)
        record = _checkpoint_at("characterization", name, cut, schedule,
                                prefix[-1], model, solver)
        restored, remaining = restart_payload(
            record, schedule=schedule,
            model_fingerprint=default_model_fingerprint(model),
            solver_fingerprint=default_solver_fingerprint(solver),
            code_fingerprint="characterization",
            state_factory=type(state0),
        )
        resumed, _ = run(model, restored, remaining)

        failures = []
        if len(remaining) != len(schedule) - cut:
            failures.append(f"remaining schedule {len(remaining)} steps, "
                            f"expected {len(schedule) - cut}")
        for field in sorted(_state_fields(uninterrupted[-1])):
            want = np.asarray(_state_fields(uninterrupted[-1])[field], dtype=float)
            got = np.asarray(_state_fields(resumed[-1]).get(field, []), dtype=float)
            if want.shape != got.shape:
                failures.append(f"{field}: shape {want.shape} became {got.shape}")
            elif not np.array_equal(want, got):
                scale = np.maximum(np.abs(want), np.finfo(float).tiny)
                failures.append(f"{field}: worst relative difference "
                                f"{float(np.max(np.abs(want - got) / scale)):.3e} "
                                f"(resume must be exact)")
        if failures:
            bad = 1
            print(f"FAIL {name}: resume differs from uninterrupted")
            for f in failures[:6]:
                print(f"       {f}")
        else:
            fields = len(_state_fields(uninterrupted[-1]))
            print(f"ok   {name}: resume from step {cut}/{len(schedule)} exact "
                  f"across {fields} state fields, via {driver_name}")
    return bad


# --------------------------------------------------------------------------
# comparison

def _compare_arrays(path, a, b, failures):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        failures.append(f"{path}: shape {b.shape} became {a.shape}")
        return
    if a.size == 0:
        return
    # Relative, atol=0. See the module docstring.
    if not np.allclose(a, b, rtol=RTOL, atol=0.0):
        scale = np.maximum(np.abs(b), np.finfo(float).tiny)
        worst = float(np.max(np.abs(a - b) / scale))
        failures.append(f"{path}: worst relative difference {worst:.3e} (rtol {RTOL:g})")


def compare(current: dict, baseline: dict) -> list[str]:
    """Differences that matter, decisions first."""
    failures: list[str] = []

    if current["shape"] != baseline["shape"]:
        failures.append(f"shape {baseline['shape']} became {current['shape']}")

    if current["decisions"] != baseline["decisions"]:
        for i, (now, was) in enumerate(zip(current["decisions"], baseline["decisions"])):
            for key in sorted(set(now) | set(was)):
                if now.get(key) != was.get(key):
                    failures.append(
                        f"solver decision changed at step {i}: "
                        f"{key} {was.get(key)!r} became {now.get(key)!r}")

    for field in sorted(baseline.get("metrics", {})):
        was = baseline["metrics"][field]
        if field not in current.get("metrics", {}):
            failures.append(f"metrics.{field}: gone")
        else:
            _compare_arrays(f"metrics.{field}", current["metrics"][field], was, failures)

    for label in sorted(baseline["states"]):
        was = baseline["states"][label]
        now = current["states"].get(label, {})
        for field in sorted(was):
            if field not in now:
                failures.append(f"states.{label}.{field}: gone")
            else:
                _compare_arrays(f"states.{label}.{field}", now[field], was[field], failures)
    return failures


def baseline_path(name: str) -> Path:
    return BASELINE_DIR / f"{name}.json"


def freeze(names) -> int:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        data = record(name)
        baseline_path(name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"froze {name}: {data['shape']['cells']} cells, "
              f"{len(data['decisions'])} steps, {data['timing']['seconds']}s")
    return 0


def check(names) -> int:
    bad = 0
    for name in names:
        path = baseline_path(name)
        if not path.exists():
            print(f"SKIP {name}: no baseline (run --freeze)")
            continue
        failures = compare(record(name), json.loads(path.read_text()))
        if failures:
            bad = 1
            print(f"FAIL {name}")
            for f in failures[:10]:
                print(f"       {f}")
            if len(failures) > 10:
                print(f"       ... and {len(failures) - 10} more")
        else:
            print(f"ok   {name}")
    return bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", action="store_true", help="record baselines")
    parser.add_argument("--check", action="store_true", help="compare against baselines")
    parser.add_argument("--invariants", action="store_true",
                        help="evaluate invariants on real states and check the "
                             "profiles differ (Milestone 1.8's gate)")
    parser.add_argument("--perf", action="store_true",
                        help="measure the six runtime/evidence costs section 16 "
                             "lists, so budgets can be set against numbers")
    parser.add_argument("--conserve", action="store_true",
                        help="run each model family's conservation adapter over "
                             "the frozen cases (Milestone 1.7's gate)")
    parser.add_argument("--bundle", action="store_true",
                        help="write a failure bundle from a real run, reload it "
                             "and replay (Milestone 1.10's gate)")
    parser.add_argument("--hostile", action="store_true",
                        help="attach a handler that mutates payloads and raises, "
                             "and prove the run is unchanged (Milestone 1.2's gate)")
    parser.add_argument("--classify", action="store_true",
                        help="run the failure classifier over every frozen "
                             "residual history (Milestone 1.9's gate)")
    parser.add_argument("--resume", action="store_true",
                        help="checkpoint mid-schedule and prove the resumed run "
                             "matches an uninterrupted one (Milestone 1.6's gate)")
    parser.add_argument("--observed", action="store_true",
                        help="re-run with an observer attached and compare "
                             "(Milestone 1.3's gate)")
    parser.add_argument("cases", nargs="*", default=None)
    args = parser.parse_args(argv)

    names = args.cases or sorted(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {', '.join(unknown)}; have {', '.join(sorted(CASES))}")
    if args.freeze:
        return freeze(names)
    if args.invariants:
        return check_invariants(names)
    if args.perf:
        return check_perf(names)
    if args.conserve:
        return check_conserve(names)
    if args.bundle:
        return check_bundle(names)
    if args.hostile:
        return check_hostile(names)
    if args.classify:
        return check_classify(names)
    if args.resume:
        return check_resume(names)
    if args.observed:
        return check_observed(names)
    if args.check:
        return check(names)
    parser.error("choose --freeze or --check")


if __name__ == "__main__":
    raise SystemExit(main())
