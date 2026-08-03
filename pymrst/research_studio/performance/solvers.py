"""M4.6 availability-aware solver qualification, never silent selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import time
from typing import Callable, Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True, slots=True)
class SolverDescriptor:
    solver_id: str
    solve: Callable | None
    dependency: str
    devices: tuple[str, ...] = ("cpu",)
    matrix_kinds: tuple[str, ...] = ("general",)
    optional: bool = False

    @property
    def available(self):
        return self.solve is not None and (
            not self.dependency or find_spec(self.dependency) is not None)


@dataclass(frozen=True, slots=True)
class LinearSolveEvidence:
    solver_id: str
    available: bool
    elapsed_seconds: float | None
    relative_residual: float | None
    maximum_reference_error: float | None
    passed: bool
    reason: str = ""


class SolverLaboratory:
    def __init__(self, solvers: Iterable[SolverDescriptor]):
        self.solvers = {value.solver_id: value for value in solvers}

    def benchmark(self, matrix, rhs, *, device="cpu", tolerance=1e-10):
        matrix = sp.csr_matrix(matrix)
        rhs = np.asarray(rhs, dtype=float)
        reference = spla.spsolve(matrix.tocsc(), rhs)
        norm_rhs = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        results = {}
        for solver_id in sorted(self.solvers):
            descriptor = self.solvers[solver_id]
            if not descriptor.available:
                results[solver_id] = LinearSolveEvidence(
                    solver_id, False, None, None, None, False,
                    f"optional dependency unavailable: {descriptor.dependency}")
                continue
            if device not in descriptor.devices:
                results[solver_id] = LinearSolveEvidence(
                    solver_id, True, None, None, None, False,
                    f"device {device} unsupported")
                continue
            started = time.perf_counter()
            solution = np.asarray(descriptor.solve(matrix, rhs), dtype=float)
            elapsed = time.perf_counter()-started
            residual = float(np.linalg.norm(matrix@solution-rhs)/norm_rhs)
            error = float(np.max(np.abs(solution-reference)))
            results[solver_id] = LinearSolveEvidence(
                solver_id, True, elapsed, residual, error,
                residual <= tolerance and error <= tolerance)
        return results


def _cg(matrix, rhs):
    value, info = spla.cg(matrix, rhs, rtol=1e-12, atol=0.0)
    if info != 0:
        raise RuntimeError(f"CG failed with status {info}")
    return value


def _gmres(matrix, rhs):
    value, info = spla.gmres(matrix, rhs, rtol=1e-12, atol=0.0)
    if info != 0:
        raise RuntimeError(f"GMRES failed with status {info}")
    return value


def reference_solver_registry():
    return SolverLaboratory((
        SolverDescriptor("scipy-direct", lambda a, b: spla.spsolve(a.tocsc(), b), "scipy"),
        SolverDescriptor("scipy-cg", _cg, "scipy", matrix_kinds=("spd",)),
        SolverDescriptor("scipy-gmres", _gmres, "scipy"),
        SolverDescriptor("petsc", None, "petsc4py", optional=True),
        SolverDescriptor("cupy", None, "cupy", devices=("cuda",), optional=True),
        SolverDescriptor("amgx", None, "pyamgx", devices=("cuda",), optional=True),
    ))


__all__ = ["LinearSolveEvidence", "SolverDescriptor", "SolverLaboratory",
           "reference_solver_registry"]
