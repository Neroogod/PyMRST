"""Measured backend and solver qualification for M4.5/M4.6."""

from .laboratory import (
    BackendDescriptor, BenchmarkMeasurement, BackendQualification,
    PerformanceLaboratory, add_at_reduction, bincount_reduction,
    sorted_segment_reduction, sparse_construction_benchmark,
)
from .solvers import (
    LinearSolveEvidence, SolverDescriptor, SolverLaboratory,
    reference_solver_registry,
)
from .patterns import (CSRPattern, SparsityKey, SparsityPatternCache,
                       batched_evaluate)

__all__ = ["BackendDescriptor", "BackendQualification",
           "BenchmarkMeasurement", "PerformanceLaboratory",
           "add_at_reduction", "bincount_reduction", "sorted_segment_reduction",
           "sparse_construction_benchmark", "LinearSolveEvidence",
           "SolverDescriptor", "SolverLaboratory", "reference_solver_registry"]
__all__ += ["CSRPattern", "SparsityKey", "SparsityPatternCache",
            "batched_evaluate"]
