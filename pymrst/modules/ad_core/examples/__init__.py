"""Runnable ad-core examples (MRST `ad-core/examples`)."""

from pymrst.modules.ad_core.examples.ad_buckley_leverett_1d import (
    run_ad_buckley_leverett_1d,
)
from pymrst.modules.ad_core.examples.benchmark_auto_diff_backends import (
    BackendBenchmarkResult,
    benchmark_auto_diff_backends,
    benchmarkAutoDiffBackends,
)
from pymrst.modules.ad_core.examples.assembly_benchmark_ad import (
    AssemblyBenchmarkCase,
    AssemblyBenchmarkResult,
    assembly_benchmark_ad,
    assemblyBenchmarkAD,
)
from pymrst.modules.ad_core.examples.tutorial_state_functions import (
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
)

__all__ = [
    "run_ad_buckley_leverett_1d",
    "AssemblyBenchmarkCase",
    "AssemblyBenchmarkResult",
    "assembly_benchmark_ad",
    "assemblyBenchmarkAD",
    "BackendBenchmarkResult",
    "benchmark_auto_diff_backends",
    "benchmarkAutoDiffBackends",
    "TutorialAdd",
    "TutorialNumber",
    "TutorialProduct",
]
