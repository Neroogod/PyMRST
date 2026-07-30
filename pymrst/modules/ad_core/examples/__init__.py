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
    ConceptualADResult,
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
    build_tutorial_state_function_group,
    exampleStateFunctionsBook,
    run_conceptual_ad,
    showConceptualAD,
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
    "ConceptualADResult",
    "build_tutorial_state_function_group",
    "run_conceptual_ad",
    "showConceptualAD",
    "exampleStateFunctionsBook",
]
