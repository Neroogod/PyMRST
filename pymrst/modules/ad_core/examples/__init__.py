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
    TutorialAddAD,
    TutorialNumberAD,
    TutorialProductAD,
    build_tutorial_state_function_group,
    run_conceptual_ad,
)
from pymrst.modules.ad_core.examples.book_ii.example_state_functions_book import (
    exampleStateFunctionsBook,
    example_state_functions_book,
)
from pymrst.modules.ad_core.examples.book_ii.show_conceptual_ad import (
    showConceptualAD,
    show_conceptual_ad,
)
from pymrst.modules.ad_core.examples.book_ii.utils import (
    TutorialAdd,
    TutorialNumber,
    TutorialProduct,
)
from pymrst.modules.ad_core.examples.packed_problem_examples import (
    PackedDemoResult,
    demo_packed_problems_ad,
    ensemble_packed_problems_example_ad,
)
from pymrst.modules.ad_core.examples.demo_packed_problems import (
    demo_packed_problems,
    demoPackedProblems,
)
from pymrst.modules.ad_core.examples.ensemble_packed_problems_example import (
    ensemble_packed_problems_example,
    ensemblePackedProblemsExample,
)
from pymrst.modules.ad_core.examples.different_region_functions_example import (
    RegionalRelPermResult,
    different_region_functions_example,
    differentRegionFunctionsExample,
)
from pymrst.modules.ad_core.examples.state_function_tutorial import (
    StateFunctionTutorialResult,
    state_function_tutorial,
    stateFunctionTutorial,
)
from pymrst.modules.ad_core.examples.mpfa_vs_tpfa_with_ad_example import (
    MPFATPFAComparison,
    mpfa_vs_tpfa_with_ad_example,
    MPFAvsTPFAwithADExample,
)
from pymrst.modules.ad_core.examples.composite_model_tutorial import (
    CompositeTutorialResult,
    composite_model_tutorial,
    compositeModelTutorial,
)
from pymrst.modules.ad_core.examples.linear_solver_example import (
    LinearSolverExampleResult,
    linear_solver_example,
    linearSolverExample,
)
from pymrst.modules.ad_core.examples.mrst_demo import (
    MRSTDemoResult,
    mrst_demo,
)
from pymrst.modules.ad_core.examples.weno_example_ad import (
    weno_example_ad,
    wenoExampleAD,
)
from pymrst.modules.ad_core.examples.weno_example_ad_ad import (
    WENOExampleResult,
    weno_example_ad_ad,
)
from pymrst.modules.ad_core.examples.paired_heterogeneous_waterflood import (
    WaterfloodResult,
    build_paired_waterflood,
    run_paired_waterflood,
    save_result,
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
    "show_conceptual_ad",
    "example_state_functions_book",
    "TutorialAddAD",
    "TutorialNumberAD",
    "TutorialProductAD",
    "exampleStateFunctionsBook",
    "PackedDemoResult",
    "demo_packed_problems",
    "demo_packed_problems_ad",
    "demoPackedProblems",
    "ensemble_packed_problems_example",
    "ensemble_packed_problems_example_ad",
    "ensemblePackedProblemsExample",
    "RegionalRelPermResult",
    "different_region_functions_example",
    "differentRegionFunctionsExample",
    "StateFunctionTutorialResult",
    "state_function_tutorial",
    "stateFunctionTutorial",
    "MPFATPFAComparison",
    "mpfa_vs_tpfa_with_ad_example",
    "MPFAvsTPFAwithADExample",
    "CompositeTutorialResult",
    "composite_model_tutorial",
    "compositeModelTutorial",
    "LinearSolverExampleResult",
    "linear_solver_example",
    "linearSolverExample",
    "MRSTDemoResult",
    "mrst_demo",
    "WENOExampleResult",
    "weno_example_ad",
    "weno_example_ad_ad",
    "wenoExampleAD",
    "WaterfloodResult",
    "build_paired_waterflood",
    "run_paired_waterflood",
    "save_result",
]
