"""MRST `linearsolvers/examples/*.m`, ported (AMGCL demonstrations).

MRST's `linearsolvers` module is almost entirely MATLAB/MEX plumbing for its
AMGCL binding (build script, MATLAB<->C++ option marshaling, MEX sources) -
none of it portable, and none of it ported. The solver classes that plumbing
serves (`AMGCLSolverAD`, `AMGCL_CPRSolverAD`) live in MRST's `ad-core/solvers`
instead, and are ported at
`pymrst.modules.ad_core.solvers.amgcl_solver_ad.AMGCLSolverAD`.

These four are the module's `examples/` scripts - genuine field-scale
demonstrations (SPE10-based option sweeps, timing comparisons, a sequential
and a CPR pairing) with no equivalent elsewhere in the port, so they are
ported here, at the same path MRST uses.
"""

from pymrst.modules.linearsolvers.examples.sequential_amgcl import (
    run_sequential_amgcl,
)
from pymrst.modules.linearsolvers.examples.show_options_amgcl import (
    run_show_options_amgcl,
)
from pymrst.modules.linearsolvers.examples.test_amgcl import run_test_amgcl
from pymrst.modules.linearsolvers.examples.test_amgcl_cpr import (
    run_test_amgcl_cpr,
)

__all__ = [
    "run_sequential_amgcl",
    "run_show_options_amgcl",
    "run_test_amgcl",
    "run_test_amgcl_cpr",
]
