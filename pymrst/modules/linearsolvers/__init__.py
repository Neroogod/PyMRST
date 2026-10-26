"""MRST `linearsolvers`, ported.

MRST's `modules/linearsolvers/` is its AMGCL MEX binding: a build script,
MATLAB<->AMGCL-C++ option marshaling, MEX sources, and `amgcl/tests/`,
`amgcl/utils/` subdirectories holding more of the same. None of it is
portable numerical logic - it exists to compile and call a MEX binary, a
problem Python does not have.

The solver classes that plumbing serves do not even live here in MRST
itself - `AMGCLSolverAD`/`AMGCL_CPRSolverAD` are in `ad-core/solvers`, and
are ported at `pymrst.modules.ad_core.solvers.amgcl_solver_ad`, using
`pyamgcl` (a pybind11 binding to AMGCL's own C++ headers) in place of the
MEX glue. `pyproject.toml`'s `amgcl` optional dependency is this port's
equivalent of `buildLinearSolvers.m`'s fetch/build step.

What *is* ported here, matching MRST's own path, is `examples/` - real
field-scale AMGCL demonstrations with no equivalent elsewhere in the port.
See `pymrst.modules.linearsolvers.examples`.
"""
