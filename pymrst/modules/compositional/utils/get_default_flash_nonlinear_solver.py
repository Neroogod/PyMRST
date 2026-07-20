"""Default flash nonlinear solver (MRST ``getDefaultFlashNonLinearSolver.m``)."""

from __future__ import annotations

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver


_ALIAS_TO_NATIVE = {
    "maxIterations": "max_iterations",
    "maxTimestepCuts": "max_timestep_cuts",
    "useRelaxation": "use_relaxation",
    "useLineSearch": "use_line_search",
}


def _set_solver_option(solver: NonLinearSolver, name: str, value) -> None:
    native = _ALIAS_TO_NATIVE.get(name, name)
    setattr(solver, native, value)
    if native == "max_iterations":
        setattr(solver, "maxIterations", value)
    elif native == "max_timestep_cuts":
        setattr(solver, "maxTimestepCuts", value)
    elif native == "use_relaxation":
        setattr(solver, "useRelaxation", value)
    elif native == "use_line_search":
        setattr(solver, "useLineSearch", value)
    else:
        setattr(solver, name, value)


def _iter_options(args, kwargs):
    if len(args) == 1 and isinstance(args[0], dict):
        yield from args[0].items()
    elif args:
        if len(args) % 2:
            raise TypeError("positional options must be name/value pairs or a single dict")
        for name, value in zip(args[0::2], args[1::2]):
            yield name, value
    yield from kwargs.items()


def getDefaultFlashNonLinearSolver(*args, **kwargs) -> NonLinearSolver:
    """Return MRST's default nonlinear solver configuration for flash problems."""
    solver = NonLinearSolver(max_iterations=500, max_timestep_cuts=0, use_relaxation=True, verbose=-1)
    solver.maxIterations = 500
    solver.maxTimestepCuts = 0
    solver.useRelaxation = True
    solver.continueOnFailure = True
    solver.errorOnFailure = False
    for name, value in _iter_options(args, kwargs):
        _set_solver_option(solver, str(name), value)
    return solver


def get_default_flash_nonlinear_solver(*args, **kwargs) -> NonLinearSolver:
    """Pythonic alias for :func:`getDefaultFlashNonLinearSolver`."""
    return getDefaultFlashNonLinearSolver(*args, **kwargs)


__all__ = ["getDefaultFlashNonLinearSolver", "get_default_flash_nonlinear_solver"]
