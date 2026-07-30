"""Create a packed simulation problem directly from Eclipse input."""

from __future__ import annotations

from pathlib import Path
import secrets
import string

from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import (
    pack_simulation_problem,
)


def init_eclipse_packed_problem_ad(
    deck,
    *,
    base_name: str | None = None,
    name: str | None = None,
    directory: str | Path | None = None,
    **initialization_options,
):
    """Initialize Eclipse state/model/schedule and pack them for execution."""
    state0, model, schedule, nonlinear = init_eclipse_problem_ad(
        deck, **initialization_options
    )
    if base_name is None:
        input_data = getattr(model, "inputdata", None)
        runspec = (input_data.get("RUNSPEC", {}) if isinstance(input_data, dict)
                   else getattr(input_data, "RUNSPEC", None))
        title = (runspec.get("TITLE") if isinstance(runspec, dict)
                 else getattr(runspec, "TITLE", None))
        base_name = str(title) if title else "".join(
            secrets.choice(string.ascii_lowercase) for _ in range(20)
        )
    return pack_simulation_problem(
        state0, model, schedule, base_name,
        directory=directory, name=name, nonlinear_solver=nonlinear,
    )


initEclipsePackedProblemAD = init_eclipse_packed_problem_ad

__all__ = [
    "init_eclipse_packed_problem_ad",
    "initEclipsePackedProblemAD",
]
