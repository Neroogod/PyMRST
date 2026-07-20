"""MRST `ad-core/models/PhysicalModel.m` base model protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import State


class PhysicalModel:
    """The AD-OO protocol: properties, equations, updates, convergence."""

    def __init__(self):
        self.groupings: List[str] = []

    def get_props(self, state: State, *names: str) -> Any:
        """One or several properties, from whichever grouping declares them."""
        out = [self._get_one(state, n) for n in names]
        return out[0] if len(out) == 1 else out

    def _get_one(self, state: State, name: str) -> Any:
        for gname in self.groupings:
            group = getattr(self, gname)
            if group.has_state_function(name):
                return group.get(self, state, name)
        if name in state:
            return state[name]
        raise KeyError(
            f"'{name}' is neither a state variable nor a state function of "
            f"{type(self).__name__}"
        )

    def validate_model(self) -> None:
        """Check every declared dependency can actually be resolved."""
        for gname in self.groupings:
            group = getattr(self, gname)
            group.evaluation_order()
            for fname, deps in group.dependency_graph().items():
                for dep in deps:
                    if dep.grouping == "state":
                        continue
                    owner = (group if dep.grouping == "self"
                             else getattr(self, dep.grouping, None))
                    if owner is None:
                        raise KeyError(
                            f"{type(group).__name__}.{fname} depends on "
                            f"grouping '{dep.grouping}', which the model "
                            f"does not have"
                        )
                    if not owner.has_state_function(dep.name):
                        raise KeyError(
                            f"{type(group).__name__}.{fname} depends on "
                            f"'{dep.name}' in {dep.grouping}, which does not "
                            f"define it"
                        )

    def get_equations(self, state0: State, state: State, dt: float):
        raise NotImplementedError

    def update_state(self, state: State, increment: np.ndarray) -> State:
        raise NotImplementedError

    def get_convergence(self, residuals, dt: float) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError


__all__ = ["PhysicalModel"]
