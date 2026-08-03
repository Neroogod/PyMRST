"""Interactive visualization for the MRST two-dimensional poromechanics cases."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _option(options: Any, name: str, default: str) -> str:
    if options is None:
        return default
    if isinstance(options, dict):
        return str(options.get(name, default))
    return str(getattr(options, name, default))


def plot_2d_poro_example(
    model: Any,
    states: Sequence[Any],
    options: Any = None,
    *,
    show: bool = True,
):
    """Open the PyMRST toolbar with flow and mechanics cell fields."""
    if not states:
        raise ValueError("plot_2d_poro_example requires at least one state")
    nc = int(model.G.cells.num)
    fields: dict[str, np.ndarray] = {}
    for name in ("pressure", "sW", "sG", "rs"):
        if all(name in state for state in states):
            values = [np.asarray(state[name], dtype=float).ravel() for state in states]
            if all(value.size == nc for value in values):
                fields[name] = np.stack(values)

    mechanics = getattr(model, "mech", None) or getattr(model, "mechModel", None)
    if mechanics is not None and all("u" in state for state in states):
        fields["volumetric strain"] = np.stack([
            np.asarray(mechanics.volumetric_strain_full(state["u"]), dtype=float)
            for state in states
        ])
        fields["displacement magnitude"] = np.stack([
            np.linalg.norm(np.asarray(state["u"], dtype=float).reshape(-1, model.G.griddim), axis=1)
            for state in states
        ])

    if not fields:
        raise ValueError("states contain no plottable flow or mechanics fields")
    fluid = _option(options, "fluid_model", "unknown")
    method = _option(options, "method", "unknown")
    from pymrst.plotting.extensions.plot_toolbar import plot_toolbar

    return plot_toolbar(
        model.G,
        fields,
        show=show,
        title=f"fluid model: {fluid}, method: {method}",
    )


plot2DPoroExample = plot_2d_poro_example

__all__ = ["plot_2d_poro_example", "plot2DPoroExample"]
