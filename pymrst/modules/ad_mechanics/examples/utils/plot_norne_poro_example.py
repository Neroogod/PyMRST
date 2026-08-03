"""Compatibility helper for MRST ``plotNornePoroExample.m``."""

import numpy as np


def plot_norne_poro_example(model, states, *, show: bool = True):
    """Open PyMRST's plotting toolbar for Norne-like state histories."""
    if not states:
        raise ValueError("plot_norne_poro_example requires at least one state")
    from pymrst.plotting.extensions.plot_toolbar import plot_toolbar
    from pymrst.utils.units import BARSA

    fields = {
        "pressure [bar]": [s["pressure"] / BARSA for s in states],
    }
    if all("sW" in state for state in states):
        fields["sW"] = [state["sW"] for state in states]
    if all("sG" in state for state in states):
        fields["sG"] = [state["sG"] for state in states]
    mechanics = getattr(model, "mech", None) or getattr(model, "mechModel", None)
    if mechanics is not None and all("u" in state for state in states):
        fields["div(u)"] = [
            mechanics.volumetric_strain_full(state["u"]) for state in states
        ]
        fields["|u|"] = [
            np.linalg.norm(
                np.asarray(state["u"]).reshape(-1, model.G.griddim),
                axis=1,
            )
            for state in states
        ]
    return plot_toolbar(model.G, fields, title="Norne poromechanics", show=show)


plotNornePoroExample = plot_norne_poro_example


__all__ = ["plot_norne_poro_example", "plotNornePoroExample"]
