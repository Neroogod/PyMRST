"""Compatibility helper for MRST ``plotNornePoroExample.m``."""


def plot_norne_poro_example(model, states):
    """Open PyMRST's plotting toolbar for Norne-like state histories."""
    from pymrst.plotting import plot_toolbar
    from pymrst.utils.units import BARSA

    fields = {
        "pressure [bar]": [s["pressure"] / BARSA for s in states],
        "sW": [s["sW"] for s in states if "sW" in s],
        "div(u)": [model.mech.volumetric_strain_full(s["u"]) for s in states],
    }
    return plot_toolbar(model.G, fields, title="Norne poromechanics")


__all__ = ["plot_norne_poro_example"]
