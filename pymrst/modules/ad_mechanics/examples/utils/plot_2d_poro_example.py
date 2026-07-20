"""Compatibility shim for MRST ``plot2DPoroExample.m``."""


def plot_2d_poro_example(*args, **kwargs):
    raise NotImplementedError(
        "plot2DPoroExample depends on MRST's exact 2D mechanics example. "
        "Use PyMRST's plot_toolbar on supported 3D slab/Norne states, or "
        "port the 2D VEM path first."
    )


__all__ = ["plot_2d_poro_example"]
