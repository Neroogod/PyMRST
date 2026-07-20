"""Permeability upscaling dispatcher (MRST ``upscalePerm.m`` layout)."""

from pymrst.modules.upscaling.flow.upscale_permeability_fixed import (
    upscale_perm_fixed,
)


def upscale_perm(g, cg, rock, method: str = "tpfa", **kwargs):
    """Upscale permeability using the validated fixed TPFA method."""
    if method.lower() != "tpfa":
        raise NotImplementedError(
            "PyMRST currently ports MRST upscalePerm only for method='tpfa'"
        )
    return upscale_perm_fixed(g, rock, cg, **kwargs)


__all__ = ["upscale_perm"]
