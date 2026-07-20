"""MRST ``VEM_linElastTEDFM.m`` TEDFM linear-elastic solve."""

from typing import Callable, Optional

import numpy as np

from pymrst.modules.vemmech.utils.vem_div_tedfm import vem_div_tedfm
from pymrst.modules.vemmech.vem import ElasticBC
from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def vem_lin_elast_tedfm(
    g,
    C: np.ndarray,
    el_bc: ElasticBC,
    load: Callable[[np.ndarray], np.ndarray],
    pressure: Optional[np.ndarray] = None,
    alpha_scaling: float = 1.0,
):
    """Solve TEDFM linear elasticity on ``G.Matrix`` and return MRST ``disc`` data."""
    mg = _matrix_grid(g)
    uu, extra = vem_lin_elast(
        mg,
        C,
        el_bc,
        load=load,
        pressure=pressure,
        alpha_scaling=alpha_scaling,
    )
    vdiv = vem_div_tedfm(g)
    isdir = extra["isdirdofs"]
    v_dir = extra["V_dir"]
    extra["disc"] = {
        "A": extra["A"],
        "isdirdofs": isdir,
        "rhs": extra["rhs"],
        "V_dir": v_dir,
        "ovol_div": vdiv,
        "gradP": vdiv[:, ~isdir].T,
        "div": vdiv[:, ~isdir],
        "divrhs": vdiv @ v_dir,
    }
    return uu, extra

__all__ = ["vem_lin_elast_tedfm"]
