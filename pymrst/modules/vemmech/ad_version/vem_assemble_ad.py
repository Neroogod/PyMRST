"""MRST ``VEM_assemble_AD.m`` AD-version VEM assembly."""

from types import SimpleNamespace

from pymrst.modules.vemmech.ad_version import (
    _ad_geometry_operators,
    _material_d_matrix,
    _value_vec,
)
from pymrst.modules.vemmech.utils.enu2c import enu2c, enu2c_2d
from pymrst.modules.vemmech.vem_assemble import vem_assemble


def vem_assemble_ad(g, E, nu, extra=None, alpha_scaling: float = 1.0):
    """Numeric matrix subset of MRST ``VEM_assemble_AD``."""
    E_val = _value_vec(E)
    nu_val = _value_vec(nu)
    C = enu2c_2d(E_val, nu_val) if g.griddim == 2 else enu2c(E_val, nu_val)
    S = vem_assemble(g, C, alpha_scaling=alpha_scaling)
    op = _ad_geometry_operators(g) if extra is None else SimpleNamespace(
        I=extra.I,
        volmap=extra.volmap,
        WC=extra.WC,
        NC=extra.NC,
        PP=extra.PP,
        assemb=extra.assemb,
        nlin=getattr(extra, "nlin", 3 if g.griddim == 2 else 6),
    )
    op.D = _material_d_matrix(g, E_val, nu_val)
    op.S = S
    op.Ax_derivs = 0
    return S, op

__all__ = ["vem_assemble_ad"]
