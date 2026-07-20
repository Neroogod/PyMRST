"""MRST ``VEM_assembleTEDFM.m`` TEDFM VEM stiffness assembly."""

import numpy as np
import scipy.sparse as sp

from pymrst.modules.vemmech.vem_assemble import vem_assemble


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def vem_assemble_tedfm(
    g,
    C: np.ndarray,
    alpha_scaling: float = 1.0,
) -> sp.csr_array:
    """Assemble TEDFM mechanics stiffness on ``G.Matrix``."""
    return vem_assemble(_matrix_grid(g), C, alpha_scaling=alpha_scaling)

__all__ = ["vem_assemble_tedfm"]
