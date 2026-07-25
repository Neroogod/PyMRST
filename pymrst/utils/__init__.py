"""Utility routines: units, AD, and array helpers (MRST utils/)."""
from pymrst.utils.eigs import multi_eig, multi_symm_eig
from pymrst.utils.flux_conversion import (
    cell_flux_to_face_flux,
    face_flux_to_cell_flux,
    face_flux_to_cell_velocity,
)

__all__ = [
    "multi_eig",
    "multi_symm_eig",
    "cell_flux_to_face_flux",
    "face_flux_to_cell_flux",
    "face_flux_to_cell_velocity",
]
