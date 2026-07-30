"""Compatibility module for MRST ``removeFaultBdryFaces``."""

from pymrst.gridprocessing.process_faults import (
    remove_fault_boundary_faces,
    removeFaultBdryFaces,
)

__all__ = ["remove_fault_boundary_faces", "removeFaultBdryFaces"]
