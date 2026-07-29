"""Logical Jacobian size across sparse and compact AD backends."""

from __future__ import annotations

from typing import Any, Optional


def size_jac(jac: Any, dim: Optional[int] = None):
    """Port MRST ``sizeJac`` with zero-based Python dimensions."""
    shape = getattr(jac, "shape", None)
    if shape is None:
        diagonal = getattr(jac, "diagonal", None)
        dims = getattr(jac, "dim", None)
        if diagonal is None or dims is None:
            raise TypeError("jacobian must expose shape or diagonal/dim")
        shape = (diagonal.shape[0], int(__import__("numpy").prod(dims)))
    shape = tuple(int(x) for x in shape)
    return shape if dim is None else shape[int(dim)]


sizeJac = size_jac

__all__ = ["size_jac", "sizeJac"]
