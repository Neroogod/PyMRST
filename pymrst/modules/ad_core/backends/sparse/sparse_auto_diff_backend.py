"""Sparse AD backend descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend


@dataclass(frozen=True)
class SparseAutoDiffBackend(AutoDiffBackend):
    """PyMRST's native sparse-Jacobian ADI backend."""

    name: str = "Sparse"
    available: bool = True


__all__ = ["SparseAutoDiffBackend"]
