"""MRST ``SparseMultiArray.m`` public tensor classes.

The canonical class definitions live in the ``ad_version`` package namespace so
all tensor operations share one class identity.
"""

from pymrst.modules.vemmech.ad_version import LazySparseTensor, SparseMultiArray

__all__ = ["LazySparseTensor", "SparseMultiArray"]
