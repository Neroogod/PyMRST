"""MRST ``getSparseBlocks`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.utils import get_sparse_blocks

getSparseBlocks = get_sparse_blocks
__all__ = ["get_sparse_blocks", "getSparseBlocks"]
