"""Block inversion helper corresponding to MRST ``blockInverter.m``."""

from pymrst.modules.mpfa.compute_multipoint_trans import _invert_blocks


def block_inverter(matrix, block_sizes):
    """Invert consecutive diagonal blocks of a sparse matrix."""
    return _invert_blocks(matrix, block_sizes)


blockInverter = block_inverter

__all__ = ["block_inverter", "blockInverter"]
