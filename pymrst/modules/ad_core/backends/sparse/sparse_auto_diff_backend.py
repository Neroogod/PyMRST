"""Sparse AD backend descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend


@dataclass(frozen=True)
class SparseAutoDiffBackend(AutoDiffBackend):
    """PyMRST's native sparse-Jacobian ADI backend."""

    name: str = "Sparse"
    available: bool = True
    #: MRST's `useBlocks`: one Jacobian block per primary variable, or a single
    #: matrix. Blocks are the default and are what PyMRST has always produced.
    use_blocks: bool = True

    def init_variables_ad(self, *values, **kwargs):
        if self.use_blocks:
            from pymrst.utils.adi import init_variables_adi
            return init_variables_adi(*values)
        from pymrst.modules.ad_core.backends.diagonal.utils import (
            init_variables_one_block,
        )
        return init_variables_one_block(*values)

    initVariablesAD = init_variables_ad

    def convert_to_ad(self, v, sample):
        if self.use_blocks:
            from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
            return double_to_adi(v, sample)
        from pymrst.modules.ad_core.backends.diagonal.utils import (
            double_to_generic_ad,
        )
        return double_to_generic_ad(v, sample)

    convertToAD = convert_to_ad

    def get_backend_description(self) -> str:
        return "Sparse" if self.use_blocks else "Sparse (single matrix)"

    getBackendDescription = get_backend_description


__all__ = ["SparseAutoDiffBackend"]
