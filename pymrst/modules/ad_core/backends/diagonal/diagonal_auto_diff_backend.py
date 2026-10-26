"""Diagonal AD backend descriptor."""

from __future__ import annotations

from dataclasses import dataclass

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
from pymrst.modules.ad_core.backends.diagonal import kernels


@dataclass(frozen=True)
class DiagonalAutoDiffBackend(AutoDiffBackend):
    """MRST-like diagonal backend configuration.

    PyMRST implements the pure-Python diagonal Jacobian path, including a
    row-major storage layout. When Numba is installed, the MEX-labelled rows in
    MRST's examples are mapped to compiled Numba row-major kernels.
    """

    name: str = "Diagonal"
    use_mex: bool = True
    row_major: bool = False
    deferred_assembly: bool = False
    available: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if (self.name == "Diagonal" and not self.row_major) or not self.use_mex:
            object.__setattr__(self, "available", True)
            object.__setattr__(self, "reason", "")
            return
        if self.use_mex and kernels.HAS_NUMBA_KERNELS:
            object.__setattr__(self, "available", True)
            object.__setattr__(self, "reason", "")
            return
        if self.use_mex or self.row_major:
            object.__setattr__(self, "available", False)
            object.__setattr__(
                self,
                "reason",
                "Compiled diagonal kernels require Numba; install numba to activate the MEX-labelled backend rows.",
            )

    def init_variables_ad(self, *values, **kwargs):
        """Seed AD variables in the diagonal representation.

        The row-major layout is a different storage order, not a different
        answer - which is why `test_both_layouts_give_the_same_derivatives`
        compares them rather than trusting the label.
        """
        from pymrst.modules.ad_core.backends.diagonal.utils import (
            init_variables_ad_diagonal, init_variables_ad_diagonal_row_major,
        )
        use_mex = kwargs.pop("use_mex", self.use_mex and self.available)
        builder = (init_variables_ad_diagonal_row_major if self.row_major
                   else init_variables_ad_diagonal)
        return builder(*values, use_mex=use_mex, **kwargs)

    initVariablesAD = init_variables_ad

    def convert_to_ad(self, v, sample):
        from pymrst.modules.ad_core.backends.diagonal.utils import (
            double_to_generic_ad,
        )
        return double_to_generic_ad(v, sample)

    convertToAD = convert_to_ad

    def get_backend_description(self) -> str:
        out = "Diagonal-MEX" if self.use_mex else "Diagonal"
        return out + "-RowMajor" if self.row_major else out

    getBackendDescription = get_backend_description

    def update_discrete_operators(self, model):
        """The diagonal backend can rebuild the discrete operators in its own
        representation. PyMRST's operators are backend-agnostic, so this is a
        no-op that exists so a caller written against MRST does not have to
        special-case it."""
        return model

    updateDiscreteOperators = update_discrete_operators

    def label(self) -> str:
        if self.name != "Diagonal":
            return self.name
        if self.use_mex and self.row_major:
            return "Diagonal (MEX+RowMajor)"
        if self.use_mex:
            return "Diagonal"
        return "Diagonal"

    def kernel_report(self) -> dict:
        """Return the compiled-kernel status for this diagonal backend.

        MRST reports separate diagonal/MEX backend rows. PyMRST maps those
        MEX-labelled rows to optional Numba kernels, so this method exposes the
        exact kernel availability instead of forcing callers to infer it from
        labels. The report is intentionally value-level metadata: solver code
        still uses the compact Jacobian classes, which dispatch to these
        kernels when available and fall back to pure Python otherwise.
        """
        names = (
            "add_row_major",
            "left_scale_row_major",
            "matmul_left_row_major",
            "matmul_left_triplet",
            "row_repeat_row_major",
            "row_repeat_triplet",
            "take_rows_row_major",
            "take_rows_triplet",
            "vstack_row_major",
            "vstack_triplet",
        )
        active = {name: getattr(kernels, name, None) is not None for name in names}
        return {
            "backend": self.label(),
            "use_mex": bool(self.use_mex),
            "row_major": bool(self.row_major),
            "deferred_assembly": bool(self.deferred_assembly),
            "available": bool(self.available),
            "numba_available": bool(kernels.HAS_NUMBA_KERNELS),
            "kernel_family": "numba" if kernels.HAS_NUMBA_KERNELS else "python-fallback",
            "active_kernels": active,
            "all_requested_kernels_active": bool(
                (not self.use_mex) or (kernels.HAS_NUMBA_KERNELS and all(active.values()))
            ),
            "reason": self.reason,
        }

    # MATLAB-oriented alias for notebooks and direct ports.
    kernelReport = kernel_report


__all__ = ["DiagonalAutoDiffBackend"]
