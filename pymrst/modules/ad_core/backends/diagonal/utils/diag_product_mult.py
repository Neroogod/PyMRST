"""MRST ``diagProductMult`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.utils import diag_product_mult

diagProductMult = diag_product_mult
__all__ = ["diag_product_mult", "diagProductMult"]
