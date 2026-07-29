"""MRST ``double2GenericAD`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.utils import double_to_generic_ad

double2GenericAD = double_to_generic_ad
__all__ = ["double_to_generic_ad", "double2GenericAD"]
