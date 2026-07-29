"""MRST ``twoPointGradient`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.operators import two_point_gradient

twoPointGradient = two_point_gradient

__all__ = ["two_point_gradient", "twoPointGradient"]
