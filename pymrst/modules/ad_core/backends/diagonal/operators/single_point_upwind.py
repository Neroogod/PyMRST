"""MRST ``singlePointUpwind`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.operators import single_point_upwind

singlePointUpwind = single_point_upwind

__all__ = ["single_point_upwind", "singlePointUpwind"]
