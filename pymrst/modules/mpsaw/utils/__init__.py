"""MRST-shaped `mpsaw/utils` namespace."""

from pymrst.modules.mpsaw import mpsaw_cell_gradient, mpsaw_recover

cell_average_operator = mpsaw_cell_gradient

__all__ = ["cell_average_operator", "mpsaw_cell_gradient", "mpsaw_recover"]
