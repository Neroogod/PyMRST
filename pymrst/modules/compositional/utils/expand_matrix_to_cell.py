"""Moved to `pymrst.modules.ad_core.utils.expand_matrix_to_cell`.

MRST keeps `expandMatrixToCell.m` in `modules/ad-core/utils/`, not in
`compositional`. PyMRST had it the other way round - the implementation here,
with `ad_core` re-exporting it - which made a lower-level module import a
higher-level one.

That was not merely untidy. It closed an import cycle:

    compositional/model -> ad_core/__init__ -> ad_core/utils/__init__
      -> ad_core/utils/expand_matrix_to_cell -> compositional/utils/__init__
      -> compositional/utils/equations_compositional
      -> compositional/model   (still half-initialised)

so `import pymrst.modules.compositional.model` raised ImportError and every
validator that touches compositional failed. The implementation now lives at
its MRST path and this re-exports it, which points the dependency downward.
"""

from pymrst.modules.ad_core.utils.expand_matrix_to_cell import (  # noqa: F401
    expand_matrix_to_cell,
    expandMatrixToCell,
)

__all__ = ["expand_matrix_to_cell", "expandMatrixToCell"]
