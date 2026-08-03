"""Compatibility re-exports; the implementation lives in `modules/coarsegrid`.

MRST keeps `coarsegrid` as a top-level module, so PyMRST does too. These names
stay importable from here because existing callers use them.

**The import order below is deliberate.** This package contains submodules named
`generate_coarse_grid` and `partition_cart_grid`, the same names as the
functions they hold. Importing either submodule binds it as an attribute of this
package and shadows a function of the same name, so
`from ...upscaling.coarsegrid import generate_coarse_grid` yields the *module*
and calling it raises `TypeError: 'module' object is not callable`. The
submodules are therefore imported first and the function names rebound
afterwards, which makes the functions win regardless of what a caller imports
later.

The same trap is documented in `compositional/utils/phase_stability_test.py`,
where class identity through this shadowing made `eos.py` fail its own tests
while working on import.
"""

from pymrst.modules.upscaling.coarsegrid import (  # noqa: F401
    generate_coarse_grid as _generate_coarse_grid_module,
    partition_cart_grid as _partition_cart_grid_module,
)
from pymrst.modules.coarsegrid import (
    CoarseGrid,
    generate_coarse_grid,
    partition_cart_grid,
)

__all__ = ["CoarseGrid", "generate_coarse_grid", "partition_cart_grid"]
