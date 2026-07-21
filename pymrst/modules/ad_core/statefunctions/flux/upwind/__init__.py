"""MRST ad-core upwind discretizations (`.../statefunctions/flux/upwind/`).

The hyperbolic (advective) discretization layer: given a per-face upwind flag,
carry the upstream cell value across each interior face.

- `UpwindDiscretization` - abstract base;
- `UpwindFunctionWrapperDiscretization` - wraps PyMRST's validated
  single-point upwind (`model.operators.upw`);
- `UpwindProperty` - mixin that gives a state function `face_upstream`;
- `get_boundary_faces_merged` - mirror-cell geometry for a boundary stencil.

`WENOUpwindDiscretization` (the higher-order reconstruction MRST also files
here) is not yet ported: it needs a MATLAB `wenoExampleAD` reference to
validate, which this environment cannot produce. The geometry it relies on
(`get_boundary_faces_merged`) is here, since its correctness is an invariant.
"""

from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_discretization import (  # noqa: E501
    UpwindDiscretization,
    UpwindFunctionWrapperDiscretization,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_property import (
    UpwindProperty,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.boundary_faces import (
    get_boundary_faces_merged,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import (
    WENOUpwindDiscretization,
)

__all__ = [
    "UpwindDiscretization",
    "UpwindFunctionWrapperDiscretization",
    "UpwindProperty",
    "WENOUpwindDiscretization",
    "get_boundary_faces_merged",
]
