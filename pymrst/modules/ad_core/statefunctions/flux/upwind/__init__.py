"""MRST ad-core upwind discretizations (`.../statefunctions/flux/upwind/`).

The hyperbolic (advective) discretization layer: given a per-face upwind flag,
carry the upstream cell value across each interior face.

- `UpwindDiscretization` - abstract base;
- `UpwindFunctionWrapperDiscretization` - wraps PyMRST's validated
  single-point upwind (`model.operators.upw`);
- `UpwindProperty` - mixin that gives a state function `face_upstream`;
- `get_boundary_faces_merged_ad` - mirror-cell geometry for a boundary stencil;
- `WENOUpwindDiscretizationAD` - higher-order reconstruction for the legacy
  models.

The `*AD`/`*_ad` names are PyMRST's earlier versions, kept for their callers.
The ports of MRST's `WENOUpwindDiscretization` and `getBoundaryFacesMerged`,
for the AD-OO models, are in `ad_core/ad_oo/statefunctions/flux/`, and
`wenoExampleAD` in `ad_core/examples/weno_example_ad.py` runs on them.
"""

from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_discretization import (  # noqa: E501
    UpwindDiscretization,
    UpwindFunctionWrapperDiscretization,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.upwind_property import (
    UpwindProperty,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.boundary_faces import (
    get_boundary_faces_merged_ad,
)
from pymrst.modules.ad_core.statefunctions.flux.upwind.weno import (
    WENOUpwindDiscretizationAD,
)

__all__ = [
    "UpwindDiscretization",
    "UpwindFunctionWrapperDiscretization",
    "UpwindProperty",
    "WENOUpwindDiscretizationAD",
    "get_boundary_faces_merged_ad",
]
