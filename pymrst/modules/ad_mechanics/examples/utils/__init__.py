"""MRST-shaped ad-mechanics example utility landing modules."""

from pymrst.modules.ad_mechanics.examples.utils.compute_uplift_for_state import (
    compute_uplift_for_state,
)
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift import (
    uplift_objective,
)
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift_aver import (
    obj_uplift_aver,
    uplift_average_objective,
)
from pymrst.modules.ad_mechanics.examples.utils.plot_2d_poro_example import (
    plot_2d_poro_example,
)
from pymrst.modules.ad_mechanics.examples.utils.plot_norne_poro_example import (
    plot_norne_poro_example,
)

__all__ = [
    "compute_uplift_for_state",
    "uplift_objective",
    "uplift_average_objective",
    "obj_uplift_aver",
    "plot_2d_poro_example",
    "plot_norne_poro_example",
]
