"""MRST ad-core well models (`ad-core/models/wells/`).

The single-well control layer, factored onto MRST's layout from PyMRST's
procedural facility model:

- `setup_well_control_equation_single_well` - the equation that closes a well
  on its target (bhp / rate / phase-rate / liquid-rate);
- `update_switched_controls` / `violated_limit` - switch a well to the limit it
  breaks (a rate injector hitting its bhp cap, a bhp producer hitting its rate
  cap).

Both reproduce the MATLAB-validated procedural versions in `FacilityModel`.
"""

from pymrst.modules.ad_core.models.wells.well_control_equations import (
    setup_well_control_equation_single_well,
)
from pymrst.modules.ad_core.models.wells.update_switched_controls import (
    update_switched_controls,
    violated_limit,
)

__all__ = [
    "setup_well_control_equation_single_well",
    "update_switched_controls",
    "violated_limit",
]
