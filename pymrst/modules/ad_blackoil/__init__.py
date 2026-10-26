"""AD black-oil reservoir models (MRST modules/ad-blackoil).

`GenericBlackOilModel`, `ThreePhaseBlackOilModel`, `TwoPhaseOilWaterModel`
and `WaterModel` are the line-for-line ports of MRST's classes, built on the
AD-OO base in `ad_core.ad_oo`. The `*AD` classes are the earlier PyMRST
models (`GenericBlackOilModelAD` assembles from the state-function graph);
they stay because Phases 5a and 5b validate them against MATLAB and the
existing examples run on them.
"""

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    BOState,
    ThreePhaseBlackOilModel,
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    GenericBlackOilModelAD,
    blackoil_status,
    initial_state_from_deck,
)
from pymrst.modules.ad_blackoil.models.tracer_black_oil_model import (
    TracerBlackOilModel,
)
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModel,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_blackoil.models.water_model import WaterModel, WaterModelAD
from pymrst.modules.ad_blackoil.backend_oil_water import (
    BackendOWFluid,
    BackendOWState,
    BackendOilWaterModel,
)
from pymrst.modules.ad_blackoil.backend_blackoil_fluid import (
    BackendBlackOilFluid,
    init_backend_blackoil_fluid,
)
from pymrst.modules.ad_blackoil.backend_blackoil_model import (
    BackendBOState,
    BackendBlackOilModel,
    BackendBlackOilResidual,
)

__all__ = [
    "GenericBlackOilModel", "GenericBlackOilModelAD",
    "initial_state_from_deck", "blackoil_status",
    "TracerBlackOilModel",
    "OWState", "TwoPhaseOilWaterModel", "TwoPhaseOilWaterModelAD",
    "BackendOWFluid", "BackendOWState", "BackendOilWaterModel",
    "BackendBlackOilFluid", "init_backend_blackoil_fluid",
    "BackendBOState", "BackendBlackOilModel", "BackendBlackOilResidual",
    "BOState", "ThreePhaseBlackOilModel", "ThreePhaseBlackOilModelAD",
    "initial_state_spe1",
    "WaterModel", "WaterModelAD", "WaterThermalModel",
]


def __getattr__(name):
    if name == "WaterThermalModel":
        from pymrst.modules.ad_blackoil.models.water_thermal_model import (
            WaterThermalModel,
        )

        return WaterThermalModel
    raise AttributeError(name)
