"""AD black-oil reservoir models (MRST modules/ad-blackoil).

`GenericBlackOilModel` is the state-function-graph model (MRST's
`GenericBlackOilModel`) and is the one to build on; the two hand-rolled
models it reproduces are kept because Phases 5a and 5b validate them
directly against MATLAB, and because they are the reference the graph is
checked against.
"""

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    BOState,
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    blackoil_status,
    initial_state_from_deck,
)
from pymrst.modules.ad_blackoil.models.tracer_black_oil_model import (
    TracerBlackOilModel,
)
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_blackoil.models.water_model import WaterModel
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
    "GenericBlackOilModel", "initial_state_from_deck", "blackoil_status",
    "TracerBlackOilModel",
    "OWState", "TwoPhaseOilWaterModelAD",
    "BackendOWFluid", "BackendOWState", "BackendOilWaterModel",
    "BackendBlackOilFluid", "init_backend_blackoil_fluid",
    "BackendBOState", "BackendBlackOilModel", "BackendBlackOilResidual",
    "BOState", "ThreePhaseBlackOilModelAD", "initial_state_spe1",
    "WaterModel", "WaterThermalModel",
]


def __getattr__(name):
    if name == "WaterThermalModel":
        from pymrst.modules.ad_blackoil.models.water_thermal_model import (
            WaterThermalModel,
        )

        return WaterThermalModel
    raise AttributeError(name)
