"""MRST `ad-blackoil/models` layout."""

from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    blackoil_status,
    initial_state_from_deck,
)
from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    BOState,
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)
from pymrst.modules.ad_blackoil.models.tracer_black_oil_model import (
    TracerBlackOilModel,
)
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_blackoil.models.water_model import WaterModel

__all__ = [
    "BOState",
    "GenericBlackOilModel",
    "OWState",
    "ThreePhaseBlackOilModelAD",
    "TracerBlackOilModel",
    "TwoPhaseOilWaterModelAD",
    "WaterModel",
    "WaterThermalModel",
    "blackoil_status",
    "initial_state_from_deck",
    "initial_state_spe1",
]


def __getattr__(name):
    if name == "WaterThermalModel":
        from pymrst.modules.ad_blackoil.models.water_thermal_model import (
            WaterThermalModel,
        )

        return WaterThermalModel
    raise AttributeError(name)
