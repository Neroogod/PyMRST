"""MRST-shaped dual-porosity transfer model landing modules."""

from pymrst.modules.dual_porosity.transfer_models.kazemi_single_phase_transfer_function import (
    KazemiSinglePhaseTransfer,
)
from pymrst.modules.dual_porosity.transfer_models.kazemi_two_phase_transfer_function import (
    KazemiTwoPhaseTransfer,
)

__all__ = ["KazemiSinglePhaseTransfer", "KazemiTwoPhaseTransfer"]
