"""Compatibility landing file for MRST `SequentialPressureTransportModel.m`."""

from pymrst.modules.sequential.seq_model import (
    SequentialOilWaterModel,
    SequentialReport,
)

__all__ = ["SequentialOilWaterModel", "SequentialReport"]
