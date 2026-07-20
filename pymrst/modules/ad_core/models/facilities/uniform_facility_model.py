"""MRST ``ad-core/models/facilities/UniformFacilityModel.m`` surface."""

from __future__ import annotations

from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel


class UniformFacilityModel(FacilityModel):
    """Simplified facility model using PyMRST's direct uniform well equations."""


__all__ = ["UniformFacilityModel"]
