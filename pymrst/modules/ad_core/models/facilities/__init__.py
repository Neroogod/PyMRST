"""MRST-style facility model entry points for `ad_core`."""

from pymrst.modules.ad_core.models.facilities.facility_model import (
    FacilityLinearizedProblem,
    FacilityModel,
)
from pymrst.modules.ad_core.models.facilities.generic_facility_model import (
    GenericFacilityModel,
)
from pymrst.modules.ad_core.models.facilities.select_facility_from_deck import (
    group_controls_from_deck,
    selectFacilityFromDeck,
    select_facility_from_deck,
    select_facility_from_deck_ad,
    vfp_tables_from_deck,
)
from pymrst.modules.ad_core.models.facilities.simple_well import SimpleWell
from pymrst.modules.ad_core.models.facilities.reorder_well_perforations_by_depth import (  # noqa: E501
    reorder_well_perforations_by_depth,
)
from pymrst.modules.ad_core.models.facilities.multisegment import (
    MultisegmentEquationResult,
    assemble_multisegment_equations,
)
from pymrst.modules.ad_core.models.facilities.separators import (
    BasicSeparator,
    EOSSeparator,
    SeparatorFlashResult,
    SeparatorGroup,
    SeparatorStreamResult,
    SeparatorMoleResult,
    SeparatorSurfaceResult,
)
from pymrst.modules.ad_core.models.facilities.vfp import VFPTable
from pymrst.modules.ad_core.models.facilities.uniform_facility_model import (
    UniformFacilityModel,
)
from pymrst.modules.ad_core.models.facilities.well_bore_friction import (
    well_bore_friction,
    well_bore_friction_ad,
)
from pymrst.modules.ad_core.models.facilities.nozzle_valve import (
    nozzleValve,
    nozzle_valve,
)
from pymrst.modules.ad_core.models.facilities.perforation_properties import (
    packPerforationProperties,
    pack_perforation_properties,
    unpackPerforationProperties,
    unpack_perforation_properties,
)
from pymrst.modules.ad_core.models.facilities.combine_ms_with_regular_wells import (
    combine_ms_with_regular_wells,
    combine_ms_with_regular_wells_ad,
    combineMSwithRegularWells,
)

__all__ = [
    "FacilityLinearizedProblem",
    "FacilityModel",
    "GenericFacilityModel",
    "SimpleWell",
    "reorder_well_perforations_by_depth",
    "UniformFacilityModel",
    "BasicSeparator",
    "EOSSeparator",
    "MultisegmentEquationResult",
    "SeparatorFlashResult",
    "SeparatorGroup",
    "SeparatorMoleResult",
    "SeparatorStreamResult",
    "SeparatorSurfaceResult",
    "VFPTable",
    "assemble_multisegment_equations",
    "group_controls_from_deck",
    "selectFacilityFromDeck",
    "select_facility_from_deck",
    "select_facility_from_deck_ad",
    "vfp_tables_from_deck",
    "well_bore_friction",
    "well_bore_friction_ad",
    "nozzleValve",
    "nozzle_valve",
    "packPerforationProperties",
    "pack_perforation_properties",
    "unpackPerforationProperties",
    "unpack_perforation_properties",
    "combine_ms_with_regular_wells",
    "combine_ms_with_regular_wells_ad",
    "combineMSwithRegularWells",
]
