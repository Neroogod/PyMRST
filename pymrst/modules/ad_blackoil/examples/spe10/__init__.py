"""SPE10 ad-blackoil example layout."""

from pymrst.modules.ad_blackoil.examples.spe10.setup_spe10_ad import (
    SPE10ADCase, run_spe10_ad, setup_spe10_ad_ad,
)
from pymrst.modules.ad_blackoil.examples.spe10.tenth_csp_model_i import (
    SPE10Model1Result, run_tenth_csp_model_i_ad, setup_tenth_csp_model_i_ad,
    tenth_csp_model_i, tenthCSP_Model_I,
)

__all__ = [
    "SPE10Model1Result", "run_tenth_csp_model_i_ad",
    "setup_tenth_csp_model_i_ad", "tenth_csp_model_i", "tenthCSP_Model_I",
    "SPE10ADCase", "run_spe10_ad", "setup_spe10_ad_ad",
]
