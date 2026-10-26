"""MRST ``ad-blackoil/utils/getCapillaryPressureBO.m``: `pcOW = p_o - p_w`
and `pcOG = p_g - p_o` from the fluid's `pcOW(sW)` and `pcOG(sG)`, each 0
when the fluid has no such function or its saturation is empty.

MATLAB errors when the fluid has `pcOG` and `sG` is not passed; here an
omitted `sG` gives `pcOG = 0`.

`get_capillary_pressure_bo_ad` is the earlier version, which read the
snake-case `pc_ow`/`pc_og` only and so returned 0 for a fluid with MRST's
field names.
"""

from typing import Any, Tuple

import numpy as np


def _isfield(fluid: Any, name: str) -> bool:
    return name in fluid if isinstance(fluid, dict) else hasattr(fluid, name)


def _get(fluid: Any, name: str) -> Any:
    return fluid[name] if isinstance(fluid, dict) else getattr(fluid, name)


def _isempty(x: Any) -> bool:
    return x is None or np.size(getattr(x, "val", x)) == 0


def get_capillary_pressure_bo(fluid: Any, sW: Any, sG: Any = None) -> Tuple[Any, Any]:
    """`(pcOW, pcOG)`, each 0 where MRST's would be."""
    pcOW = pcOG = 0
    if _isfield(fluid, "pcOW") and not _isempty(sW):
        pcOW = _get(fluid, "pcOW")(sW)
    if _isfield(fluid, "pcOG") and not _isempty(sG):
        pcOG = _get(fluid, "pcOG")(sG)
    return pcOW, pcOG


def get_capillary_pressure_bo_ad(fluid, s_w, s_g=None):
    """Return oil-water and oil-gas capillary pressures.

    `pcOW = p_o - p_w` and `pcOG = p_g - p_o`, matching MRST. Missing
    functions evaluate to zero.
    """
    pc_ow = fluid.pc_ow(s_w) if getattr(fluid, "pc_ow", None) else 0.0
    pc_og = (
        fluid.pc_og(s_g)
        if s_g is not None and getattr(fluid, "pc_og", None)
        else 0.0
    )
    return pc_ow, pc_og


getCapillaryPressureBO = get_capillary_pressure_bo
__all__ = ["get_capillary_pressure_bo", "getCapillaryPressureBO",
           "get_capillary_pressure_bo_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_mrst_field_names_are_read():
        fluid = SimpleNamespace(pcOW=lambda s: 2*s, pcOG=lambda s: 3*s)
        pcow, pcog = get_capillary_pressure_bo(fluid, np.array([0.5]), np.array([0.25]))
        np.testing.assert_array_equal(pcow, [1.0])
        np.testing.assert_array_equal(pcog, [0.75])
        print("ok test_mrst_field_names_are_read")

    def test_missing_functions_and_empty_saturations_give_zero():
        assert get_capillary_pressure_bo(SimpleNamespace(), np.array([0.5]),
                                         np.array([0.1])) == (0, 0)
        fluid = {"pcOW": lambda s: 2*s, "pcOG": lambda s: 3*s}
        assert get_capillary_pressure_bo(fluid, np.array([]), np.array([])) == (0, 0)
        assert get_capillary_pressure_bo(fluid, np.array([1.0]))[1] == 0
        print("ok test_missing_functions_and_empty_saturations_give_zero")

    def test_legacy_version_reads_snake_case_only():
        fluid = SimpleNamespace(pcOW=lambda s: 2*s)
        assert get_capillary_pressure_bo_ad(fluid, 0.5) == (0.0, 0.0)
        print("ok test_legacy_version_reads_snake_case_only")

    test_mrst_field_names_are_read()
    test_missing_functions_and_empty_saturations_give_zero()
    test_legacy_version_reads_snake_case_only()
    print("\nAll tests passed!")
