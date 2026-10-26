"""MRST ``ad-blackoil/utils/getbG_BO.m``: the reciprocal gas formation
volume factor, `fluid.bG(p, rv, isLiquid)` for a model with vaporised oil
and `fluid.bG(p)` otherwise.

`get_b_g_bo_ad` is the earlier version, kept for `ThreePhaseBlackOilModelAD`:
it reads a legacy `raw_fluid` and snake-case `b_g` first.
"""

from typing import Any

from pymrst.modules.ad_blackoil.utils.get_b_o_bo import _raw_fluid


def get_b_g_bo(model: Any, p: Any, rv: Any = None, isLiquid: Any = None) -> Any:
    """`bG` at pressure `p`; wet gas (`rv`, `isLiquid`) when `model.vapoil`.

    MRST reads `model.vapoil` unguarded, so a model without it is an error.
    """
    if model.vapoil:
        return model.fluid.bG(p, rv, isLiquid)
    return model.fluid.bG(p)


def get_b_g_bo_ad(model, pressure, rv=None, is_liquid=None):
    """Evaluate reciprocal gas formation volume factor.

    MRST calls the wet-gas three-argument PVT function only when vaporized oil
    is enabled. For dry gas, ``rv`` and ``is_liquid`` are ignored.
    """
    fluid = _raw_fluid(model)
    if fluid is None:
        raise AttributeError("model must define fluid or raw_fluid")
    b_g = getattr(fluid, "b_g", getattr(fluid, "bG", None))
    if b_g is None:
        raise AttributeError("model fluid must define b_g or bG")
    if bool(getattr(model, "vapoil", False)):
        return b_g(pressure, rv, is_liquid)
    return b_g(pressure)


getbG_BO = get_b_g_bo
__all__ = ["get_b_g_bo", "getbG_BO", "get_b_g_bo_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    FLUID = SimpleNamespace(bG=lambda p, *rest: 3 + 0.4*p + (0.5*rest[0] + 0.6*rest[1]
                                                           if rest else 0),
                            b_g=lambda p, *rest: -1.0)

    def test_dry_gas_reads_pressure_only():
        model = SimpleNamespace(fluid=FLUID, vapoil=False)
        np.testing.assert_allclose(get_b_g_bo(model, np.array([10.0]), 1.0, True), [7.0])
        print("ok test_dry_gas_reads_pressure_only")

    def test_wet_gas_passes_rv_and_flag():
        model = SimpleNamespace(fluid=FLUID, vapoil=True)
        np.testing.assert_allclose(get_b_g_bo(model, np.array([10.0]), np.array([1.0]),
                                              np.array([True])), [8.1])
        print("ok test_wet_gas_passes_rv_and_flag")

    def test_a_model_without_vapoil_is_an_error_as_in_mrst():
        try:
            get_b_g_bo(SimpleNamespace(fluid=FLUID), 0.0)
        except AttributeError:
            print("ok test_a_model_without_vapoil_is_an_error_as_in_mrst")
        else:
            raise AssertionError("expected AttributeError")

    def test_legacy_version_prefers_snake_case():
        assert get_b_g_bo_ad(SimpleNamespace(fluid=FLUID), 0.0) == -1.0
        print("ok test_legacy_version_prefers_snake_case")

    test_dry_gas_reads_pressure_only()
    test_wet_gas_passes_rv_and_flag()
    test_a_model_without_vapoil_is_an_error_as_in_mrst()
    test_legacy_version_prefers_snake_case()
    print("\nAll tests passed!")
