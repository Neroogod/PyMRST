"""MRST ``ad-blackoil/utils/getbO_BO.m``: the reciprocal oil formation
volume factor, `fluid.bO(p, rs, isSaturated)` for a model with dissolved
gas and `fluid.bO(p)` otherwise.

`get_b_o_bo_ad` is the earlier version, kept for `ThreePhaseBlackOilModelAD`:
it reads a legacy `raw_fluid` and snake-case `b_o` first.
"""

from typing import Any


def get_b_o_bo(model: Any, p: Any, rs: Any = None, isSaturated: Any = None) -> Any:
    """`bO` at pressure `p`; live oil (`rs`, `isSaturated`) when `model.disgas`."""
    if getattr(model, "disgas", False):
        return model.fluid.bO(p, rs, isSaturated)
    return model.fluid.bO(p)


def _raw_fluid(model):
    """Return the MRST-style fluid object owned by ``model``."""
    return getattr(model, "raw_fluid", getattr(model, "fluid", None))


def get_b_o_bo_ad(model, pressure, rs=None, is_saturated=None):
    """Evaluate reciprocal oil formation volume factor.

    MRST calls the live-oil three-argument PVT function only when dissolved
    gas is enabled. For dead oil, ``rs`` and ``is_saturated`` are deliberately
    ignored and the pressure-only function is used.
    """
    fluid = _raw_fluid(model)
    if fluid is None:
        raise AttributeError("model must define fluid or raw_fluid")
    b_o = getattr(fluid, "b_o", getattr(fluid, "bO", None))
    if b_o is None:
        raise AttributeError("model fluid must define b_o or bO")
    if bool(getattr(model, "disgas", False)):
        return b_o(pressure, rs, is_saturated)
    return b_o(pressure)


getbO_BO = get_b_o_bo
__all__ = ["get_b_o_bo", "getbO_BO", "get_b_o_bo_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    FLUID = SimpleNamespace(bO=lambda p, *rest: 2 + 0.1*p + (0.2*rest[0] + 0.3*rest[1]
                                                           if rest else 0),
                            b_o=lambda p, *rest: -1.0)

    def test_dead_oil_reads_pressure_only():
        model = SimpleNamespace(fluid=FLUID, disgas=False)
        np.testing.assert_array_equal(get_b_o_bo(model, np.array([10.0]), 1.0, True), [3.0])
        print("ok test_dead_oil_reads_pressure_only")

    def test_live_oil_passes_rs_and_flag():
        model = SimpleNamespace(fluid=FLUID, disgas=True)
        np.testing.assert_allclose(get_b_o_bo(model, np.array([10.0]), np.array([1.0]),
                                              np.array([True])), [3.5])
        print("ok test_live_oil_passes_rs_and_flag")

    def test_a_model_without_disgas_is_dead_oil():
        np.testing.assert_array_equal(get_b_o_bo(SimpleNamespace(fluid=FLUID), 0.0), 2.0)
        print("ok test_a_model_without_disgas_is_dead_oil")

    def test_legacy_version_prefers_snake_case():
        assert get_b_o_bo_ad(SimpleNamespace(fluid=FLUID), 0.0) == -1.0
        print("ok test_legacy_version_prefers_snake_case")

    test_dead_oil_reads_pressure_only()
    test_live_oil_passes_rs_and_flag()
    test_a_model_without_disgas_is_dead_oil()
    test_legacy_version_prefers_snake_case()
    print("\nAll tests passed!")
