"""Water PVT from PVTW (MRST ``ad-props/props/assignPVTW.m``).

Water is compressible but never changes composition, so its properties are
closed forms around a reference point rather than tables:

    b_w(p) = (1 + X + X^2/2) / B_wr,   X =  c_w (p - p_ref)
    mu_w(p) = mu_wr / (1 + Y + Y^2/2), Y = -c_mu (p - p_ref)

Those are ECLIPSE's second-order expansions, not exponentials. They agree to
first order and diverge measurably far from the reference pressure, so the
expansion is the definition here rather than an approximation of one.

A zero viscosibility means constant viscosity, and MRST takes that branch
explicitly rather than letting the formula collapse - which matters because
it makes muW return a constant of the right length rather than a scalar.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.utils.adi import value


def b_w(p, c_w: float, b_ref: float, p_ref: float):
    """Water shrinkage factor, ECLIPSE's second-order expansion."""
    x = c_w * (p - p_ref)
    return (1.0 + x + x ** 2 / 2.0) / b_ref


def mu_w(p, p_ref: float, mu_ref: float, viscosibility: float):
    """Water viscosity, ECLIPSE's second-order expansion."""
    y = -viscosibility * (p - p_ref)
    return mu_ref / (1.0 + y + y ** 2 / 2.0)


def _tabulated(fn: Callable, reg) -> Callable:
    """Evaluate ``fn`` directly, or through a table when ``reg.prange`` is set."""
    pressures = reg.prange
    if pressures is None or len(pressures) == 0:
        return fn
    sampled = fn(np.asarray(pressures, dtype=float))
    return lambda p: reg.interp_uniform(pressures, sampled, p)


def assign_pvtw(f: Dict[str, Any], pvtw, reg) -> Dict[str, Any]:
    """Store water compressibility, reference viscosity, ``bW`` and ``muW``."""
    pvtw = np.atleast_2d(np.asarray(pvtw, dtype=float))
    if pvtw.shape[0] != reg.pvt:
        raise ValueError(
            f"PVTW has {pvtw.shape[0]} records but TABDIMS declares "
            f"{reg.pvt} PVT regions")

    f["cW"] = pvtw[:, 2]
    f["muWr"] = pvtw[:, 3]

    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []
    for index in range(pvtw.shape[0]):
        p_ref, b_ref, compressibility, mu_ref, viscosibility = pvtw[index, :5]
        b_functions.append(_tabulated(
            lambda p, *, c=compressibility, b=b_ref, pr=p_ref: b_w(p, c, b, pr),
            reg))
        if viscosibility > 0:
            mu_functions.append(_tabulated(
                lambda p, *, pr=p_ref, m=mu_ref, v=viscosibility:
                    mu_w(p, pr, m, v), reg))
        else:
            mu_functions.append(
                lambda p, *, m=mu_ref: np.full(np.size(value(p)), m))

    f["bW"], f["muW"] = b_functions, mu_functions
    return f


assignPVTW = assign_pvtw


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    RECORD = [2.7680382e7, 1.029, 4.5396812e-10, 3.1e-4, 0.0]

    def test_b_is_the_reciprocal_reference_fvf_at_p_ref() -> None:
        f = assign_pvtw({}, [RECORD], Regions())
        assert np.isclose(f["bW"][0](RECORD[0]), 1.0 / RECORD[1])
        print("ok test_b_is_the_reciprocal_reference_fvf_at_p_ref")

    def test_water_expands_as_pressure_falls() -> None:
        f = assign_pvtw({}, [RECORD], Regions())
        below = f["bW"][0](RECORD[0] * 0.5)
        above = f["bW"][0](RECORD[0] * 1.5)
        assert below < above, "b increases with pressure for compressible water"
        print("ok test_water_expands_as_pressure_falls")

    def test_zero_viscosibility_gives_constant_viscosity() -> None:
        f = assign_pvtw({}, [RECORD], Regions())
        out = f["muW"][0](np.array([1e6, 5e7, 1e8]))
        assert np.allclose(out, RECORD[3]), out
        assert out.shape == (3,), "must return one value per pressure"
        print("ok test_zero_viscosibility_gives_constant_viscosity")

    def test_positive_viscosibility_makes_viscosity_rise_with_pressure() -> None:
        record = list(RECORD)
        record[4] = 1e-9
        f = assign_pvtw({}, [record], Regions())
        assert f["muW"][0](1e8) > f["muW"][0](1e6)
        print("ok test_positive_viscosibility_makes_viscosity_rise_with_pressure")

    def test_the_expansion_is_second_order_not_exponential() -> None:
        """Far from p_ref the two differ; the deck's convention is the expansion."""
        p_ref, b_ref, c = 1e7, 1.0, 1e-8
        x = c * (1e8 - p_ref)
        assert np.isclose(b_w(1e8, c, b_ref, p_ref), (1 + x + x ** 2 / 2))
        assert not np.isclose(b_w(1e8, c, b_ref, p_ref), np.exp(x), rtol=1e-4)
        print("ok test_the_expansion_is_second_order_not_exponential")

    def test_one_function_per_pvt_region() -> None:
        f = assign_pvtw({}, [RECORD, [1e7, 1.0, 1e-9, 5e-4, 0.0]],
                        Regions(pvt=2))
        assert len(f["bW"]) == 2 and len(f["muW"]) == 2
        assert not np.isclose(f["muW"][0](1e7), f["muW"][1](1e7))
        print("ok test_one_function_per_pvt_region")

    def test_a_record_count_mismatch_is_rejected() -> None:
        try:
            assign_pvtw({}, [RECORD], Regions(pvt=2))
        except ValueError as error:
            assert "PVT regions" in str(error)
            print("ok test_a_record_count_mismatch_is_rejected")
            return
        raise AssertionError("a short PVTW must be rejected")

    def test_prange_tabulates_without_changing_the_values() -> None:
        pressures = np.linspace(1e6, 1e8, 200)
        direct = assign_pvtw({}, [RECORD], Regions())["bW"][0]
        tabulated = assign_pvtw({}, [RECORD], Regions(prange=pressures))["bW"][0]
        probe = np.array([2e6, 3e7, 9e7])
        assert np.allclose(direct(probe), tabulated(probe), rtol=1e-6)
        print("ok test_prange_tabulates_without_changing_the_values")

    test_b_is_the_reciprocal_reference_fvf_at_p_ref()
    test_water_expands_as_pressure_falls()
    test_zero_viscosibility_gives_constant_viscosity()
    test_positive_viscosibility_makes_viscosity_rise_with_pressure()
    test_the_expansion_is_second_order_not_exponential()
    test_one_function_per_pvt_region()
    test_a_record_count_mismatch_is_rejected()
    test_prange_tabulates_without_changing_the_values()
    print("\nAll tests passed!")
