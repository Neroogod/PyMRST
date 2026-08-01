"""The relative-permeability dispatcher (MRST ``props/assignRelPerm.m``).

Once the saturation keywords have been assigned, this wires up ``f['relPerm']``:
the single entry point that returns every phase's relative permeability at
once. Which form it takes is decided by *which curves the deck supplied*, not
by a flag:

- no ``krOG``  -> two-phase water/oil, called as ``relPerm(sw)``;
- no ``krOW``  -> two-phase oil/gas, called as ``relPerm(sg)``;
- both        -> three-phase, called as ``relPerm(sw, sg)``.

The three-phase oil relative permeability is the saturation-weighted average
of the two two-phase curves, which is ECLIPSE's default model:

    krO = (1 - ww) krOG + ww krOW,   ww = (Sw - Swcon) / (Sg + Sw - Swcon)

`swcon` is clamped to just below Sw before use. Without that clamp a cell at
exactly connate water gives ww = 0/0; the clamp keeps the weight finite and
equal to zero there, which is the physically right answer.

Multi-region fluids are not supported here, and that is MRST's limitation too:
`initDeckADIFluid` unwraps a one-element list into a bare function before
calling this, so a genuinely multi-region fluid would leave `krW` a list and
`f.krW(sw)` would index rather than evaluate it. This port raises instead of
producing nonsense.
"""

from typing import Any, Dict

import numpy as np

from pymrst.utils.adi import value


def _one(curve, name: str):
    """A single callable, unwrapping the one-element list form."""
    if callable(curve):
        return curve
    if isinstance(curve, (list, tuple)):
        if len(curve) == 1:
            return curve[0]
        raise ValueError(
            f"relPerm needs a single {name} curve but the fluid has "
            f"{len(curve)} regions; MRST has the same limitation")
    raise TypeError(f"{name} is neither callable nor a list of curves")


def assign_relperm(f: Dict[str, Any]) -> Dict[str, Any]:
    """Attach ``relPerm`` (and ``relPermSft`` when surfactant is present)."""
    has_og, has_ow = "krOG" in f, "krOW" in f

    if not has_og:                                   # water/oil
        kr_w = _one(f["krW"], "krW")
        kr_o = _one(f.get("krO") or f["krOW"], "krO")

        def rel_perm(sw, *args, **kwargs):
            return kr_w(sw, *args, **kwargs), kr_o(1 - sw, *args, **kwargs)

        f["relPerm"] = rel_perm

        if "krWSft" in f:
            kr_w_sft = _one(f["krWSft"], "krWSft")
            kr_o_sft = _one(f.get("krOSft") or f["krOWSft"], "krOSft")

            def rel_perm_sft(sw, *args, **kwargs):
                return (kr_w_sft(sw, *args, **kwargs),
                        kr_o_sft(1 - sw, *args, **kwargs))

            f["relPermSft"] = rel_perm_sft

    elif not has_ow:                                 # oil/gas
        kr_g = _one(f["krG"], "krG")
        kr_o = _one(f.get("krO") or f["krOG"], "krO")

        def rel_perm(sg, *args, **kwargs):
            return kr_o(1 - sg, *args, **kwargs), kr_g(sg, *args, **kwargs)

        f["relPerm"] = rel_perm

    else:                                            # three-phase
        kr_w = _one(f["krW"], "krW")
        kr_g = _one(f["krG"], "krG")
        kr_ow = _one(f["krOW"], "krOW")
        kr_og = _one(f["krOG"], "krOG")
        s_wcon = f.get("sWcon", 0.0)

        def rel_perm(sw, sg, *args, **kwargs):
            swcon = np.minimum(s_wcon, value(sw) - 1e-5)
            weight = (sw - swcon) / (sg + sw - swcon)
            so = 1 - sw - sg
            kr_o = ((1 - weight) * kr_og(so, *args, **kwargs)
                    + weight * kr_ow(so, *args, **kwargs))
            return (kr_w(sw, *args, **kwargs), kr_o,
                    kr_g(sg, *args, **kwargs))

        f["relPerm"] = rel_perm
    return f


assignRelPerm = assign_relperm


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_sgof import assign_sgof
    from pymrst.modules.ad_props.props.assign_swof import assign_swof
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    SWOF = np.array([[0.10, 0.00, 1.00, 0.0],
                     [0.50, 0.20, 0.40, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])
    SGOF = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.40, 0.30, 0.35, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])

    def _three_phase():
        reg = Regions()
        f = assign_swof({}, [SWOF], reg)
        f = assign_sgof(f, [SGOF], reg)
        f["krW"], f["krOW"] = f["krW"][0], f["krOW"][0]
        f["krG"], f["krOG"] = f["krG"][0], f["krOG"][0]
        f["sWcon"] = 0.10
        return assign_relperm(f)

    def test_water_oil_returns_two_curves() -> None:
        f = assign_swof({}, [SWOF], Regions())
        f["krW"], f["krOW"] = f["krW"][0], f["krOW"][0]
        f = assign_relperm(f)
        kr_w, kr_o = f["relPerm"](np.array([0.50]))
        assert np.isclose(kr_w, 0.20) and np.isclose(kr_o, 0.40)
        print("ok test_water_oil_returns_two_curves")

    def test_oil_gas_returns_oil_first() -> None:
        """The order is (krO, krG), not (krG, krO)."""
        f = assign_swof({}, [SWOF], Regions())
        f = assign_sgof(f, [SGOF], Regions())
        del f["krOW"]
        f["krG"], f["krOG"] = f["krG"][0], f["krOG"][0]
        f = assign_relperm(f)
        kr_o, kr_g = f["relPerm"](np.array([0.40]))
        assert np.isclose(kr_g, 0.30), kr_g
        print("ok test_oil_gas_returns_oil_first")

    def test_three_phase_returns_water_oil_gas() -> None:
        f = _three_phase()
        out = f["relPerm"](np.array([0.30]), np.array([0.20]))
        assert len(out) == 3
        print("ok test_three_phase_returns_water_oil_gas")

    def test_three_phase_oil_is_between_the_two_curves() -> None:
        f = _three_phase()
        sw, sg = np.array([0.30]), np.array([0.20])
        _, kr_o, _ = f["relPerm"](sw, sg)
        so = 1 - sw - sg
        low, high = sorted([f["krOW"](so)[0], f["krOG"](so)[0]])
        assert low - 1e-12 <= kr_o[0] <= high + 1e-12, (low, kr_o, high)
        print("ok test_three_phase_oil_is_between_the_two_curves")

    def test_the_weight_is_well_defined_at_connate_water() -> None:
        """Without the clamp this is 0/0."""
        f = _three_phase()
        _, kr_o, _ = f["relPerm"](np.array([0.10]), np.array([0.20]))
        assert np.all(np.isfinite(kr_o)), kr_o
        print("ok test_the_weight_is_well_defined_at_connate_water")

    def test_no_free_gas_gives_the_water_oil_curve() -> None:
        f = _three_phase()
        sw, sg = np.array([0.50]), np.array([0.0])
        _, kr_o, _ = f["relPerm"](sw, sg)
        assert np.isclose(kr_o, f["krOW"](1 - sw - sg))
        print("ok test_no_free_gas_gives_the_water_oil_curve")

    def test_the_form_follows_the_curves_the_deck_supplied() -> None:
        f = assign_swof({}, [SWOF], Regions())
        f["krW"], f["krOW"] = f["krW"][0], f["krOW"][0]
        assign_relperm(f)
        f["relPerm"](np.array([0.5]))          # one argument: water/oil
        g = _three_phase()
        g["relPerm"](np.array([0.3]), np.array([0.2]))   # two: three-phase
        print("ok test_the_form_follows_the_curves_the_deck_supplied")

    def test_a_multi_region_fluid_is_rejected() -> None:
        reg = Regions(sat=2)
        f = assign_swof({}, [SWOF, SWOF], reg)
        try:
            assign_relperm(f)
        except ValueError as error:
            assert "2 regions" in str(error), str(error)
            print("ok test_a_multi_region_fluid_is_rejected")
            return
        raise AssertionError("a multi-region fluid must be rejected")

    def test_a_one_element_list_is_unwrapped() -> None:
        f = assign_swof({}, [SWOF], Regions())
        f = assign_relperm(f)
        kr_w, _ = f["relPerm"](np.array([0.50]))
        assert np.isclose(kr_w, 0.20)
        print("ok test_a_one_element_list_is_unwrapped")

    test_water_oil_returns_two_curves()
    test_oil_gas_returns_oil_first()
    test_three_phase_returns_water_oil_gas()
    test_three_phase_oil_is_between_the_two_curves()
    test_the_weight_is_well_defined_at_connate_water()
    test_no_free_gas_gives_the_water_oil_curve()
    test_the_form_follows_the_curves_the_deck_supplied()
    test_a_multi_region_fluid_is_rejected()
    test_a_one_element_list_is_unwrapped()
    print("\nAll tests passed!")
