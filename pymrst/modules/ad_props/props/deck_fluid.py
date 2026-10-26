"""Bridge between MRST's fluid *struct* and PyMRST's fluid *object*.

No MRST counterpart, and there could not be one: MATLAB structs are duck-typed,
so `initDeckADIFluid` returns `fluid.bO` and every model reads `fluid.bO`. This
port has two spellings instead - `init_deck_adi_fluid` returns MRST's names in
a dict, and the models were written against the `BlackOilFluid` dataclass with
`b_o`, `mu_o`, `rhoS`. Both are ports of real MRST things; the gap between them
is an artefact of translating a struct into a class, and this closes it.

What it is *not* is a re-implementation. Every property function here is the
one the deck reader built; only the name it answers to changes.

Two things the translation has to get right:

* **which oil relperm.** A deck's SWOF gives `krOW`, the oil relperm in the
  oil-water system; SGOF gives `krOG`. A two-phase water-oil model wants
  `krOW` directly, and a three-phase one wants MRST's saturation-weighted
  blend of the two. Handing a two-phase model the blend needs a gas saturation
  it does not have;
* **the connate water saturation.** It is not a keyword; it is the first
  column of the SWOF table, which `init_deck_adi_fluid` records as
  `krPts['w'][region][0]`. `kr_o`'s three-phase blend divides by
  `sg + sw - swcon`, so a wrong `swcon` is a wrong oil relperm everywhere, not
  just at the endpoint.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from pymrst.modules.ad_props.props.init_deck_adi_fluid import BlackOilFluid


def deck_fluid_object(struct: Any, *, region: int = 0) -> BlackOilFluid:
    """A `BlackOilFluid` view of the struct `init_deck_adi_fluid` returns.

    A **multi-region** deck gives every curve as a list, one entry per PVT or
    saturation region; a single-region deck gives a bare callable. Both shapes
    reach here - Norne is multi-region - and handing a list to the model puts
    a `list` where a function belongs, which raises only once the equilibrium
    tries to call it. `region` selects which one.
    """
    if isinstance(struct, BlackOilFluid):
        return struct
    if not isinstance(struct, dict):
        raise TypeError(
            "expected the fluid struct init_deck_adi_fluid returns, or a "
            f"BlackOilFluid; got {type(struct).__name__}")

    b_w = _need(struct, "bW", region)
    b_o = _need(struct, "bO", region)
    mu_w, mu_o = _need(struct, "muW", region), _need(struct, "muO", region)
    kr_w = _need(struct, "krW", region)
    kr_ow = _one(struct.get("krOW"), region) or _one(struct.get("krO"), region)
    if kr_ow is None:
        raise KeyError(
            "the deck fluid has neither krOW nor krO; a SWOF or SOF3 table is "
            "needed for the oil relative permeability")

    has_gas = "bG" in struct and "krG" in struct
    kr_og = _one(struct.get("krOG"), region) or kr_ow
    b_g = _one(struct.get("bG"), region) or _unused("bG")
    mu_g = _one(struct.get("muG"), region) or _unused("muG")
    kr_g = _one(struct.get("krG"), region) or _unused("krG")

    rho_s = np.array([_scalar(struct, "rhoWS", region),
                      _scalar(struct, "rhoOS", region),
                      _scalar(struct, "rhoGS", region, default=1.0)],
                     dtype=float)

    fluid = BlackOilFluid(
        rhoS=rho_s,
        b_w=b_w, mu_w=mu_w,
        b_o=_widen(b_o), mu_o=_widen(mu_o),
        rs_sat=_one(struct.get("rsSat"), region),
        b_g=_widen(b_g), mu_g=_widen(mu_g),
        kr_w=kr_w, kr_g=kr_g, kr_ow=kr_ow, kr_og=kr_og,
        pv_mult=_one(struct.get("pvMultR"), region) or (lambda p: 1.0),
        swcon=connate_water(struct, region),
        pc_ow=_one(struct.get("pcOW"), region),
        pc_og=_one(struct.get("pcOG"), region),
        rv_sat=_one(struct.get("rvSat"), region),
        vapoil=bool(struct.get("rvSat") is not None and has_gas),
    )
    # Stated rather than inferred. The gas slots above are filled with a
    # refusing placeholder rather than left `None`, so "has a `b_g`" is not a
    # usable test for whether this fluid has gas - and that is exactly the test
    # a model would otherwise reach for when choosing its phase adapter.
    fluid.n_phases_declared = 3 if has_gas else 2
    return fluid


def connate_water(struct: Any, region: int = 0) -> float:
    """The connate water saturation, from SWOF's first row.

    Not a deck keyword - the first saturation the water table tabulates.
    Defaulting it to zero makes the three-phase oil relperm divide by `sg + sw`
    rather than `sg + sw - swcon`, which is wrong in every cell rather than
    only at the endpoint.
    """
    if isinstance(struct, BlackOilFluid):
        return float(struct.swcon)
    points = (struct or {}).get("krPts") or {}
    water = points.get("w")
    if water is None:
        return 0.0
    rows = np.atleast_2d(np.asarray(water, dtype=float))
    # ECLIPSE numbers saturation regions (SATNUM) and PVT regions (PVTNUM)
    # separately, and a deck may have more of one than the other. Clamping
    # rather than indexing blindly keeps a single-SATNUM deck usable from a
    # multi-region PVT caller, which is the common asymmetry.
    return float(rows[min(int(region), rows.shape[0] - 1)][0])


def _need(struct: Dict[str, Any], name: str, region: int = 0) -> Callable:
    found = _one(struct.get(name), region)
    if found is None:
        raise KeyError(f"the deck fluid has no {name!r}")
    return found


def _one(curve: Any, region: int = 0):
    """One region's curve, from a bare callable or a per-region list.

    A multi-region deck stores a list; calling it raises `'list' object is not
    callable` - but only once something evaluates it, which for `rsSat` is
    inside the equilibrium, several layers from the cause.
    """
    if curve is None or callable(curve):
        return curve
    try:
        entries = list(curve)
    except TypeError:
        return curve
    if not entries:
        return None
    return entries[min(max(int(region), 0), len(entries) - 1)]


def _scalar(struct: Dict[str, Any], name: str, region: int,
            default: Optional[float] = None) -> float:
    """One PVT region's value of a per-region constant."""
    value = struct.get(name)
    if value is None:
        if default is None:
            raise KeyError(f"the deck fluid has no {name!r}")
        return float(default)
    flat = np.ravel(np.asarray(value, dtype=float))
    return float(flat[min(region, flat.size - 1)])


def _widen(function: Callable) -> Callable:
    """Accept the extra `(rs, saturated)` arguments a live-oil model passes.

    A dead-oil deck's `bO` takes pressure alone, and the black-oil model calls
    every `b` with three arguments. Passing them through to a two-argument
    function raises; dropping them silently would be wrong for a live-oil deck,
    which is why they are dropped only when the function will not take them.
    """
    def widened(p, *extra):
        if not extra:
            return function(p)
        try:
            return function(p, *extra)
        except TypeError:
            return function(p)

    return widened


def _unused(name: str) -> Callable:
    def refuse(*_args, **_kwargs):
        raise KeyError(
            f"the deck declares no gas phase, so {name!r} does not exist; a "
            "three-phase model cannot be built from this fluid")

    return refuse


deckFluidObject = deck_fluid_object

__all__ = ["connate_water", "deckFluidObject", "deck_fluid_object"]


if __name__ == "__main__":

    def _struct(**extra):
        struct = {
            "bW": lambda p: np.ones_like(np.asarray(p, float)),
            "bO": lambda p: np.full(np.size(p), 1.2),
            "muW": lambda p: np.full(np.size(p), 1.0e-3),
            "muO": lambda p: np.full(np.size(p), 5.0e-3),
            "krW": lambda s: np.asarray(s, float),
            "krOW": lambda s: np.asarray(s, float)**2,
            "pvMultR": lambda p: np.ones_like(np.asarray(p, float)),
            "rhoWS": np.array([1000.0]),
            "rhoOS": np.array([900.0]),
            "rhoGS": np.array([1.0]),
            "krPts": {"w": np.array([[0.1, 0.2, 0.9, 0.75]])},
        }
        struct.update(extra)
        return struct

    def test_the_surface_densities_come_out_in_mrsts_phase_order():
        """`rhoS` is `[water, oil, gas]`. Any other order puts the oil density
        on the water equation, and the run still converges."""
        fluid = deck_fluid_object(_struct())
        assert np.allclose(fluid.rhoS, [1000.0, 900.0, 1.0]), fluid.rhoS
        print("ok test_the_surface_densities_come_out_in_mrsts_phase_order")

    def test_a_per_region_density_is_reduced_to_its_region():
        struct = _struct(rhoOS=np.array([900.0, 850.0]))
        assert deck_fluid_object(struct, region=1).rhoS[1] == 850.0
        print("ok test_a_per_region_density_is_reduced_to_its_region")

    def test_the_property_functions_are_the_decks_own():
        """Not re-implemented - the same objects, under different names."""
        struct = _struct()
        fluid = deck_fluid_object(struct)
        assert fluid.b_w is struct["bW"]
        assert fluid.kr_w is struct["krW"]
        assert fluid.kr_ow is struct["krOW"]
        print("ok test_the_property_functions_are_the_decks_own")

    def test_the_connate_water_comes_from_the_swof_table():
        """`swcon` is the first row of SWOF, not a keyword. Defaulting it to
        zero makes the three-phase oil relperm divide by the wrong span."""
        assert deck_fluid_object(_struct()).swcon == 0.1
        print("ok test_the_connate_water_comes_from_the_swof_table")

    def test_a_two_phase_deck_uses_krow_for_both_oil_curves():
        """With no SGOF there is no oil-gas curve, and the three-phase blend
        of `krOW` with itself is `krOW` - so a two-phase model is unaffected
        by machinery it does not use."""
        fluid = deck_fluid_object(_struct())
        assert fluid.kr_og is fluid.kr_ow
        print("ok test_a_two_phase_deck_uses_krow_for_both_oil_curves")

    def test_a_dead_oil_b_accepts_the_live_oil_call():
        """The black-oil model calls every `b` with `(p, rs, saturated)`. A
        dead-oil deck's `bO` takes pressure alone."""
        fluid = deck_fluid_object(_struct())
        p = np.array([2.0e7, 2.1e7])
        assert np.allclose(fluid.b_o(p, np.zeros(2), np.zeros(2)), 1.2)
        assert np.allclose(fluid.b_o(p), 1.2)
        print("ok test_a_dead_oil_b_accepts_the_live_oil_call")

    def test_a_live_oil_b_still_receives_its_ratio():
        """The extra arguments are dropped only when the function will not
        take them; a live-oil `bO` must still see `rs`."""
        seen = {}

        def b_o(p, rs=None, saturated=None):
            seen["rs"] = rs
            return np.full(np.size(p), 1.2)

        fluid = deck_fluid_object(_struct(bO=b_o))
        fluid.b_o(np.array([2.0e7]), np.array([50.0]), np.array([True]))
        assert seen["rs"] is not None and seen["rs"][0] == 50.0
        print("ok test_a_live_oil_b_still_receives_its_ratio")

    def test_a_gasless_deck_refuses_the_gas_properties_clearly():
        """A silent zero would give the gas equation no fluid at all."""
        fluid = deck_fluid_object(_struct())
        try:
            fluid.b_g(np.array([2.0e7]))
        except KeyError as refused:
            assert "no gas phase" in str(refused)
            print("ok test_a_gasless_deck_refuses_the_gas_properties_clearly")
            return
        raise AssertionError("a two-phase deck has no gas b-factor")

    def test_a_deck_without_an_oil_relperm_is_refused():
        struct = _struct()
        del struct["krOW"]
        try:
            deck_fluid_object(struct)
        except KeyError as refused:
            assert "krOW" in str(refused)
            print("ok test_a_deck_without_an_oil_relperm_is_refused")
            return
        raise AssertionError("the oil relperm cannot be invented")

    def test_an_already_converted_fluid_passes_through():
        fluid = deck_fluid_object(_struct())
        assert deck_fluid_object(fluid) is fluid
        print("ok test_an_already_converted_fluid_passes_through")

    test_the_surface_densities_come_out_in_mrsts_phase_order()
    test_a_per_region_density_is_reduced_to_its_region()
    test_the_property_functions_are_the_decks_own()
    test_the_connate_water_comes_from_the_swof_table()
    test_a_two_phase_deck_uses_krow_for_both_oil_curves()
    test_a_dead_oil_b_accepts_the_live_oil_call()
    test_a_live_oil_b_still_receives_its_ratio()
    test_a_gasless_deck_refuses_the_gas_properties_clearly()
    test_a_deck_without_an_oil_relperm_is_refused()
    test_an_already_converted_fluid_passes_through()
    print("\nAll tests passed!")
