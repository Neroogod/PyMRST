"""MRST ``ad-core/utils/selectModelFromDeck.m``.

`select_model_from_deck` is the port: the AD-OO model a deck's RUNSPEC calls
for - MRST's `GenericBlackOilModel` with the deck's phases, `disgas` and
`vapoil` - with operators, rock, the deck's facility and aquifers, and
`minimumPressure = 0`. The compositional and EOR branches need classes not
yet on the AD-OO base and say so.

`select_model_from_deck_ad`, below, is PyMRST's earlier factory for the
legacy models, kept for its callers. Its notes follow.

Pick the simulation model class for a parsed deck (legacy).

A deck's RUNSPEC section declares which phases and physics are active - water,
oil, gas, and options like compositional (COMPS), polymer, surfactant - and that
declaration determines which `PhysicalModel` subclass should simulate it. This
is the factory at the top of the deck-to-simulation path: it reads those flags
and returns the matching model, already carrying the deck, rock and operators.

This port covers the black-oil and single-phase-water branches that PyMRST has
models for. `disgas`/`vapoil` are resolved exactly as MRST does - from the PVTO
/ PVTG tables under a BLACKOIL run, then overridden by an explicit RUNSPEC
DISGAS / VAPOIL. The compositional and EOR (polymer/surfactant) branches are
refused with a clear message: their deck-construction path (an EOS model, the
ad-eor models) is not ported, so silently returning a black-oil model would be
wrong.
"""

from typing import Any, Dict, Optional


def _check(runspec: Dict[str, Any], name: str) -> bool:
    """MRST's `check`: a truthy RUNSPEC flag, case-insensitively."""
    for key in (name.upper(), name.lower(), name):
        if key in runspec and runspec[key]:
            return True
    return False


def _has_props(deck: Dict[str, Any], name: str) -> bool:
    props = deck.get("PROPS", deck.get("props", {})) or {}
    return name.upper() in props or name.lower() in props or name in props


def _runspec(deck: Dict[str, Any]) -> Dict[str, Any]:
    return deck.get("RUNSPEC", deck.get("runspec", {})) or {}


def deck_model_choice(deck: Dict[str, Any]) -> Dict[str, Any]:
    """The model decision from a deck, without building anything.

    Returns `{"kind": "water"}` or `{"kind": "blackoil", "disgas": bool,
    "vapoil": bool}`; raises `NotImplementedError` for the compositional and EOR
    branches PyMRST has no deck path for. Separated from construction so the
    routing can be checked without a full 3-phase fluid.
    """
    rs = _runspec(deck)
    hasgas = _check(rs, "gas")
    hasoil = _check(rs, "oil")
    haswat = _check(rs, "water")

    if _check(rs, "comps"):
        return {"kind": "compositional", "water": haswat}
    if _check(rs, "polymer") or _check(rs, "surfact"):
        raise NotImplementedError(
            "EOR (polymer/surfactant) deck models are not ported (ad-eor)"
        )

    if haswat and not hasoil and not hasgas:
        return {"kind": "water"}

    # black-oil (two- or three-phase). Resolve disgas/vapoil exactly as MRST.
    disgas = False
    vapoil = False
    if _check(rs, "BLACKOIL"):
        disgas = _has_props(deck, "PVTO")
        vapoil = _has_props(deck, "PVTG")
    if "VAPOIL" in rs:
        vapoil = bool(rs["VAPOIL"])
    if "DISGAS" in rs:
        disgas = bool(rs["DISGAS"])
    return {"kind": "blackoil", "disgas": disgas, "vapoil": vapoil}


def select_model_from_deck_ad(g, rock, fluid, deck: Dict[str, Any], **kwargs):
    """Return the model matching the deck's RUNSPEC flags, built and ready.

    `g`, `rock`, `fluid` are the usual simulation inputs; `deck` is a parsed deck
    with a `RUNSPEC` section (and `PROPS` for the PVT tables). Extra keyword
    arguments are forwarded to the model constructor.
    """
    choice = deck_model_choice(deck)
    wells = kwargs.pop("wells", ())
    # `init_deck_adi_fluid` returns MRST's fluid *struct*; the models are
    # written against the `BlackOilFluid` object. Same property functions, one
    # naming - see `ad_props.props.deck_fluid`. MRST needs no such step
    # because a MATLAB struct is duck-typed.
    if isinstance(fluid, dict):
        from pymrst.modules.ad_props.props.deck_fluid import deck_fluid_object
        fluid = deck_fluid_object(fluid)
    if choice["kind"] == "compositional":
        model = _build_compositional(g, rock, deck, wells, choice, **kwargs)
    elif choice["kind"] == "water":
        from pymrst.modules.ad_blackoil.models.water_model import WaterModelAD

        model = WaterModelAD(g, rock, fluid, wells, **kwargs)
    else:
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModelAD,
        )

        model = GenericBlackOilModelAD(g, rock, fluid, wells,
                                     disgas=choice["disgas"],
                                     vapoil=choice["vapoil"], **kwargs)
    model.input_data = deck
    model.rock = rock
    return model


def _build_compositional(g, rock, deck, wells, choice, **kwargs):
    """A `GenericNaturalVariablesModel` from a COMPS deck.

    MRST writes this as `GenericNaturalVariablesModel(G, rock, fluid, eos,
    'water', haswat)` - it hands over both the black-oil-style fluid and the
    equation of state. PyMRST's compositional model takes the **mixture** and a
    temperature directly, because the phase behaviour is the EOS's job here and
    there is no separate PVT fluid to carry.

    So the port reads the EOS from the deck exactly as MRST does
    (`init_deck_eos_model`) and then unpacks the mixture out of it, rather than
    passing an object the constructor has no use for.

    **Natural variables, not overall composition.** Both formulations solve the
    same system; they differ in the unknowns. Natural variables carries
    saturations and phase compositions, so it can hold a cell exactly at a phase
    boundary, which is where a deck-initialised reservoir usually starts.
    Overall composition would have to flash such a cell every iteration to find
    out what is there. MRST picks natural variables for deck models, and so does
    this.
    """
    from pymrst.modules.compositional.natural_variables import (
        GenericNaturalVariablesModel,
    )
    from pymrst.modules.compositional.utils.init_deck_eos_model import (
        init_deck_eos_model,
    )

    eos = init_deck_eos_model(deck)
    mixture = getattr(eos, "CompositionalMixture", None)
    # `None` is defensive and, measured, unreachable through this path -
    # `init_deck_eos_model` either raises or builds a mixture. The *empty*
    # case is not: a deck with `CNAMES` present but empty builds an EOS with
    # zero components quite happily, and the model that comes out flashes an
    # empty component set forever without ever failing.
    if mixture is None or not len(getattr(mixture, "names", ())):
        raise ValueError(
            "the deck's EOS section produced no components; PROPS needs "
            "CNAMES, TCRIT, PCRIT, ACF and MW, and CNAMES must be non-empty")

    temperature = _deck_temperature(deck, eos)
    model = GenericNaturalVariablesModel(g, rock, mixture, temperature,
                                         wells, **kwargs)
    # Carried so the model can reach the EOS it was built from - the flash and
    # the property evaluations both need it, and rebuilding it from the deck
    # each time would be the same computation done twice.
    model.EOSModel = eos
    model.eos = eos
    model.water = bool(choice.get("water", False))
    return model


def _deck_temperature(deck, eos) -> float:
    """Reservoir temperature for the compositional model.

    A deck states it as `RTEMP` (a scalar) or `TEMPVD` (a depth table). The
    table's first temperature is used when only that is present - a
    depth-varying temperature needs a thermal model to mean anything, and
    silently averaging it would invent a number the deck does not contain.
    """
    props = deck.get("PROPS", deck.get("props", {})) or {}
    rs = _runspec(deck)
    for source in (props, rs):
        for key in ("RTEMP", "rtemp"):
            if key in source and source[key] is not None:
                import numpy as np
                return float(np.ravel(source[key])[0])
    for key in ("TEMPVD", "tempvd"):
        if key in props and props[key] is not None:
            import numpy as np
            table = np.atleast_2d(np.asarray(props[key], dtype=float))
            # (depth, temperature) pairs; take the shallowest entry's value.
            return float(table[0, -1])
    return float(getattr(eos, "reference_temperature", 273.15 + 15.0))


def select_model_from_deck(G, rock, fluid, deck: Dict[str, Any], **options):
    """MRST `selectModelFromDeck`: the AD-OO model for a deck's RUNSPEC."""
    options.pop("useLegacyModels", False)
    rs = _runspec(deck)
    hasgas, hasoil, haswat = _check(rs, "gas"), _check(rs, "oil"), _check(rs, "water")
    haspoly, hassurf, hascomp = _check(rs, "polymer"), _check(rs, "surfact"), _check(rs, "comps")
    if hascomp:
        assert not hassurf, "Compositional model does not support surfactant"
        assert not haspoly, "Compositional model does not support polymer"
        raise NotImplementedError(
            "GenericNaturalVariablesModel is not ported onto the AD-OO base; "
            "select_model_from_deck_ad builds PyMRST's compositional model.")
    if haspoly or hassurf:
        raise NotImplementedError(
            "The ad-eor models (GenericSurfactantPolymerModel and the legacy "
            "polymer/surfactant models) are not ported.")
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
        ThreePhaseBlackOilModel,
    )
    from pymrst.modules.ad_core.models.aquifers.select_aquifer_from_deck import (
        select_aquifer_from_deck,
    )
    from pymrst.modules.ad_core.models.facilities.select_facility_from_deck import (
        select_facility_from_deck,
    )
    model = GenericBlackOilModel(G, None, fluid, water=haswat, oil=hasoil,
                                 gas=hasgas, **options)
    if isinstance(model, ThreePhaseBlackOilModel) and rs:
        if _check(rs, "BLACKOIL"):
            model.disgas = _has_props(deck, "PVTO")
            model.vapoil = _has_props(deck, "PVTG")
        if "VAPOIL" in rs:
            model.vapoil = bool(rs["VAPOIL"])
        if "DISGAS" in rs:
            model.disgas = bool(rs["DISGAS"])
    model.inputdata = deck
    if rock is not None:
        model = model.setup_operators(G, rock)
    model.rock = rock
    model.FacilityModel = select_facility_from_deck(deck, model)
    model.AquiferModel = select_aquifer_from_deck(deck, model)
    model.minimumPressure = 0
    return model


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
selectModelFromDeck = select_model_from_deck

if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _inputs():
        g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 1.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[0.0, 1e-6 / BARSA], p_ref=200 * BARSA)
        return g, rock, fluid

    def test_three_phase_disgas_deck_routes_to_live_black_oil():
        """WATER+OIL+GAS with DISGAS and a PVTO table -> black-oil with disgas
        on, vapoil off (no PVTG). Checks the decision without a 3-phase fluid."""
        deck = {"RUNSPEC": {"WATER": True, "OIL": True, "GAS": True,
                            "DISGAS": True},
                "PROPS": {"PVTO": object()}}
        c = deck_model_choice(deck)
        assert c == {"kind": "blackoil", "disgas": True, "vapoil": False}, c
        print("ok test_three_phase_disgas_deck_routes_to_live_black_oil")

    def test_blackoil_run_reads_disgas_vapoil_from_pvt_tables():
        """Under a BLACKOIL run the tables decide: PVTO -> disgas, PVTG ->
        vapoil, before any explicit RUNSPEC override."""
        deck = {"RUNSPEC": {"WATER": True, "OIL": True, "GAS": True,
                            "BLACKOIL": True},
                "PROPS": {"PVTO": object(), "PVTG": object()}}
        c = deck_model_choice(deck)
        assert c == {"kind": "blackoil", "disgas": True, "vapoil": True}, c
        print("ok test_blackoil_run_reads_disgas_vapoil_from_pvt_tables")

    def test_dead_oil_water_deck_builds_two_phase_model():
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModelAD,
        )
        g, rock, fluid = _inputs()
        deck = {"RUNSPEC": {"WATER": True, "OIL": True}}
        m = select_model_from_deck_ad(g, rock, fluid, deck)
        assert isinstance(m, GenericBlackOilModelAD)
        assert m.disgas is False and m.vapoil is False
        assert m.input_data is deck and m.rock is rock
        print("ok test_dead_oil_water_deck_builds_two_phase_model")

    def test_water_only_deck_is_water_model():
        from pymrst.modules.ad_blackoil.models.water_model import WaterModelAD
        g, rock, fluid = _inputs()
        m = select_model_from_deck_ad(g, rock, fluid, {"RUNSPEC": {"WATER": True}})
        assert isinstance(m, WaterModelAD)
        print("ok test_water_only_deck_is_water_model")

    def test_compositional_is_routed_not_refused():
        """This used to assert COMPS was *refused*.

        The refusal's stated reason - that `initDeckEOSModel` and the
        `GenericNaturalVariablesModel` deck path were unported - had stopped
        being true; both were ported, and the message outlived them. A stale
        refusal is a stale comment that also breaks things.
        """
        choice = deck_model_choice(
            {"RUNSPEC": {"COMPS": 3, "OIL": True, "GAS": True, "WATER": True}})
        assert choice["kind"] == "compositional", choice
        assert choice["water"] is True
        print("ok test_compositional_is_routed_not_refused")

    def test_eor_is_still_refused():
        """`ad-eor` is a different MRST module, so this one stands."""
        for rs in ({"POLYMER": True, "WATER": True, "OIL": True},
                   {"SURFACT": True, "WATER": True, "OIL": True}):
            try:
                deck_model_choice({"RUNSPEC": rs})
            except NotImplementedError as refused:
                assert "ad-eor" in str(refused)
                continue
            raise AssertionError(f"expected refusal for {rs}")
        print("ok test_eor_is_still_refused")

    def test_the_port_builds_mrsts_generic_black_oil_model():
        deck = {"RUNSPEC": {"OIL": True, "GAS": True, "WATER": True, "DISGAS": True}}
        model = select_model_from_deck(None, None, {}, deck)
        assert type(model).__name__ == "GenericBlackOilModel"
        assert model.disgas and not model.vapoil
        assert (model.water, model.oil, model.gas) == (True, True, True)
        assert model.minimumPressure == 0 and model.inputdata is deck
        assert type(model.FacilityModel).__name__ == "GenericFacilityModel"
        print("ok test_the_port_builds_mrsts_generic_black_oil_model")

    def test_the_port_builds_a_water_only_deck_as_generic_black_oil_too():
        """MRST has no water-model branch: phases are switched off instead."""
        model = select_model_from_deck(None, None, {}, {"RUNSPEC": {"WATER": True}})
        assert type(model).__name__ == "GenericBlackOilModel"
        assert (model.water, model.oil, model.gas) == (True, False, False)
        print("ok test_the_port_builds_a_water_only_deck_as_generic_black_oil_too")

    def test_the_port_names_what_a_compositional_deck_needs():
        try:
            select_model_from_deck(None, None, {}, {"RUNSPEC": {"COMPS": 3, "OIL": True}})
        except NotImplementedError as refused:
            assert "GenericNaturalVariablesModel" in str(refused)
        else:
            raise AssertionError("expected NotImplementedError")
        print("ok test_the_port_names_what_a_compositional_deck_needs")

    test_the_port_builds_mrsts_generic_black_oil_model()
    test_the_port_builds_a_water_only_deck_as_generic_black_oil_too()
    test_the_port_names_what_a_compositional_deck_needs()
    test_three_phase_disgas_deck_routes_to_live_black_oil()
    test_blackoil_run_reads_disgas_vapoil_from_pvt_tables()
    test_dead_oil_water_deck_builds_two_phase_model()
    test_water_only_deck_is_water_model()
    test_compositional_is_routed_not_refused()
    test_eor_is_still_refused()
    print("\nAll tests passed!")
