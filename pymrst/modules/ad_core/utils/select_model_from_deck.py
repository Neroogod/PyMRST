"""Pick the simulation model class for a parsed deck (MRST `selectModelFromDeck`).

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
        raise NotImplementedError(
            "compositional deck selection needs initDeckEOSModel and the "
            "GenericNaturalVariablesModel deck path, which are not ported"
        )
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


def select_model_from_deck(g, rock, fluid, deck: Dict[str, Any], **kwargs):
    """Return the model matching the deck's RUNSPEC flags, built and ready.

    `g`, `rock`, `fluid` are the usual simulation inputs; `deck` is a parsed deck
    with a `RUNSPEC` section (and `PROPS` for the PVT tables). Extra keyword
    arguments are forwarded to the model constructor.
    """
    choice = deck_model_choice(deck)
    wells = kwargs.pop("wells", ())
    if choice["kind"] == "water":
        from pymrst.modules.ad_blackoil.models.water_model import WaterModel

        model = WaterModel(g, rock, fluid, wells, **kwargs)
    else:
        from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
            GenericBlackOilModel,
        )

        model = GenericBlackOilModel(g, rock, fluid, wells,
                                     disgas=choice["disgas"],
                                     vapoil=choice["vapoil"], **kwargs)
    model.input_data = deck
    model.rock = rock
    return model


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
            GenericBlackOilModel,
        )
        g, rock, fluid = _inputs()
        deck = {"RUNSPEC": {"WATER": True, "OIL": True}}
        m = select_model_from_deck(g, rock, fluid, deck)
        assert isinstance(m, GenericBlackOilModel)
        assert m.disgas is False and m.vapoil is False
        assert m.input_data is deck and m.rock is rock
        print("ok test_dead_oil_water_deck_builds_two_phase_model")

    def test_water_only_deck_is_water_model():
        from pymrst.modules.ad_blackoil.models.water_model import WaterModel
        g, rock, fluid = _inputs()
        m = select_model_from_deck(g, rock, fluid, {"RUNSPEC": {"WATER": True}})
        assert isinstance(m, WaterModel)
        print("ok test_water_only_deck_is_water_model")

    def test_compositional_and_eor_are_refused():
        for rs in ({"COMPS": 3, "OIL": True, "GAS": True},
                   {"POLYMER": True, "WATER": True, "OIL": True},
                   {"SURFACT": True, "WATER": True, "OIL": True}):
            try:
                deck_model_choice({"RUNSPEC": rs})
            except NotImplementedError:
                continue
            raise AssertionError(f"expected refusal for {rs}")
        print("ok test_compositional_and_eor_are_refused")

    test_three_phase_disgas_deck_routes_to_live_black_oil()
    test_blackoil_run_reads_disgas_vapoil_from_pvt_tables()
    test_dead_oil_water_deck_builds_two_phase_model()
    test_water_only_deck_is_water_model()
    test_compositional_and_eor_are_refused()
    print("\nAll tests passed!")
