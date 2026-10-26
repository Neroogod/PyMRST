"""Phase 359: selectModelFromDeck routing on SPE1, against MRST (AC23).

AC23 ports `selectModelFromDeck`, the factory that turns a parsed deck's RUNSPEC
flags into the right model class and its disgas/vapoil settings.
`dump_phase359_reference.m` records the decision MRST makes on the real SPE1
deck - the model class, and the resolved disgas/vapoil - together with the
RUNSPEC flags and PVT-table presence that drove it. Here PyMRST rebuilds a deck
dict from those same flags and must reach the identical decision.

SPE1 is the sharp case for the override path: its BLACKOIL flag is off, so
disgas is *not* read from the PVTO table but from the explicit RUNSPEC DISGAS -
which is exactly the branch this pins.
"""

from pathlib import Path

from scipy.io import loadmat

from pymrst.modules.ad_core.utils.select_model_from_deck import (
    deck_model_choice,
    select_model_from_deck,
)

REF = Path(__file__).with_name("ref_p359_selectmodel.mat")

CLASS_TO_KIND = {"GenericBlackOilModel": "blackoil", "WaterModel": "water"}


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase359_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    def flag(name: str) -> bool:
        return bool(int(ref[name].ravel()[0]))

    mrst_class = str(ref["model_class"][0]).strip()
    mrst_disgas = flag("disgas")
    mrst_vapoil = flag("vapoil")

    # rebuild a deck dict from exactly the flags MRST reported
    runspec = {}
    if flag("flag_water"):
        runspec["WATER"] = True
    if flag("flag_oil"):
        runspec["OIL"] = True
    if flag("flag_gas"):
        runspec["GAS"] = True
    if flag("flag_blackoil"):
        runspec["BLACKOIL"] = True
    if flag("flag_disgas_rs"):
        runspec["DISGAS"] = True
    if flag("flag_vapoil_rs"):
        runspec["VAPOIL"] = True
    props = {}
    if flag("has_pvto"):
        props["PVTO"] = object()
    if flag("has_pvtg"):
        props["PVTG"] = object()
    deck = {"RUNSPEC": runspec, "PROPS": props}

    choice = deck_model_choice(deck)
    failed = 0
    print(f"selectModelFromDeck on SPE1 (MRST: {mrst_class}, "
          f"disgas={int(mrst_disgas)}, vapoil={int(mrst_vapoil)}):")
    failed += _check(f"model class matches ({choice['kind']})",
                     choice["kind"] == CLASS_TO_KIND.get(mrst_class))
    failed += _check(f"disgas matches ({choice['disgas']})",
                     choice["disgas"] == mrst_disgas)
    failed += _check(f"vapoil matches ({choice['vapoil']})",
                     choice["vapoil"] == mrst_vapoil)

    model = select_model_from_deck(None, None, {}, deck)
    print(f"select_model_from_deck (the port) builds {type(model).__name__}:")
    failed += _check("class is MRST's", type(model).__name__ == mrst_class)
    failed += _check(f"disgas matches ({model.disgas})", model.disgas == mrst_disgas)
    failed += _check(f"vapoil matches ({model.vapoil})", model.vapoil == mrst_vapoil)
    failed += _check("minimumPressure is 0", model.minimumPressure == 0)
    failed += _check("facility is GenericFacilityModel",
                     type(model.FacilityModel).__name__ == "GenericFacilityModel")

    print("\nModel selection matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
