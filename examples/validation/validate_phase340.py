"""Phase 340 (AC4): black-oil PVT flow properties, on MRST's class hierarchy.

The pre-graph PyMRST PVT (`ad_core.statefunctions.pvt_property_functions`)
folded the black-oil case into single monolithic statefunctions. That path is
MATLAB-validated - SPE 1 matched MRST through it (Phases 5b/24/47). AC4 splits
it onto MRST's own base/`BlackOil*` hierarchy in `statefunctions/flowprops/`.

The proof is internal equivalence against that validated path, which is a
tighter bound than re-comparing to MATLAB (round-off, not solver tolerance):
on the live-oil SPE 1 model, evaluated at a genuinely saturated + under-
saturated mixed state, the new statefunctions must reproduce the old ones to
machine precision, value and Jacobian:

    BlackOilShrinkageFactors == (legacy) ShrinkageFactors
    BlackOilViscosity        == (legacy) Viscosity
    BlackOilDensity          == (legacy) Density
    BlackOilPoreVolume       == (legacy) PoreVolume

plus two structural checks that need no legacy counterpart: SurfaceDensity
broadcasts rhoS, and DensityDerivedShrinkageFactors inverts BlackOilDensity.

Skips gracefully if the SPE 1 deck is not present next door.
"""

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    initial_state_from_deck,
)
from pymrst.modules.ad_core.statefunctions.flowprops import (
    BlackOilDensity,
    BlackOilPoreVolume,
    BlackOilShrinkageFactors,
    BlackOilViscosity,
    DensityDerivedShrinkageFactors,
    SurfaceDensity,
)
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    Density as LegacyDensity,
    PoreVolume as LegacyPoreVolume,
    ShrinkageFactors as LegacyShrinkageFactors,
    Viscosity as LegacyViscosity,
)
from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.adi import value

HERE = Path(__file__).parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def build_spe1_model():
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name in ("PRODUCER", "INJECTOR"):
        wd = deck["wells"][name]
        wells = add_well(wells, g, rock, wd["cells"], well_type=wd["type"],
                         val=wd["val"], radius=wd["radius"], compi=wd["compi"],
                         sign=wd["sign"], name=name, ref_depth=wd["ref_depth"],
                         gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
    model = GenericBlackOilModel(g, rock, fluid, wells, disgas=True,
                                 gravity=NORM_GRAVITY)
    st = initial_state_from_deck(model, deck)
    # push part of the field over the bubble point so both the saturated and
    # under-saturated PVT branches are exercised in one state
    st = dict(st)
    nc = g.cells.num
    st["sG"] = np.where(np.arange(nc) % 2 == 0, 0.15, 0.0)
    st["status"] = np.where(np.arange(nc) % 2 == 0, 2, 1)
    from pymrst.modules.ad_core.statefunctions.state_function import State
    return model, model.init_state_ad(State(st))


def _worst(a, b):
    v = max(np.max(np.abs(value(a[i]) - value(b[i]))) for i in range(len(a)))
    j = 0.0
    for i in range(len(a)):
        if hasattr(a[i], "jac") and hasattr(b[i], "jac"):
            for ba, bb in zip(a[i].jac, b[i].jac):
                j = max(j, np.max(np.abs((ba - bb).toarray())))
    return v, j


def check_equivalence(model, st):
    P = model.PVTPropertyFunctions
    pairs = [
        ("ShrinkageFactors", BlackOilShrinkageFactors, LegacyShrinkageFactors),
        ("Viscosity", BlackOilViscosity, LegacyViscosity),
    ]
    for name, new_cls, old_cls in pairs:
        P.set_state_function(name, old_cls(model))
        old = model.get_props(st, name)
        st.clear_cache()
        P.set_state_function(name, new_cls(model))
        new = model.get_props(st, name)
        st.clear_cache()
        v, j = _worst(new, old)
        print(f"  {new_cls.__name__:28s} vs legacy: val {v:.2e}, jac {j:.2e}")
        assert v < 1e-12 and j < 1e-9

    # Density needs SurfaceDensity present; compare against legacy Density
    P.set_state_function("ShrinkageFactors", BlackOilShrinkageFactors(model))
    P.set_state_function("SurfaceDensity", SurfaceDensity(model))
    P.set_state_function("Density", LegacyDensity(model))
    old = model.get_props(st, "Density")
    st.clear_cache()
    P.set_state_function("Density", BlackOilDensity(model))
    new = model.get_props(st, "Density")
    st.clear_cache()
    v, j = _worst(new, old)
    print(f"  {'BlackOilDensity':28s} vs legacy: val {v:.2e}, jac {j:.2e}")
    assert v < 1e-9 and j < 1e-6

    # PoreVolume
    P.set_state_function("PoreVolume", LegacyPoreVolume(model))
    old = [model.get_props(st, "PoreVolume")]
    st.clear_cache()
    P.set_state_function("PoreVolume", BlackOilPoreVolume(model))
    new = [model.get_props(st, "PoreVolume")]
    st.clear_cache()
    v, j = _worst(new, old)
    print(f"  {'BlackOilPoreVolume':28s} vs legacy: val {v:.2e}, jac {j:.2e}")
    assert v < 1e-9 and j < 1e-6


def check_structural(model, st):
    P = model.PVTPropertyFunctions
    P.set_state_function("ShrinkageFactors", BlackOilShrinkageFactors(model))
    P.set_state_function("SurfaceDensity", SurfaceDensity(model))
    P.set_state_function("Density", BlackOilDensity(model))
    P.set_state_function("DDShrink", DensityDerivedShrinkageFactors(model))
    rho = model.get_props(st, "Density")
    rhoS = model.get_props(st, "SurfaceDensity")
    b = model.get_props(st, "ShrinkageFactors")
    b_der = model.get_props(st, "DDShrink")
    # DDShrink is rho/rhoS by definition (an invariant, no legacy counterpart)
    defn = max(np.max(np.abs(value(b_der[i]) - value(rho[i]) / value(rhoS[i])))
               for i in range(model.n_phases))
    print(f"  DensityDerivedShrinkageFactors == Density/rhoS: {defn:.2e}")
    assert defn < 1e-12
    # water (phase 0) carries no dissolved/vaporised mass, so there it also
    # recovers the direct shrinkage factor b_W exactly
    w = np.max(np.abs(value(b_der[0]) - value(b[0])))
    print(f"  ... and recovers b for the carry-free water phase: {w:.2e}")
    assert w < 1e-9


def main():
    print("Phase 340 (AC4): black-oil PVT flow properties")
    if not DECK.exists():
        print("  (skipped: SPE 1 deck not present)")
        print("Phase 340 skipped")
        return
    model, st = build_spe1_model()
    check_equivalence(model, st)
    check_structural(model, st)
    print("Phase 340 passed - the MRST-shaped black-oil PVT hierarchy "
          "reproduces the validated PVT to machine precision.")


if __name__ == "__main__":
    main()
