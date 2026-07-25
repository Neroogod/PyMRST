"""Phase 354: live-oil equilibration density, against MATLAB ad-core (AC17).

Phase 42 validated equilibrium initialisation with an *analytic, dead-oil*
fluid. This does the case that fluid can never reach: a **live-oil** column,
where the oil shrinkage factor depends on the dissolved gas `rs`, and the
hydrostatic density that carries the column carries its gas with it -
`rho_o = b_o(p, rs) (rhoS_o + rs rhoS_g)`.

That path was silently broken: with no `rs` supplied, `b_o` fell into its
undersaturated branch, returned NaN, and sent the ODE integrator into an
unbounded step-halving loop (a 300-cell case took over five minutes and never
finished). Giving the equilibration MRST's own density formula
(`getInitializationRegionsBlackOil` / `initStateBlackOilAD`) fixed both the
correctness and the hang, and this pins that formula against MRST.

The reference is a **constant** `rs = rsSat(p_datum)`, which - because saturated
`rsSat` rises with pressure - leaves the oil column *undersaturated* below the
datum. That is the exact branch that used to NaN, so it is the one worth
checking: `rs = min(rs_given, rsSat)` in the oil, and the density that follows
from it. MRST's fluid is SPE 1's real PVTO/PVDG tables; only the initialisation
is replaced by a hydrostatic region.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.initialization.equilibrium import (
    EquilibriumRegion,
    init_state_equil,
)
from pymrst.modules.ad_props.blackoil_fluid import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, make_rock
from pymrst.utils.units import BARSA, DARCY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p354_equil_disgas.mat"
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def build(ref):
    nz = int(ref["nz"])
    top, thick = float(ref["top"]), float(ref["thick"])

    g = cart_grid([1, 1, nz], [10.0, 10.0, thick])
    g.nodes.coords[:, 2] += top
    g = compute_geometry(g)
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)

    deck = read_spe1_deck(DECK)
    fluid = init_deck_blackoil_fluid(deck)
    model = GenericBlackOilModel(g, rock, fluid, [], disgas=True,
                                 gravity=NORM_GRAVITY)

    rs_const = float(ref["rs_const"])
    region = EquilibriumRegion(
        datum_depth=float(ref["datum_depth"]),
        datum_pressure=float(ref["datum_pressure"]),
        contacts=[float(ref["goc"]), float(ref["woc"])],  # shallowest-first
        rs=rs_const,                                       # constant composition
    )
    return model, region


def compare(name, py, ml, rtol, atol=0.0) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale + atol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p354_equil_disgas.mat missing (run dump_phase354_reference.m)")
        return 1
    if not DECK.exists():
        print("SPE1 deck not found; skipping")
        return 0
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    state = ref["state"]                     # struct: pressure, s, rs
    ml_p = np.asarray(state["pressure"].item(), float).ravel()
    ml_s = np.atleast_2d(np.asarray(state["s"].item(), float))
    ml_rs = np.asarray(state["rs"].item(), float).ravel()

    model, region = build(ref)
    st = init_state_equil(model, region, NORM_GRAVITY)

    failed = 0
    print("Live-oil hydrostatic column (SPE 1 PVT), constant rs:")
    failed += compare("pressure", st["pressure"], ml_p, 1e-8)
    failed += compare("sW", st["sW"], ml_s[:, 0], 1e-6, atol=1e-9)
    failed += compare("sO", st["sO"], ml_s[:, 1], 1e-6, atol=1e-9)
    failed += compare("sG", st["sG"], ml_s[:, 2], 1e-6, atol=1e-9)
    failed += compare("rs", st["rs"], ml_rs, 1e-8)

    # The point of the whole phase: the stored rs is the *specified*, capped
    # composition - not the saturated value everywhere. Had the code reverted to
    # "saturated everywhere", rs would read rsSat and both rs and (through the
    # density) the pressure would move. Confirm the undersaturated branch is
    # genuinely exercised (rs strictly below rsSat somewhere), so a pass is not
    # a fully-saturated column trivially agreeing.
    rs_max = np.asarray(model.raw_fluid.rs_sat(ml_p), float).ravel()
    undersat = int(np.sum(ml_rs < rs_max - 1e-6 * rs_max.max()))
    print(f"  {'ok  ' if undersat > 0 else 'FAIL'} the undersaturated branch is "
          f"exercised: {undersat} cells carry rs < rsSat "
          f"(rs {ml_rs.min():.2f}-{ml_rs.max():.2f}, rsSat to {rs_max.max():.2f})")
    failed += 0 if undersat > 0 else 1

    print("\nLive-oil equilibration matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
