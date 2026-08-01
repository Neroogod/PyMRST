"""Phase 47: vaporised oil (`rv`), against MATLAB ad-blackoil.

Black-oil has two dissolutions and this port only ever had one. `rs` is gas
dissolved in oil; `rv` is oil **vaporised into gas** - and without it a gas cap
is dry, a gas-condensate reservoir cannot be modelled at all, and gas produced
from a volatile oil arrives at the surface carrying no liquid. It is not an
exotic option: it is half of the black-oil model.

What it costs is a third state per cell. The hydrocarbon primary variable `x`
already meant two things (`rs` where the oil is undersaturated, `sG` where both
phases exist); with vaporised oil it means a third - `rv`, where the *gas* is
undersaturated because there is no free oil at all. MRST encodes this as
`status = oil_present + 2*gas_present`:

- **1** - oil, no free gas. `x` is `rs`.
- **2** - gas, no free oil. `x` is `rv`.   <- the state this port never had
- **3** - both. `x` is `sG`.

`dump_phase47_reference.m` runs a gas-condensate column - a gas cap with **no
oil phase at all** over an undersaturated oil leg - through MRST, twice:

- **steady** (a gentle 230 bar producer): every cell stays where it started, so
  status 2 and status 1 are exercised steadily, for every step;
- **depletion** (160 bar, shorter steps): cells cross into status 3 as gas
  breaks out of solution - so the *transitions* are exercised, which is where a
  variable switch actually goes wrong.

A model that got the switch right in one and wrong in the other would pass
either test on its own, which is why both are here.

Both cases are chosen so that **MRST takes exactly one ministep per report
step** (the dump asserts it). If MRST had to sub-divide, it would be solving a
different discrete problem from a single-step solve, and matching it would mean
reproducing its timestep controller rather than its physics.

The PVT is **analytic, not tabulated**, deliberately: both codes evaluate the
same closed-form functions, so a difference is the physics and not the
interpolation. `bO`/`muO` take `(p, rs, gas_present)` and `bG`/`muG` take
`(p, rv, oil_present)` - two *different* flags, which is the easiest thing to
get backwards here and would be nearly invisible, because in a saturated cell
both are true.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    blackoil_status,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props.props.init_deck_adi_fluid import BlackOilFluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import ad_max
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p47_vapoil.mat"

# the constants of dump_phase47_reference.m, verbatim
P_REF = 300 * BARSA
RHO_S = np.array([1000.0, 800.0, 1.2])
C_W, MU_W0 = 4.0e-5 / BARSA, 0.4 * CENTI * POISE
C_O, MU_O0 = 1.2e-4 / BARSA, 2.0 * CENTI * POISE
C_G, MU_G0 = 3.0e-3 / BARSA, 0.02 * CENTI * POISE
RS0, RV0 = 150.0, 8.0e-4
GAM_O, GAM_G = 1.0e-3, 50.0
D_O, D_G = 2.0e-3, 100.0
SWCON = 0.1


def _blend(flag, saturated, undersaturated):
    """`flag*saturated + ~flag*undersaturated`, AD-transparent.

    The saturation flag picks which curve the property is on. Written as a
    weighted sum rather than a `where` because either branch may be an AD
    number, and `np.where` would strip the derivatives.
    """
    f = np.asarray(flag, dtype=float)
    return f * saturated + (1.0 - f) * undersaturated


def build_fluid() -> BlackOilFluid:
    rs_sat = lambda p: RS0 * (p / P_REF)
    rv_sat = lambda p: RV0 * (p / P_REF)

    def b_o(p, rs, flag):
        return (1.0 + C_O * (p - P_REF)) / (
            1.0 + GAM_O * _blend(flag, rs_sat(p), rs))

    def mu_o(p, rs, flag):
        return MU_O0 * (1.0 + 5.0e-4 / BARSA * (p - P_REF)) / (
            1.0 + D_O * _blend(flag, rs_sat(p), rs))

    def b_g(p, rv, flag):
        return (1.0 + C_G * (p - P_REF)) * (
            1.0 + GAM_G * _blend(flag, rv_sat(p), rv))

    def mu_g(p, rv, flag):
        return MU_G0 * (1.0 + 2.0e-4 / BARSA * (p - P_REF)) * (
            1.0 + D_G * _blend(flag, rv_sat(p), rv))

    return BlackOilFluid(
        rhoS=RHO_S,
        b_w=lambda p: 1.0 + C_W * (p - P_REF),
        mu_w=lambda p: MU_W0 + 0.0 * p,
        b_o=b_o, mu_o=mu_o, rs_sat=rs_sat,
        b_g=b_g, mu_g=mu_g, rv_sat=rv_sat, vapoil=True,
        # `ad_max`, not `np.maximum`: the saturations are AD numbers here and
        # numpy's ufunc would silently strip their derivatives.
        kr_w=lambda sw: ad_max((sw - SWCON) / (1.0 - SWCON), 0.0) ** 2,
        kr_g=lambda sg: ad_max(sg, 0.0) ** 2,
        kr_ow=lambda so: ad_max(so, 0.0) ** 2,
        kr_og=lambda so: ad_max(so, 0.0) ** 2,
        pv_mult=lambda p: 1.0 + 1.0e-5 / BARSA * (p - P_REF),
        swcon=SWCON,
    )


def build(ref, bhp_val: float):
    nz, top, thick = int(ref["nz"]), float(ref["top"]), float(ref["thick"])
    ngc = int(ref["ngc"])

    g = cart_grid([1, 1, nz], [10.0, 10.0, thick])
    g.nodes.coords[:, 2] += top
    g = compute_geometry(g)
    nc = g.cells.num
    # 1 mD, matching the dump. At 100 mD this column drains to the bottom-hole
    # pressure in minutes, so a 15-day step resolves no transient at all and
    # the comparison tests nothing but the equilibrium.
    rock = make_rock(g, 1 * MILLI * DARCY, 0.25)
    fluid = build_fluid()

    # MATLAB's `floor(nz/2)` is a *1-based* cell index, so it is one cell
    # shallower than the same arithmetic in Python. Getting this wrong put the
    # well in the neighbouring cell and produced a 4 bar disagreement that
    # looked exactly like a physics bug.
    w = add_well(None, g, rock, np.array([nz // 2 - 1]), well_type="bhp",
                 val=bhp_val, compi=np.array([0.0, 1.0, 0.0]), sign=-1,
                 radius=0.1, name="PROD")

    # **MRST's own tolerances** (`toleranceCNV = 1e-3`, `toleranceMB = 1e-7`),
    # not PyMRST's much tighter defaults - and for a reason worth knowing.
    #
    # MRST's flash will only let gas nucleate once `rs > rsSat*(1 + sqrt(eps))`.
    # That dead-band is there to stop a cell chattering in and out of the
    # two-phase state on round-off. But it also means a cell sitting *exactly*
    # on its bubble point - which is what a depleting oil leg produces, in
    # quantity - can be asked by Newton for a gas increment smaller than the
    # dead-band, have it clipped straight back, and stall. The residual parks
    # at ~1e-7 and will not go lower however long you iterate.
    #
    # MRST never sees this because it stops at 1e-3. PyMRST, asking for 1e-8,
    # walks into it. The tolerance is the thing that is wrong here, not the
    # physics: 1e-8 is not a tolerance this discretisation can always reach.
    model = GenericBlackOilModel(g, rock, fluid, w, disgas=True, vapoil=True,
                                 gravity=0.0, tol_cnv=1e-3, tol_mb=1e-7,
                                 tol_well=1e-8)

    p0 = np.full(nc, P_REF)
    cap = np.arange(nc) < ngc
    sw = np.full(nc, SWCON)
    so = np.where(cap, 0.0, 1.0 - SWCON)
    sg = np.where(cap, 1.0 - SWCON, 0.0)
    rs = np.where(cap, 0.0, 0.55 * fluid.rs_sat(p0))
    rv = np.where(cap, 0.55 * fluid.rv_sat(p0), 0.0)

    st = State(
        pressure=p0, sW=sw, sO=so, sG=sg, rs=rs, rv=rv,
        status=blackoil_status(sw, so, sg, disgas=True, vapoil=True),
        qWs=np.zeros(1), qOs=np.zeros(1), qGs=np.zeros(1),
        bhp=np.array([bhp_val]),
    )
    return model, st


def run(ref, bhp_val: float, nstep: int, dt: float):
    model, st = build(ref, bhp_val)
    solver = NonLinearSolver(max_iterations=25)
    hist = {k: [] for k in ("p", "sw", "so", "sg", "rs", "rv", "st",
                            "qWs", "qOs", "qGs", "bhp")}
    for i in range(nstep):
        st, rep = solver.solve_ministep(model, st, st.copy(), dt)
        assert rep.converged, f"step {i} did not converge"
        for key, val in (("p", st["pressure"]), ("sw", st["sW"]),
                         ("so", st["sO"]), ("sg", st["sG"]), ("rs", st["rs"]),
                         ("rv", st["rv"]), ("st", st["status"]),
                         ("qWs", st["qWs"]), ("qOs", st["qOs"]),
                         ("qGs", st["qGs"]), ("bhp", st["bhp"])):
            hist[key].append(np.asarray(val, float).copy())
    return model, {k: np.vstack(v) for k, v in hist.items()}


def compare(name, py, ml, rtol) -> int:
    py = np.asarray(py, float).ravel()
    ml = np.asarray(ml, float).ravel()
    scale = max(np.abs(ml).max(), 1e-300)
    diff = float(np.abs(py - ml).max())
    ok = diff <= rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p47_vapoil.mat missing (run dump_phase47_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    for tag, bhp_val, label in (
        ("steady", 230 * BARSA,
         "Steady - the cells stay undersaturated (status 2 in the gas cap, "
         "1 in the oil leg)"),
        ("depletion", 160 * BARSA,
         "Depletion - gas breaks out of solution (the status transitions)"),
    ):
        nstep = int(ref[f"{tag}_nstep"])
        dt = float(ref[f"{tag}_dt"])
        print(f"\n{label}:")
        model, py = run(ref, bhp_val, nstep, dt)

        failed += compare("WI", model.wells[0].WI, ref[f"{tag}_WI"], 1e-12)
        for key, rtol in (("p", 2e-6), ("sw", 2e-6), ("so", 2e-6),
                          ("sg", 2e-6), ("rs", 2e-6), ("rv", 2e-6),
                          ("qOs", 2e-6), ("qGs", 2e-6), ("bhp", 2e-6)):
            failed += compare(f"{key} history", py[key], ref[f"{tag}_{key}"],
                              rtol)
        # qWs is not compared as a ratio: the water is immobile at connate
        # saturation, so MRST's is ~1e-19 and dividing by it would report a
        # meaningless "relative error" of 1e7 for a difference of 1e-12.
        wat = float(np.abs(py["qWs"]).max())
        ok = wat < 1e-9
        print(f"  {'ok  ' if ok else 'FAIL'} qWs history: immobile water, "
              f"|qWs| <= {wat:.1e} (MATLAB: "
              f"{float(np.abs(ref[f'{tag}_qWs']).max()):.1e})")
        failed += 0 if ok else 1

        # The status is an *integer* decision - which variable each cell is
        # solving for - so "close" is meaningless. It matches exactly or the
        # two codes are solving different problems.
        ml_st = np.atleast_2d(np.asarray(ref[f"{tag}_st"], float))
        same = np.array_equal(py["st"].astype(int), ml_st.astype(int))
        n_wrong = int((py["st"].astype(int) != ml_st.astype(int)).sum())
        print(f"  {'ok  ' if same else 'FAIL'} status history: "
              f"{'identical' if same else f'{n_wrong} cells differ'} "
              f"(states present: {sorted(set(ml_st.astype(int).ravel()))})")
        failed += 0 if same else 1

    # And the point of the whole phase: `rv` must not be a decoration. If the
    # gas carried no oil at all, every number above would still be *some*
    # answer - just not this one - so check the mechanism actually fired.
    ml_rv = np.asarray(ref["depletion_rv"], float)
    span = float(ml_rv.max() - ml_rv.min())
    _, dep = run(ref, 160 * BARSA, int(ref["depletion_nstep"]),
                 float(ref["depletion_dt"]))
    py_span = float(dep["rv"].max() - dep["rv"].min())
    moved = span > 1e-5 and abs(py_span - span) < 1e-3 * span
    print(f"\n  {'ok  ' if moved else 'FAIL'} vaporised oil is doing something: "
          f"rv spans {py_span:.3e} across the field (MATLAB: {span:.3e})")
    failed += 0 if moved else 1

    print("\nVaporised oil matches MATLAB - black-oil is complete!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
