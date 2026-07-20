"""Phase 42: equilibrium initialisation, against MATLAB.

Every initial state in this port was set by hand - a uniform pressure and a
uniform saturation, which is not a reservoir. A real one sits in gravity-
capillary equilibrium: the fluids have had geological time to sort themselves
by density. Start anywhere else and the first thing a simulation does is fall
towards this state, which is not the answer you asked for.

`dump_phase42_reference.m` runs MRST's `initStateBlackOilAD` twice - once with a
sharp water-oil contact and once with a capillary pressure that turns it into a
transition zone - and this compares both, field by field:

- the **phase pressures** (each phase follows its own hydrostatic gradient,
  anchored at the contact);
- the **saturations** (found by inverting the capillary curve; with no capillary
  pressure that inversion degenerates to a step, which is the sharp contact);
- and the **pressure the model carries**, which is the subtle one - see below.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import GenericBlackOilModel
from pymrst.modules.ad_core.initialization import (
    EquilibriumRegion,
    init_state_equil,
    initialize_equilibrium_pressures,
    initialize_equilibrium_saturations,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import NORM_GRAVITY, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

HERE = Path(__file__).parent
REF = HERE / "ref_p42_equil.mat"


def build(ref, use_pc: bool):
    nz = int(ref["nz"])
    top, thick = float(ref["top"]), float(ref["thick"])
    p_ref = float(ref["pRef"])
    pc0 = float(ref["pc0"])
    rho_w, rho_o = float(ref["rhoW"]), float(ref["rhoO"])

    g = cart_grid([1, 1, nz], [10.0, 10.0, thick])
    g.nodes.coords[:, 2] += top
    g = compute_geometry(g)
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)

    pc_ow = (lambda sw: pc0 * (1.0 - sw)) if use_pc else None
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[rho_w, rho_o],
        n=[2, 2], c=[0.0, 0.0], p_ref=p_ref, pc_ow=pc_ow,
    )
    model = GenericBlackOilModel(g, rock, fluid, [], gravity=NORM_GRAVITY)

    region = EquilibriumRegion(
        datum_depth=top, datum_pressure=p_ref, contacts=[float(ref["woc"])],
        pc_functions=([pc_ow, None] if use_pc else None),
        # MRST's BlackOilCapillaryPressure convention: the water phase carries
        # -pcOW, so its sign is negative.
        pc_sign=(np.array([-1.0, 1.0]) if use_pc else None),
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
        print("ref_p42_equil.mat missing (run dump_phase42_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)
    failed = 0

    for use_pc, tag, label in (
        (False, "no", "Sharp water-oil contact (no capillary pressure)"),
        (True, "pc", "With capillary pressure - a transition zone"),
    ):
        print(f"\n{label}:")
        model, region = build(ref, use_pc)

        p = initialize_equilibrium_pressures(model, region, NORM_GRAVITY)
        s = initialize_equilibrium_saturations(model, region, p)
        st = init_state_equil(model, region, NORM_GRAVITY)

        # each phase's own hydrostatic pressure
        failed += compare("phase pressures", p, ref[f"{tag}_pp"], 1e-9)
        # the saturations
        ml_s = np.atleast_2d(np.asarray(ref[f"{tag}_s"], float))
        failed += compare("saturations", s, ml_s, 1e-6, atol=1e-9)
        # and the pressure the model actually carries
        failed += compare("state pressure", st["pressure"], ref[f"{tag}_p"],
                          1e-9)

        n_zone = int(np.sum((s[:, 0] > 1e-6) & (s[:, 0] < 1 - 1e-6)))
        n_zone_ml = int(np.sum((ml_s[:, 0] > 1e-6) & (ml_s[:, 0] < 1 - 1e-6)))
        same = n_zone == n_zone_ml
        print(f"  {'ok  ' if same else 'FAIL'} transition zone: {n_zone} cells "
              f"(MATLAB: {n_zone_ml})")
        failed += 0 if same else 1

    print("\nEquilibrium initialisation matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
