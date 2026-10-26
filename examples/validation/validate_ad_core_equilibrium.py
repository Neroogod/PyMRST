"""Validate MRST's equilibration functions in `ad-core/initialization`:
`getInitializationRegionsBase`, `initializeEquilibriumPressures`,
`initializeEquilibriumSaturations` and `getMinMaxPhaseSaturations`.

The reference is MRST 2024a (`dump_ad_core_equilibrium_reference.m`) on a
60-cell column with gravity:
- the defaults of an initialization region;
- three phases with pressure-dependent oil and gas densities, the datum
  between the contacts, capillary transition zones with an offset capillary
  pressure at the water contact, and saturation limits - pressures from
  MRST's `ode45` integration (ported in `pymrst.utils.ode45`) and the
  saturations inverted from the capillary curves;
- overlapping transition zones, where oil would go negative and the gas-water
  curve takes over (25 cells);
- two phases with a sharp contact;
- saturation limits from the relative permeabilities, without and with the
  fluid's connate water.
Indices in the region are 0-based (MRST's `reference_index` less one).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _close(name: str, a, b, tol: float = RTOL) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float).reshape(np.shape(a))
    err = float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))
    return report(err < tol, f"{name} rel. diff {err:.1e}")


def main() -> int:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.initialization.get_initialization_regions_base import (
        get_initialization_regions_base,
    )
    from pymrst.modules.ad_core.initialization.get_min_max_phase_saturations import (
        get_min_max_phase_saturations,
    )
    from pymrst.modules.ad_core.initialization.initialize_equilibrium_pressures import (
        initialize_equilibrium_pressures,
    )
    from pymrst.modules.ad_core.initialization.initialize_equilibrium_saturations import (
        initialize_equilibrium_saturations,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.params.gravity.gravity import gravity, isolated_gravity
    from pymrst.utils.units import ATM, BARSA, CENTI, POISE

    ref = loadmat(HERE/"ref_ad_core_equilibrium.mat", squeeze_me=True, struct_as_record=False)
    with isolated_gravity():
        gravity("reset", "on")
        G = compute_geometry(cart_grid([1, 1, 60], [1.0, 1.0, 60.0]))
        rock = make_rock(G, 1e-13, 0.2)
        fluid = init_simple_adi_fluid(phases="WOG", mu=np.array([1, 5, 0.2])*CENTI*POISE,
                                      rho=[1000, 800, 150], n=[2, 2, 2])
        model = GenericBlackOilModel(G, rock, fluid)
        rho = [lambda p, z: 1000 + 0*p, lambda p, z: 800*np.exp(1e-9*(p - 2e7)),
               lambda p, z: 150*(p/2e7)]
        pc = [lambda s: 1e5*(1 - s)**2, None, lambda s: 5e4*s]

        base = get_initialization_regions_base(model, rho, [40.0, 15.0])
        d = ref["defaults"]
        ok = report(base.datum_pressure == d.datum_pressure == ATM and base.datum_depth == d.datum_depth
                    and base.reference_index + 1 == d.reference_index and base.cells.size == d.ncells
                    and all(np.array_equal(np.ravel(getattr(base, k)), np.ravel(getattr(d, k)))
                            for k in ("pc_sign", "pc_scale", "contacts_pc", "s_max", "s_min")),
                    "getInitializationRegionsBase: defaults as MRST's")

        region = get_initialization_regions_base(
            model, rho, [40.0, 15.0], datum_pressure=200*BARSA, datum_depth=30.0, pc_functions=pc,
            pc_sign=np.array([1.0, 1.0, -1.0]), contacts_pc=np.array([1e4, 0.0]),
            s_min=np.array([0.1, 0.0, 0.0]), s_max=np.array([1.0, 1.0, 0.9]), pc_scale=np.ones((60, 3)))
        p1 = initialize_equilibrium_pressures(model, region)
        ok &= _close("three phases: pressures", p1, ref["p1"], 1e-13)
        s1 = initialize_equilibrium_saturations(model, region, p1)
        ok &= _close("three phases: saturations", s1, ref["s1"])

        region.contacts = np.array([30.0, 29.0])
        region.pc_functions = [lambda s: 4e5*(1 - s)**2, None, lambda s: 3e5*s]
        p2 = initialize_equilibrium_pressures(model, region)
        ok &= _close("overlapping transition zones: pressures", p2, ref["p2"], 1e-13)
        s2 = initialize_equilibrium_saturations(model, region, p2)
        ok &= _close("overlapping transition zones: saturations (gas-water fallback)", s2, ref["s2"])

        fluid2 = init_simple_adi_fluid(phases="WO", mu=np.array([1, 5])*CENTI*POISE, rho=[1000, 800],
                                       n=[2, 2])
        model2 = GenericBlackOilModel(G, rock, fluid2, gas=False)
        region2 = get_initialization_regions_base(model2, rho[:2], [35.0], datum_pressure=150*BARSA,
                                                  pc_functions=[lambda s: 0*s, None])
        p3 = initialize_equilibrium_pressures(model2, region2)
        ok &= _close("two phases, sharp contact: pressures", p3, ref["p3"], 1e-13)
        ok &= _close("two phases, sharp contact: saturations",
                     initialize_equilibrium_saturations(model2, region2, p3), ref["s3"])

        smin, smax = get_min_max_phase_saturations(model, np.array([1, 1, 1]), np.array([0, 1, 2]))
        ok &= _close("getMinMaxPhaseSaturations from the relperms: s_min", smin, ref["smin1"])
        ok &= _close("getMinMaxPhaseSaturations from the relperms: s_max", smax, ref["smax1"])
        fluid.sWcon = 0.15
        modelc = GenericBlackOilModel(G, rock, fluid)
        smin, smax = get_min_max_phase_saturations(modelc, np.array([1, 1]), np.array([0, 1]))
        ok &= _close("with connate water: s_min", smin, ref["smin2"])
        ok &= _close("with connate water: s_max", smax, ref["smax2"])
    print("\nThe equilibration functions match MRST" if ok
          else "\nThe equilibration functions do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
