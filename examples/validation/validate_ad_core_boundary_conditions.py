"""Validate MRST's boundary-condition and source utilities: the ports of
`getBoundaryConditionFluxesAD`, `getSourceFluxesAD`,
`computeSourcesAndBoundaryConditionsAD` and `addFluxesFromSourcesAndBC` in
`ad_core/utils/` (one file each).

The reference is MRST 2024a (`dump_ad_core_boundary_conditions_reference.m`)
on AD inputs - pressure and water saturation as primary variables of a
4x3x2 grid with gravity, mobilities and densities as closed-form functions of
them - so every flux is compared with both its Jacobian blocks:
- eight boundary faces: pressure inflow and outflow, flux injection and
  production, reservoir-rate faces (sequential upwinding), all-zero
  saturations; then again with outside densities and mobilities;
- sources injecting and producing, and injecting only (MRST keeps those
  rates plain numbers);
- the contribution structs for oil-water, and for three phases with gas
  dissolved in oil and an injection dissolution ratio;
- the equations after `addFluxesFromSourcesAndBC`.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RTOL = 1e-12


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _dense(block):
    block = block.to_sparse() if hasattr(block, "to_sparse") else block
    return block.toarray() if hasattr(block, "toarray") else np.asarray(block)


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float).reshape(np.shape(a))
    scale = max(np.abs(b).max() if b.size else 0.0, 1e-300)
    return float(np.abs(a - b).max()/scale) if b.size else 0.0


def _ad(tag: str, q, ref) -> bool:
    from pymrst.utils.adi import value
    err = max(_rel(np.ravel(value(q)), np.ravel(ref.val)),
              _rel(_dense(q.jac[0]), ref.jp), _rel(_dense(q.jac[1]), ref.js))
    return report(err < RTOL, f"{tag}: value and Jacobians rel. diff {err:.1e}")


def _phases(tag: str, qs, refs) -> bool:
    ok = True
    for i, (q, r) in enumerate(zip(qs, np.atleast_1d(refs))):
        ok &= _ad(f"{tag}, phase {i + 1}", q, r)
    return ok


def main() -> int:
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
    from pymrst.modules.ad_core.utils.add_fluxes_from_sources_and_bc import (
        add_fluxes_from_sources_and_bc,
    )
    from pymrst.modules.ad_core.utils.compute_sources_and_boundary_conditions_ad import (
        compute_sources_and_boundary_conditions_ad,
    )
    from pymrst.modules.ad_core.utils.get_boundary_condition_fluxes_ad import (
        get_boundary_condition_fluxes_ad,
    )
    from pymrst.modules.ad_core.utils.get_source_fluxes_ad import get_source_fluxes_ad
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import make_rock
    from pymrst.params.gravity.gravity import gravity, isolated_gravity
    from pymrst.params.wells_and_bc import add_bc, add_source
    from pymrst.utils.adi import ad_exp, init_variables_adi, value
    from pymrst.utils.units import CENTI, POISE

    ref = loadmat(HERE/"ref_ad_core_boundary_conditions.mat", squeeze_me=True, struct_as_record=False)
    with isolated_gravity():
        gravity("reset", "on")
        G = compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))
        nc = G.cells.num
        rock = make_rock(G, (1 + np.arange(1, nc + 1))*1e-13, 0.25)
        fluid = init_simple_adi_fluid(phases="WO", mu=np.array([1, 5])*CENTI*POISE, rho=[1000, 800],
                                      n=[2, 2])
        model = GenericBlackOilModel(G, rock, fluid, gas=False).validate_model()
        model.AutoDiffBackend = AutoDiffBackend()
        p, sw = init_variables_adi(1e7 + 1e5*np.arange(1, nc + 1), 0.2 + 0.6*np.arange(1, nc + 1)/nc)
        so = 1 - sw
        mob = [sw**2/1e-3, so**2/5e-3]
        rho = [1000*ad_exp(1e-9*(p - 1e7)), 800*ad_exp(5e-9*(p - 1e7))]
        b = [rho[0]/1000, rho[1]/800]
        pressure, s = [p, p], [sw, so]
        faces = np.atleast_1d(ref["faces"]).astype(np.int64) - 1
        bc = add_bc(None, faces[0:2], "pressure", np.array([1.2e7, 0.9e7]), sat=np.array([1.0, 0.0]))
        bc = add_bc(bc, faces[2:4], "flux", np.array([2e-4, -3e-4]), sat=np.array([0.0, 1.0]))
        bc = add_bc(bc, faces[4:6], "pressure", np.array([1.05e7, 1.4e7]), sat=np.array([0.0, 0.0]))
        bc = add_bc(bc, faces[6:8], "flux", np.array([1e-4, 5e-5]), sat=np.array([0.3, 0.7]))
        bc.type = list(bc.type)
        bc.type[6:8] = ["rate", "rate"]

        qS, bmap, cells, qR = get_boundary_condition_fluxes_ad(model, pressure, s, mob, rho, b, bc)
        r = ref["bc1"]
        ok = report(np.array_equal(cells + 1, np.atleast_1d(r.cells))
                    and np.array_equal(bmap.toarray(), r.map), "boundary faces: cells and map identical")
        ok &= _phases("boundary surface fluxes", qS, r.qS)
        ok &= _phases("boundary reservoir fluxes", qR, r.qR)

        import copy
        bco = copy.deepcopy(bc)
        bco.rho = np.column_stack([np.full(8, 1010.0), np.full(8, 790.0)])
        bco.mob = np.column_stack([np.full(8, 100.0), np.full(8, 30.0)])
        bco.type[6:8] = ["flux", "flux"]
        qS, _, _, qR = get_boundary_condition_fluxes_ad(model, pressure, s, mob, rho, b, bco)
        ok &= _phases("outside rho and mob: surface fluxes", qS, ref["bc2"].qS)
        ok &= _phases("outside rho and mob: reservoir fluxes", qR, ref["bc2"].qR)

        src = add_source(None, np.array([1, 5, 9]) - 1, np.array([1e-4, -2e-4, -1e-4]),
                         sat=np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
        qS, smap, scells = get_source_fluxes_ad(model, mob, s, src)
        r = ref["srcs"]
        ok &= report(np.array_equal(scells + 1, np.atleast_1d(r.cells))
                     and np.array_equal(smap.toarray(), r.map), "sources: cells and map identical")
        ok &= _phases("source surface fluxes", qS, r.qS)
        src_inj = add_source(None, np.array([2, 3]) - 1, np.array([1e-4, 2e-4]),
                             sat=np.array([[1.0, 0.0], [0.5, 0.5]]))
        qS, _, _ = get_source_fluxes_ad(model, mob, s, src_inj)
        r = ref["srcInjOnly"]
        same = all(np.allclose(np.ravel(q), np.ravel(v), rtol=RTOL, atol=0) for q, v in zip(qS, r.val))
        ok &= report(same and not any(hasattr(q, "jac") for q in qS) and not np.any(r.isAD),
                     "injecting sources only: plain rates, as MRST's")

        forces = {"bc": bc, "src": src}
        cs, cb = compute_sources_and_boundary_conditions_ad(model, pressure, s, mob, rho, [], forces)
        for tag, c, rc in (("source", cs, ref["contrib"].src), ("boundary", cb, ref["contrib"].bc)):
            ok &= _phases(f"{tag} contributions: phase mass", c.phaseMass, rc.mass)
            ok &= _phases(f"{tag} contributions: phase volume", c.phaseVolume, rc.vol)
            ok &= report(np.array_equal(c.mapping.toarray(), rc.mapping)
                         and np.array_equal(c.sourceCells + 1, np.atleast_1d(rc.cells)),
                         f"{tag} contributions: mapping and cells identical")

        eqs = [sw*1e-3, so*2e-3]
        out = add_fluxes_from_sources_and_bc(model, eqs, pressure, rho, mob, s, forces)
        ok &= _phases("equations after addFluxesFromSourcesAndBC", out[0], ref["added"].eqs)

        fluid3 = init_simple_adi_fluid(phases="WOG", mu=np.array([1, 5, 0.2])*CENTI*POISE,
                                       rho=[1000, 800, 100], n=[2, 2, 2])
        model3 = GenericBlackOilModel(G, rock, fluid3, disgas=True).validate_model()
        model3.AutoDiffBackend = AutoDiffBackend()
        sg = 0.1 + 0*sw
        so3 = 1 - sw - sg
        mob3 = [sw**2/1e-3, so3**2/5e-3, sg**2/2e-4]
        rho3 = [rho[0], rho[1], 100*ad_exp(1e-8*(p - 1e7))]
        rs = 50 + 1e-6*(p - 1e7)
        dissolved = [[None, None, None], [None, None, None], [None, rs, None]]
        bc3 = add_bc(None, faces[0:3], "pressure", np.array([1.2e7, 0.9e7, 1.3e7]),
                     sat=np.array([0.0, 0.5, 0.5]))
        src3 = add_source(None, np.array([1, 5]) - 1, np.array([1e-4, -2e-4]),
                          sat=np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]]))
        src3.dissolution = np.array([0, 0, 0, 0, 0, 0, 0, 30, 0], dtype=float).reshape(3, 3, order="F")
        cs3, cb3 = compute_sources_and_boundary_conditions_ad(
            model3, [p, p, p], [sw, so3, sg], mob3, rho3, dissolved, {"bc": bc3, "src": src3})
        for tag, c, rc in (("three-phase source", cs3, ref["contrib3"].src),
                           ("three-phase boundary", cb3, ref["contrib3"].bc)):
            ok &= _phases(f"{tag} contributions: phase mass", c.phaseMass, rc.mass)
            ok &= _phases(f"{tag} contributions: phase volume", c.phaseVolume, rc.vol)
    print("\nThe boundary-condition utilities match MRST" if ok
          else "\nThe boundary-condition utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
