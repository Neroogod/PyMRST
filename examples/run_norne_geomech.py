"""Coupled flow and geomechanics on the Norne field - PyMRST's runNorneExample.

The Python equivalent of MRST's `ad-mechanics/examples/runNorneExample.m`,
runnable end to end on the real field:

    python examples/run_norne_geomech.py                 # mini Norne (605 cells)
    python examples/run_norne_geomech.py --subset full   # the 7392-cell model
    python examples/run_norne_geomech.py --plot          # + the interactive inspector

`--plot` opens `plot_toolbar` (MRST's plotToolbar) on the whole run: scrub the
timesteps, switch between pressure, saturation, dilation and pore-volume
change, threshold, clip, and read the histogram of what is shown. It needs
pyvista (`pip install 'pymrst[gui]'` also gets the Qt window with menus).
Nothing is plotted without it - the run prints its results and exits.

Where the time goes, on the full field: assembling the residual takes 0.06 s,
and the *linear solve* of the resulting 45,000 x 45,000 coupled system takes
35 s per Newton iteration. So the run is entirely a linear-algebra problem,
and `PoroelasticSchurSolver` is what makes it tractable - it eliminates the
constant mechanics block against a cached factorisation and preconditions by
fixed stress, for the same answer ~20x faster. `--direct` restores the sparse
direct solve it is checked against.

It downloads Norne's corner-point grid and petrophysics from the OPM
`opm-tests` repository (the same pinned revision MRST uses, ~15 MB, once),
builds the grid with `process_grdecl`, and runs water injection into an
oil-water reservoir **fully coupled to poroelastic mechanics**: injection
raises the pressure, the pressure opens the pore space, the rock deforms,
and the deformation feeds back into the flow.

What it prints is what you would check by hand:

- the grid and rock, so you can see it really is Norne (~2.6 km deep, a
  sealing shale in layer 4, permeabilities spanning four orders of magnitude);
- pressure and saturation as injection proceeds;
- **surface displacement** - the reason to run a coupled model at all;
- three invariants that must hold and would break if the coupling were wrong.

The MATLAB original is `runNorneExample('fluid_model', 'oil water',
'method', 'fully coupled', 'bc_case', 'bottom fixed')`. The boundary
conditions, wells, schedule and reference pressure are the same.
"""

import argparse
import time
from typing import Tuple

import numpy as np

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_mechanics import (
    MechBlackOilModel,
    MechanicModel,
    MechOilWaterModel,
    PoroelasticSchurSolver,
)
from pymrst.modules.ad_mechanics.poromech import (
    FixedStressBlackOilWellModel,
    FixedStressOilWaterWellModel,
    FixedStressWaterWellModel,
    MechWaterWellModel,
)
from pymrst.modules.ad_mechanics.examples.run_2d_case import (
    _blackoil_fluid_from_spe1,
)
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models import (
    simulate_fixed_stress_schedule,
)
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import blackoil_status
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.modules.deckformat.norne import (
    fetch_norne,
    norne_rock,
    read_norne_grdecl,
)
from pymrst.modules.vemmech.vem import ElasticBC, create_augmented_grid, enu2c
from pymrst.params import NORM_GRAVITY, add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE
from pymrst.utils.adi import ad_exp

P_REF = 270 * BARSA          # runNorneExample.m
E_YOUNG = 1e9                # 1 GPa
NU_POISSON = 0.3
BIOT_ALPHA = 1.0


def build_bc(g, p_ref: float) -> ElasticBC:
    """MRST's 'bottom fixed': the base is clamped, the sides and top carry
    the overburden as a confining pressure.

    The traction is `p * n_inward`. PyMRST's face normals run from
    `neighbors[:, 0]` to `neighbors[:, 1]`, so on a boundary face the normal
    points *into* the domain when column 0 is the outside and *out of* it
    when column 1 is - hence the sign flip. (MRST does the same thing with
    the same two-term expression; get it backwards and the reservoir is
    pulled apart instead of squeezed.)
    """
    nb = g.faces.neighbors
    is_bdry = np.any(nb == -1, axis=1)
    sign = ((nb[:, 0] == -1).astype(float) - (nb[:, 1] == -1).astype(float))
    n_inward = g.faces.normals * (sign / np.maximum(g.faces.areas, 1e-30))[:, None]

    # The base, clamped in all three components. Found by its *normal*, not
    # by depth: Norne's base is a corner-point surface with tens of metres of
    # relief, so "the deepest boundary faces" is a couple of faces on one low
    # spot. Clamp those alone and the model is under-constrained - it stays
    # solvable, gives a plausible-looking dilation, and quietly moves by tens
    # of metres. (MRST picks the same faces by their direction tag.)
    bottom = np.flatnonzero(is_bdry & (n_inward[:, 2] < -0.5))
    nodes = np.unique(np.concatenate([
        g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
        for f in bottom
    ]))
    mask = np.ones((nodes.size, 3), dtype=bool)

    # Everything else on the boundary carries the confining pressure.
    outer = np.flatnonzero(is_bdry & ~np.isin(np.arange(g.faces.num), bottom))

    return ElasticBC(
        disp_nodes=nodes,
        disp_mask=mask,
        disp_vals=np.zeros((nodes.size, 3)),
        force_faces=outer,
        force=p_ref * n_inward[outer],
    )


def build_case(subset: str, verbose: bool = True,
               fluid_model: str = "oil water",
               method: str = "fully coupled",
               splitting_tolerance: float = 1.0e-6,
               mrst_oilwater_bug: bool = False,
               mrst_wellbore_bug: bool = False,
               rock_compressibility: float = 4.0e-10):
    """Grid, rock, mechanics, fluid, wells - the whole Norne case."""
    if verbose:
        print("Fetching Norne (OPM opm-tests, pinned revision) ...")
    fetch_norne(verbose=verbose)

    grdecl = read_norne_grdecl(subset=subset)
    g = compute_geometry(process_grdecl(grdecl))
    g = create_augmented_grid(g)
    rock = norne_rock(g, grdecl)

    if verbose:
        kx_md = rock.perm[:, 0] / (MILLI * DARCY)
        kz = rock.perm[:, 2]
        print(f"\nGrid    : {grdecl['cartDims']} = {g.cells.num} cells, "
              f"{g.faces.num} faces, {g.nodes.num} nodes")
        print(f"Depth   : {g.cells.centroids[:, 2].min():.0f} - "
              f"{g.cells.centroids[:, 2].max():.0f} m below datum")
        print(f"Porosity: {rock.poro.min():.3f} - {rock.poro.max():.3f}")
        print(f"Perm    : {kx_md.min():.2f} - {kx_md.max():.0f} mD "
              f"(kx), {int(np.sum(kz == 0))} cells with a sealed kz "
              f"(the layer-4 shale)")

    inj_cell, prod_cell = 9, g.cells.num - 1          # runNorneExample.m
    ref_depth = float(g.cells.centroids[0, 2])
    grav = np.array([0.0, 0.0, NORM_GRAVITY])

    fluid_key = str(fluid_model).lower().replace("-", " ")
    method_key = str(method).lower().replace("-", " ").replace("_", " ")
    split = method_key in {"fixed stress", "fixed stress splitting", "split", "splitting"}
    if method_key not in {"fully coupled", "full", "monolithic"} and not split:
        raise ValueError(f"unsupported Norne coupling method {method!r}")
    blackoil_fluid = None
    p_ref = P_REF
    injection_rate = 1e4 / DAY
    if fluid_key in {"blackoil", "black oil"}:
        deck, blackoil_fluid = _blackoil_fluid_from_spe1()
        p_ref = float(deck["pvtw"]["p_ref"])
        injection_rate = 2.5e6 / DAY
        comp_inj, comp_prod = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    elif fluid_key == "water":
        comp_inj = comp_prod = np.array([1.0])
    elif fluid_key in {"oil water", "oil_water"}:
        comp_inj, comp_prod = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    else:
        raise ValueError(f"unsupported Norne fluid model {fluid_model!r}")

    C = enu2c(np.full(g.cells.num, E_YOUNG), np.full(g.cells.num, NU_POISSON))
    mech = MechanicModel(g, C, alpha=BIOT_ALPHA, el_bc=build_bc(g, p_ref))
    w = add_well(None, g, rock, np.array([inj_cell]), well_type="rate",
                 val=injection_rate, compi=comp_inj, sign=1,
                 name="inj", ref_depth=ref_depth, gravity=grav)
    w = add_well(w, g, rock, np.array([prod_cell]), well_type="bhp",
                 val=p_ref, compi=comp_prod, sign=-1,
                 name="prod", ref_depth=ref_depth, gravity=grav)
    if fluid_key in {"blackoil", "black oil"}:
        # MRST retains the BHP equation when the initially hydrostatic
        # black-oil producer briefly crossflows.  Disallowing that reversal
        # pins qO=0 on the first step and changes the subsequent flash path.
        w[-1].allow_sign_change = True
        w[-1].mrst_zero_flux_mixture = bool(mrst_wellbore_bug)

    if fluid_key in {"blackoil", "black oil"}:
        if split:
            model = FixedStressBlackOilWellModel(
                g, rock, blackoil_fluid, w, mech,
                splitting_tol=splitting_tolerance,
                tol_well=1e-9, gravity=NORM_GRAVITY,
            )
            st = model.initial_state(np.full(g.cells.num, p_ref))
        else:
            model = MechBlackOilModel(
                g, rock, blackoil_fluid, w, mech, disgas=True,
                gravity=NORM_GRAVITY, tol_mech=1e-6, tol_well=1e-9,
            )
            sw = np.zeros(g.cells.num)
            sg = np.zeros(g.cells.num)
            so = np.ones(g.cells.num)
            st = State(
                pressure=np.full(g.cells.num, p_ref), sW=sw, sO=so, sG=sg,
                rs=0.5 * blackoil_fluid.rs_sat(np.full(g.cells.num, p_ref)),
                status=blackoil_status(sw, so, sg, disgas=True),
                qWs=np.zeros(2), qOs=np.zeros(2),
                qGs=np.array([injection_rate, 0.0]),
                bhp=np.full(2, p_ref),
            )
            st = model.seed_state(st)
        return g, model, st

    if fluid_key == "water":
        compressibility = 1e-10

        def b(pressure):
            return ad_exp(compressibility * (pressure - p_ref))

        common = dict(
            pv_mult=lambda pressure: 1.0 + 4e-10 * (pressure - p_ref),
            tol_well=1e-9,
        )
        if split:
            model = FixedStressWaterWellModel(
                g, rock, mech, w, b, 1.0 * CENTI * POISE, 1000.0,
                splitting_tol=splitting_tolerance, **common,
            )
        else:
            model = MechWaterWellModel(
                g, rock, mech, w, b, 1.0 * CENTI * POISE, 1000.0,
                tol_mech=1e-6, **common,
            )
        st = model.initial_state(np.full(g.cells.num, p_ref))
        st["sW"] = np.ones(g.cells.num)
        return g, model, st

    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 10.0]) * CENTI * POISE,
        rho=[1000.0, 700.0], n=[1, 1],
        c=[1e-10, 1e-10], p_ref=p_ref,
    )
    # MRST's initSimpleADIFluid(..., cR=4e-10) stores this multiplier on
    # the fluid.  It is modest in the corrected model but essential when
    # reproducing MRST's missing-volume oil-water Biot term.
    fluid.pv_mult = lambda pressure: (
        1.0 + rock_compressibility * (pressure - p_ref)
    )
    if split:
        model = FixedStressOilWaterWellModel(
            g, rock, fluid, w, mech,
            splitting_tol=splitting_tolerance, tol=1e-7, tol_well=1e-6,
            max_iterations=100,
            gravity=NORM_GRAVITY,
            # Use the dimensionally correct Biot term.  The compatibility
            # switch exists only to reproduce an old MRST example defect.
            mrst_oilwater_bug=mrst_oilwater_bug,
        )
        return g, model, model.initial_state(
            np.full(g.cells.num, p_ref), sw0=0.0
        )
    model = MechOilWaterModel(g, rock, fluid, w, mech,
                              gravity=NORM_GRAVITY, tol_mech=1e-6,
                              tol_cnv=1e-7, tol_mb=1e-7, tol_well=1e-6,
                              mrst_oilwater_bug=mrst_oilwater_bug)
    st = State(
        pressure=np.full(g.cells.num, p_ref), sW=np.zeros(g.cells.num),
        qWs=np.array([1e4 / DAY, 0.0]), qOs=np.array([0.0, -1e4 / DAY]),
        bhp=np.array([p_ref, p_ref]),
    )
    return g, model, model.seed_state(st)


def surface_nodes(g) -> np.ndarray:
    """Nodes on the top surface - where uplift and subsidence show up.

    Found by the *faces*, not by depth: Norne's top is a corner-point
    surface with 40 m of relief, so "the shallowest nodes" is a handful of
    nodes on one high spot, not the surface.
    """
    nb = g.faces.neighbors
    is_bdry = np.any(nb == -1, axis=1)
    sign = ((nb[:, 0] == -1).astype(float) - (nb[:, 1] == -1).astype(float))
    n_inward = g.faces.normals * (sign / g.faces.areas)[:, None]
    # z points down, so a top face is one whose inward normal points down
    top = np.flatnonzero(is_bdry & (n_inward[:, 2] > 0.5))
    return np.unique(np.concatenate([
        g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]] for f in top
    ]))


def run(subset: str = "mini", n_steps: int = 20, plot: bool = False,
        direct: bool = False, fluid_model: str = "oil water",
        method: str = "fully coupled", splitting_tolerance: float = 1.0e-6,
        mrst_oilwater_bug: bool = False, mrst_wellbore_bug: bool = False,
        check: bool = True):
    g, model, st0 = build_case(
        subset, fluid_model=fluid_model, method=method,
        splitting_tolerance=splitting_tolerance,
        mrst_oilwater_bug=mrst_oilwater_bug,
        mrst_wellbore_bug=mrst_wellbore_bug,
    )
    top = surface_nodes(g)

    # runNorneExample.m: one day, then twenty five-day steps
    dts = [1.0 * DAY] + [5.0 * DAY] * n_steps

    # The coupled Jacobian is 45,000 x 45,000 on the full field, and a sparse
    # direct solve of it costs 35 s per Newton iteration - the entire cost of
    # the run. The Schur solver eliminates the (constant) mechanics block
    # against a cached factorisation and preconditions by fixed stress,
    # reaching the same answer ~20x faster. `--direct` restores the direct
    # solve, which is what it is checked against.
    split = isinstance(model, (
        FixedStressWaterWellModel, FixedStressOilWaterWellModel,
        FixedStressBlackOilWellModel,
    ))
    fluid_key = str(fluid_model).lower().replace("-", " ")
    linear = None if direct else PoroelasticSchurSolver()
    nonlinear_iterations = 25 if fluid_key == "water" else 100
    solver = None if split else NonLinearSolver(
        max_iterations=nonlinear_iterations, linear_solver=linear,
        # Gas injection can make the producer's hydrostatic well equation
        # alternate between two Newton states.  MRST globalizes this solve;
        # the equivalent backtracking removes that control-cycle here.
        use_line_search=fluid_key != "water",
        use_relaxation=fluid_key == "oil water",
    )
    print(f"\nLinear solver: "
          f"{'fixed-stress sequential' if split else ('sparse direct (spsolve)' if direct else 'Schur + fixed-stress')}")
    pv = g.cells.volumes * model.rock.poro.ravel()

    coupling_name = "fixed-stress splitting" if split else "fully coupled"
    print(f"\nSimulating {len(dts)} steps "
          f"({sum(dts) / DAY:.0f} days of injection), {coupling_name} ...")
    saturation_key = "sG" if fluid_key in {"blackoil", "black oil"} else "sW"
    saturation_label = f"{saturation_key}_max"
    print(f"{'step':>4}  {'days':>6}  {'p_mean':>9}  {'p_max':>9}  "
          f"{saturation_label:>6}  {'bhp_inj':>9}  {'uz_surf':>9}  {'its':>4}")
    print("-" * 72)

    st = st0
    states = [st0]
    acceptance_reports = []
    t0 = time.time()
    for n, dt in enumerate(dts):
        if split:
            split_states, split_reports = simulate_fixed_stress_schedule(
                model, st, [float(dt)], max_timestep_cuts=6,
            )
            st = split_states[-1]
            split_report = split_reports[-1]
            converged = bool(split_report["Converged"])
            iterations = int(split_report["Iterations"])
            residual = ("" if converged else
                        model.last_split_report.get("failure", ""))
            acceptance_reports.append(split_report)
        else:
            st, rep = solver.solve_timestep(model, st, dt)
            converged, iterations = rep.converged, rep.iterations
            residuals = getattr(rep, "residuals", [])
            residual = residuals[-1] if residuals else ""
            acceptance_reports.append({
                "Converged": bool(rep.converged),
                "Iterations": int(rep.iterations),
                "Residuals": residuals,
                "Cuts": 0,
            })
        if not converged:
            raise RuntimeError(
                f"step {n} did not converge: {residual}")
        states.append(st)

        uz = st["u"].reshape(-1, 3)[top, 2]
        # z points down, so a negative uz is the ground rising
        print(f"{n:>4}  {sum(dts[:n + 1]) / DAY:>6.0f}  "
              f"{st['pressure'].mean() / BARSA:>8.1f}b  "
              f"{st['pressure'].max() / BARSA:>8.1f}b  "
              f"{st.get(saturation_key, np.ones(g.cells.num)).max():>6.3f}  "
              f"{st['bhp'][0] / BARSA:>8.1f}b  "
              f"{-uz.min() * 1000:>8.3f}mm  {iterations:>4}")

    print(f"\nDone in {time.time() - t0:.1f} s.")
    model.acceptance_reports = acceptance_reports
    model.acceptance_timesteps = np.asarray(dts, dtype=float)
    # The solver this run actually used, alongside the reports it produced.
    # `None` on the fixed-stress branch, which drives itself. Attached because
    # the configuration above is not reconstructible from the outside - the
    # Schur linear solver, the iteration cap and the line-search/relaxation
    # switches all vary by fluid - so anything wanting to re-run this schedule
    # the way it was run must be handed the same object rather than build a
    # plausible-looking one.
    model.acceptance_solver = solver
    if check:
        report(g, model, states, dts, top, pv)
    if plot:
        show(g, model, states, dts)
    else:
        print("\nNothing was plotted. Re-run with --plot to open the")
        print("interactive inspector on this schedule (pressure, sW,")
        print("dilation, stress/strain and pore-volume change).")
    return g, model, states


def report(g, model, states, dts, top, pv) -> None:
    """The three things that must be true, and would not be if the coupling
    were wrong. This is the part to check."""
    st0, st = states[0], states[-1]

    uz0 = st0["u"].reshape(-1, 3)[top, 2]
    uz = st["u"].reshape(-1, 3)[top, 2]
    uplift = -(uz - uz0)                     # z is down, so this is upward
    diag = model.mech.diagnostics(st)
    diag0 = model.mech.diagnostics(st0)
    divu = diag["volumetric_strain"]
    divu0 = diag0["volumetric_strain"]
    strain = diag["strain"]
    eff = diag["effective_stress"] / BARSA
    total = diag["total_stress"] / BARSA

    print("\n=== Results " + "=" * 49)
    print(f"Pressure   : {st0['pressure'].mean() / BARSA:.1f} bar -> "
          f"{st['pressure'].mean() / BARSA:.1f} bar (mean), "
          f"max {st['pressure'].max() / BARSA:.1f} bar")
    print(f"Water swept: {100 * np.sum(st['sW'] * pv) / pv.sum():.2f}% "
          f"of the pore volume")
    if "sG" in st:
        print(f"Gas swept  : {100 * np.sum(st['sG'] * pv) / pv.sum():.2f}% "
              f"of the pore volume")
    print(f"Surface    : uplift {uplift.max() * 1000:+.3f} mm, "
          f"subsidence {uplift.min() * 1000:+.3f} mm "
          f"(over {len(top)} surface nodes)")
    print(f"Pore strain: div(u) in [{divu.min():+.2e}, {divu.max():+.2e}]")
    print(f"Strain     : exx/eyy/ezz in "
          f"[{strain[:, :3].min():+.2e}, {strain[:, :3].max():+.2e}]")
    print(f"Eff stress : normal components in "
          f"[{eff[:, :3].min():+.1f}, {eff[:, :3].max():+.1f}] bar")
    print(f"Tot stress : normal components in "
          f"[{total[:, :3].min():+.1f}, {total[:, :3].max():+.1f}] bar")

    print("\n=== Checks " + "=" * 50)

    # 0. The initial state must be in equilibrium: at alpha = 1 a confining
    # pressure of pRef exactly cancels a pore pressure of pRef, so the rock
    # starts undeformed. This is why runNorneExample.m picks that pressure,
    # and it is a sharp test of the boundary conditions - get the sign or the
    # face set wrong and the field starts out visibly bent.
    u0_max = float(np.abs(st0["u"]).max())
    ok0 = u0_max < 1e-6
    print(f"  [{'ok' if ok0 else 'FAIL'}] the initial state is in equilibrium "
          f"(|u| = {u0_max:.2e} m before injection)")

    # 1. The reservoir as a whole dilates when its pressure rises. Note this
    # is a *net* statement: an individual cell can be squeezed by its
    # expanding neighbours, so asserting it cell-by-cell would be wrong.
    dp = st["pressure"] - st0["pressure"]
    ddiv = divu - divu0
    net_dp = float(np.sum(pv * dp))
    net_dilation = float(np.sum(g.cells.volumes * ddiv))
    ok1 = net_dp > 0 and net_dilation > 0
    print(f"  [{'ok' if ok1 else 'FAIL'}] the pressurised reservoir has "
          f"dilated (net dV = {net_dilation:+.1f} m3 for a "
          f"pore-volume-weighted dp of {net_dp / pv.sum() / BARSA:+.1f} bar)")

    # 2. The rock above the injector goes UP, not down: this is the coupling.
    ok2 = uplift.max() > 0
    print(f"  [{'ok' if ok2 else 'FAIL'}] injection lifts the surface "
          f"(max {uplift.max() * 1000:+.3f} mm)")

    # 3. Water is conserved: what went in is in the rock or came back out.
    injected = sum(float(s["qWs"][0]) * dt
                   for s, dt in zip(states[1:], dts))
    # Facility primary-rate signs differ between the single-phase and
    # multiphase well formulations; the producer's declared well sign fixes
    # its physical direction, so its phase-rate magnitude is the withdrawal.
    produced = sum(abs(float(s["qWs"][1])) * dt
                   for s, dt in zip(states[1:], dts))
    # Route through the state-function graph: live-oil fluids require Rs and
    # phase-status arguments even when only the water shrinkage factor is
    # requested, while simple fluids depend on pressure alone.
    b = model.get_props(st.copy(), "ShrinkageFactors")[0]
    b0 = model.get_props(st0.copy(), "ShrinkageFactors")[0]
    # Evaluate the physical Biot pore volume explicitly.  A fixed-stress
    # model deliberately keeps separate current/old linearisation terms;
    # asking its state-function container for both states would apply the
    # current terms to the old state and manufacture a balance error.
    def physical_pore_volume(state, volumetric_strain):
        return (
            g.cells.volumes * model.rock.poro.ravel()
            * model.fluid.pv_mult(state["pressure"])
            + g.cells.volumes * model.alpha * volumetric_strain
        )

    pv_now = physical_pore_volume(st, divu)
    pv_0 = physical_pore_volume(st0, divu0)
    stored = np.sum(pv_now * b * st["sW"]) - np.sum(pv_0 * b0 * st0["sW"])
    balance_scale = max(abs(injected), abs(produced), abs(stored), 1.0)
    err = abs(injected - produced - stored) / balance_scale
    nonlinear_balance_tolerance = 25.0 * max(
        float(getattr(model, "tol_cnv", 0.0)),
        float(getattr(model, "tol_mb", 0.0)),
        float(getattr(model, "tol", 0.0)),
    )
    conservation_tolerance = (
        max(float(getattr(model, "splitting_tol", 0.0)), 1.0e-6)
        if isinstance(model, (
            FixedStressWaterWellModel, FixedStressOilWaterWellModel,
            FixedStressBlackOilWellModel,
        )) else max(1.0e-6, nonlinear_balance_tolerance)
    )
    ok3 = err < conservation_tolerance
    print(f"  [{'ok' if ok3 else 'FAIL'}] water balance closes: "
          f"injected {injected:.4e} - produced {produced:.4e} "
          f"= stored {stored:.4e}  (rel. error {err:.2e})")
    print("\nNote the pore volume in that balance is the *poromechanical* one:")
    print("it includes alpha*div(u), so the balance only closes if the")
    print("mechanics and the flow agree about how much the rock opened.")

    if not (ok0 and ok1 and ok2 and ok3):
        raise SystemExit("\nA check FAILED - see above.")
    print("\nAll checks passed.")


def show(g, model, states, dts) -> None:
    """Open the interactive inspector (MRST's plotToolbar) on the run.

    The whole schedule, not one frame: scrub the timesteps, switch between
    pressure, saturation and the poromechanical fields, threshold, clip, and
    read the histogram of whatever is displayed.
    """
    try:
        import pyvista                                             # noqa: F401
    except ImportError:
        print("\n(--plot needs pyvista: pip install 'pymrst[plotting]')")
        return
    from pymrst.plotting import plot_toolbar

    # Cell fields worth scrubbing through. The mechanical ones are the point
    # of a coupled run: div(u) is how far the rock has opened, and the pore
    # volume is what the flow equation actually sees because of it.
    pv_ref = g.cells.volumes * model.rock.poro.ravel()
    fields = {
        "pressure [bar]": np.vstack([s["pressure"] / BARSA for s in states]),
        "sW": np.vstack([s["sW"] for s in states]),
        "dilation div(u)": np.vstack([
            model.mech.volumetric_strain_full(s["u"]) for s in states]),
        "strain exx": np.vstack([
            model.mech.strain_full(s["u"])[:, 0] for s in states]),
        "strain eyy": np.vstack([
            model.mech.strain_full(s["u"])[:, 1] for s in states]),
        "strain ezz": np.vstack([
            model.mech.strain_full(s["u"])[:, 2] for s in states]),
        "effective stress xx [bar]": np.vstack([
            model.mech.effective_stress_full(s["u"])[:, 0] / BARSA
            for s in states]),
        "effective stress yy [bar]": np.vstack([
            model.mech.effective_stress_full(s["u"])[:, 1] / BARSA
            for s in states]),
        "effective stress zz [bar]": np.vstack([
            model.mech.effective_stress_full(s["u"])[:, 2] / BARSA
            for s in states]),
        "total stress xx [bar]": np.vstack([
            model.mech.total_stress_full(s["u"], s["pressure"])[:, 0] / BARSA
            for s in states]),
        "total stress yy [bar]": np.vstack([
            model.mech.total_stress_full(s["u"], s["pressure"])[:, 1] / BARSA
            for s in states]),
        "total stress zz [bar]": np.vstack([
            model.mech.total_stress_full(s["u"], s["pressure"])[:, 2] / BARSA
            for s in states]),
        "pore volume change [%]": np.vstack([
            100.0 * (model.get_props(s, "PoreVolume") - pv_ref) / pv_ref
            for s in states]),
        "perm kx [mD]": model.rock.perm[:, 0] / (MILLI * DARCY),
    }

    print("\nOpening the interactive inspector (plot_toolbar).")
    print("  - the slider scrubs the 21 timesteps")
    print("  - the dataset menu switches field")
    print("  - close the window to finish")
    plot_toolbar(g, fields, title="Norne: coupled flow and geomechanics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", choices=("mini", "full"), default="mini",
                    help="mini Norne (605 cells, ~1 min) or the full "
                         "7392-cell model (slower)")
    ap.add_argument("--steps", type=int, default=20,
                    help="number of 5-day steps after the first 1-day step")
    ap.add_argument("--direct", action="store_true",
                    help="use the sparse direct linear solver instead of the "
                         "Schur + fixed-stress one (same answer, ~20x slower)")
    ap.add_argument("--plot", action="store_true",
                    help="open the interactive inspector (plot_toolbar) on "
                         "the whole run: scrub timesteps, switch between "
                         "pressure / sW / dilation / pore-volume change")
    args = ap.parse_args()
    run(subset=args.subset, n_steps=args.steps, plot=args.plot,
        direct=args.direct)


if __name__ == "__main__":
    main()
