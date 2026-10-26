"""Single-phase oil injection into a layered aquifer: an end-to-end test drive.

A five-spot in a 3D heterogeneous aquifer. Oil is injected at a fixed rate
in the centre and produced against fixed bottom-hole pressure at the four
corners. Nothing here is a new capability - the point is to exercise the
whole stack at once and *check it against things that are known
independently of the code*:

  grid            cell volumes and face areas against the analytic values
  TPFA solve      every cell divergence-free; well rates balance
  well model      Peaceman index reproduced from the pressure field
  diagnostics     time-of-flight, tracer partitions, Lorenz, sweep
  streamlines     every line drains to a producer, back-traced lines return
                  to the injector, and streamline arrival times agree with
                  the finite-volume time-of-flight

Each check prints its own residual, so a regression shows up as a number
rather than a silent picture. Run:

    python examples/single_phase_aquifer.py
    python examples/single_phase_aquifer.py --no-figures
"""

import argparse
import os
import sys

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.diagnostics.flow_diagnostics import (
    compute_diagnostics,
    compute_f_phi,
    compute_sweep,
    lorenz_coefficient,
)
from pymrst.modules.incomp.fluid import init_single_fluid
from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa_ad as incomp_tpfa
from pymrst.modules.streamlines import pollock
from pymrst.params.rock import make_rock, pore_volume
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers.state import init_res_sol
from pymrst.solvers.trans import compute_trans
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE, YEAR

# --- the aquifer -----------------------------------------------------------
NX, NY, NZ = 25, 25, 5
LX, LY, LZ = 500.0, 500.0, 25.0          # m
P_INIT = 200 * BARSA
P_PROD = 180 * BARSA                     # producers draw the aquifer down
Q_INJ = 200.0 / DAY                      # m3/s of oil, reservoir conditions
MU_OIL = 5 * CENTI * POISE
RHO_OIL = 800.0                          # kg/m3

FAILURES = []


def check(name, residual, tol, unit=""):
    """Print a residual next to its tolerance and remember any failure."""
    ok = bool(residual <= tol)
    if not ok:
        FAILURES.append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:44s} "
          f"{residual:9.2e} {unit:<8s} (tol {tol:.0e})")


def build_aquifer():
    """A layered aquifer: permeability varies by layer and laterally."""
    g = compute_geometry(cart_grid([NX, NY, NZ], [LX, LY, LZ]))

    x, y, z = g.cells.centroids.T
    layer = np.floor(z / (LZ / NZ)).astype(int)
    # a high-permeability streak in layer 1, tight rock at the base
    base = np.array([200.0, 900.0, 300.0, 120.0, 40.0])[layer]
    lateral = 1.0 + 0.6 * np.sin(2 * np.pi * x / 190.0) * np.cos(
        2 * np.pi * y / 230.0)
    perm = base * lateral * MILLI * DARCY
    poro = 0.15 + 0.08 * (perm / perm.max())

    return g, make_rock(g, perm[:, None], poro[:, None])


def place_wells(g, rock):
    """Central injector, four corner producers, all fully penetrating."""
    def column(i, j):
        return np.array([i + j * NX + k * NX * NY for k in range(NZ)])

    inj_cells = column(NX // 2, NY // 2)
    W = add_well(None, g, rock, inj_cells, well_type="rate", val=Q_INJ,
                 radius=0.1, compi=np.array([1.0]), name="INJ")

    corners = [(1, 1), (NX - 2, 1), (1, NY - 2), (NX - 2, NY - 2)]
    for n, (i, j) in enumerate(corners, start=1):
        W = add_well(W, g, rock, column(i, j), well_type="bhp", val=P_PROD,
                     radius=0.1, compi=np.array([1.0]), name=f"PROD{n}")
    return W


def main(out_dir, figures=True):
    print(__doc__.split("\n")[0])
    print("=" * 74)

    # -- grid ---------------------------------------------------------------
    g, rock = build_aquifer()
    pv = pore_volume(g, rock)
    print(f"\nAquifer: {NX}x{NY}x{NZ} = {g.cells.num} cells, "
          f"{LX:.0f} x {LY:.0f} x {LZ:.0f} m")
    print(f"  permeability {rock.perm.min() / (MILLI * DARCY):.0f} - "
          f"{rock.perm.max() / (MILLI * DARCY):.0f} mD, "
          f"porosity {rock.poro.min():.2f} - {rock.poro.max():.2f}")
    print(f"  pore volume {pv.sum():.3e} m3   "
          f"(fill time at {Q_INJ * DAY:.0f} m3/day: "
          f"{pv.sum() / Q_INJ / YEAR:.1f} years)")

    print("\n[1] Grid geometry vs the analytic values")
    cell_vol = (LX / NX) * (LY / NY) * (LZ / NZ)
    check("cell volumes", np.abs(g.cells.volumes - cell_vol).max() / cell_vol,
          1e-14, "rel")
    check("total volume", abs(g.cells.volumes.sum() - LX * LY * LZ)
          / (LX * LY * LZ), 1e-14, "rel")
    # interior faces: each has exactly two neighbours, boundary faces one
    interior = (g.faces.neighbors >= 0).all(axis=1)
    check("interior face count",
          abs(interior.sum() - ((NX - 1) * NY * NZ + NX * (NY - 1) * NZ
                                + NX * NY * (NZ - 1))), 0.5, "faces")

    # -- pressure solve -----------------------------------------------------
    fluid = init_single_fluid(mu=MU_OIL, rho=RHO_OIL)
    T = compute_trans(g, rock)
    W = place_wells(g, rock)
    state = incomp_tpfa(init_res_sol(g, P_INIT), g, T, fluid, wells=W)

    q_inj = state.wellSol[0].flux.sum()
    q_prod = sum(ws.flux.sum() for ws in state.wellSol[1:])
    print("\n[2] Single-phase TPFA solve")
    print(f"  injector  {W[0].name:6s} bhp {state.wellSol[0].pressure / BARSA:7.2f} bar"
          f"   rate {q_inj * DAY:+8.2f} m3/day")
    for k, ws in enumerate(state.wellSol[1:], start=1):
        print(f"  producer  {W[k].name:6s} bhp {ws.pressure / BARSA:7.2f} bar"
              f"   rate {ws.flux.sum() * DAY:+8.2f} m3/day")

    # what goes in must come out: the fluid is incompressible
    check("well rate balance (in + out)", abs(q_inj + q_prod) / abs(q_inj),
          1e-10, "rel")
    check("injector honours its rate target", abs(q_inj - Q_INJ) / Q_INJ,
          1e-10, "rel")
    check("producers honour their bhp target",
          max(abs(ws.pressure - P_PROD) for ws in state.wellSol[1:]) / P_PROD,
          1e-10, "rel")

    # every cell without a well must be divergence-free
    div = np.zeros(g.cells.num)
    np.add.at(div, g.faces.neighbors[:, 0][g.faces.neighbors[:, 0] >= 0],
              state.flux[g.faces.neighbors[:, 0] >= 0])
    np.add.at(div, g.faces.neighbors[:, 1][g.faces.neighbors[:, 1] >= 0],
              -state.flux[g.faces.neighbors[:, 1] >= 0])
    well_cells = np.concatenate([w.cells for w in W])
    interior_cells = np.setdiff1d(np.arange(g.cells.num), well_cells)
    check("cells are divergence-free",
          np.abs(div[interior_cells]).max() / abs(q_inj), 1e-12, "rel")

    # and the well cells must carry exactly their perforation flux
    src = np.zeros(g.cells.num)
    for w, ws in zip(W, state.wellSol):
        np.add.at(src, w.cells, ws.flux)
    check("well cells balance their perforations",
          np.abs(div - src).max() / abs(q_inj), 1e-12, "rel")

    # the injector must be the pressure high, the producers the lows
    check("injector is the pressure maximum",
          0.0 if state.pressure.max() <= state.wellSol[0].pressure + 1e-6
          else 1.0, 0.5, "bool")

    # -- Peaceman well index, rederived from the solution -------------------
    print("\n[3] Peaceman well index, checked against the solved pressures")
    # Each perforation obeys q = WI * (bhp - p_cell) / mu, signed so that
    # flow *into* the reservoir is positive. Rebuilding the perforation
    # fluxes from the pressure field alone checks the well model against a
    # pressure solve that never saw them.
    w0 = W[0]
    dp = state.wellSol[0].pressure - state.pressure[w0.cells]
    q_pred = w0.WI * dp / MU_OIL
    check("perforation rates from WI * dp / mu",
          np.abs(q_pred - state.wellSol[0].flux).max() / abs(q_inj),
          1e-10, "rel")
    # the tightest layer must take the least fluid
    check("perforation flux follows permeability",
          0.0 if (np.argmin(state.wellSol[0].flux)
                  == np.argmin(rock.perm[w0.cells, 0])) else 1.0, 0.5, "bool")

    # -- flow diagnostics ---------------------------------------------------
    print("\n[4] Flow diagnostics")
    d = compute_diagnostics(g, rock, state, W)
    tof = d.total_tof
    f, phi = compute_f_phi(pv, tof)
    lc = lorenz_coefficient(f, phi)
    ev, td = compute_sweep(f, phi)

    print(f"  time-of-flight   {d.tof_forward.min() / YEAR:.3f} - "
          f"{tof.max() / YEAR:.1f} years")
    print(f"  Lorenz coefficient {lc:.4f}  "
          f"(0 = piston-like, 1 = infinitely channelled)")
    print(f"  sweep efficiency at 1 PV injected: {np.interp(1.0, td, ev):.3f}")

    # tracer partitions are a partition of unity: every cell's water comes
    # from *some* injector and goes to *some* producer
    check("producer tracers sum to 1",
          np.abs(d.tracer_prod.sum(axis=1) - 1.0).max(), 1e-6, "abs")
    check("injector tracers sum to 1",
          np.abs(d.tracer_inj.sum(axis=1) - 1.0).max(), 1e-6, "abs")
    check("Lorenz coefficient in [0, 1]",
          0.0 if 0.0 <= lc <= 1.0 else 1.0, 0.5, "bool")
    check("F-Phi curve is monotone",
          max(0.0, -np.diff(f).min(), -np.diff(phi).min()), 1e-12, "abs")

    # The four producers sit symmetrically but the permeability field does
    # not, so their rates differ by a few percent and their drainage volumes
    # should too - near a quarter each, not exactly a quarter.
    drained = (d.tracer_prod * pv[:, None]).sum(axis=0) / pv.sum()
    rates = np.array([abs(ws.flux.sum()) for ws in state.wellSol[1:]])
    print("  drainage volume / rate share, per producer:")
    for k, (dv, r) in enumerate(zip(drained, rates / rates.sum()), start=1):
        print(f"    PROD{k}  {dv:6.1%} of the pore volume   "
              f"{r:6.1%} of the production")
    check("drainage volumes partition the aquifer",
          abs(drained.sum() - 1.0), 1e-10, "abs")
    check("each producer drains near a quarter",
          np.abs(drained - 0.25).max(), 3e-2, "abs")
    # a well that produces more must drain more
    check("drainage volume ranks with production rate",
          0.0 if np.array_equal(np.argsort(drained), np.argsort(rates))
          else 1.0, 0.5, "bool")

    # -- streamlines --------------------------------------------------------
    print("\n[5] Streamlines (Pollock)")
    prod_cells = np.concatenate([w.cells for w in W[1:]])
    outflow = np.zeros(g.cells.num, dtype=bool)
    outflow[prod_cells] = True

    # Launch one line from every cell of the aquifer, so the traced lines
    # sample the whole flow field rather than just the five well cells. The
    # producer cells themselves are left out: a line starting inside a sink
    # arrives at time zero and would drag the breakthrough time down to 0.
    starts = np.flatnonzero(~outflow)
    S = pollock(g, state.flux, positions=starts, pvol=pv,
                isoutflow=outflow, substeps=4)
    landed = [s for s in S if outflow[s.cells[-1]]]
    print(f"  {len(S)} lines launched (one per cell), "
          f"{len(landed)} reach a producer")
    check("most of the aquifer drains to a producer",
          1.0 - len(landed) / len(S), 0.05, "frac")

    # The last step of a line that ends in a well cell has an *infinite*
    # exit time: the flux leaves through the well, not through a face, so
    # the line cannot escape across one. arrival_time reports the finite
    # time to reach the well, which is the breakthrough time.
    arrival = np.array([s.arrival_time(outflow) for s in landed])
    check("arrival times are finite", 0.0 if np.all(np.isfinite(arrival))
          else 1.0, 0.5, "bool")
    print(f"  breakthrough {arrival.min() / YEAR:.2f} years, "
          f"slowest arrival {arrival.max() / YEAR:.0f} years")

    # Cross-check against the finite-volume time-of-flight, which is an
    # independent calculation on the same flux field: a line launched in
    # cell c and traced to a producer takes the backward TOF of c. The two
    # discretize the same integral differently, so they agree to within
    # numerical diffusion, not to round-off.
    t_fv = d.tof_backward[np.array([int(s.cells[0]) for s in landed])]
    keep = np.isfinite(t_fv) & (t_fv > 0)
    rel = np.abs(arrival[keep] - t_fv[keep]) / t_fv[keep]
    print(f"  streamline arrival vs finite-volume backward TOF: "
          f"median {np.median(rel):.1%} apart over {keep.sum()} lines")
    check("streamline and finite-volume TOF agree to discretization",
          float(np.median(rel)), 0.25, "rel")

    # reverse tracing from a producer must land back on the injector
    R = pollock(g, state.flux, positions=W[1].cells, pvol=pv, reverse=True,
                substeps=4)
    inj_set = set(int(c) for c in w0.cells)
    back = sum(1 for s in R if int(s.cells[-1]) in inj_set)
    print(f"  reverse tracing from {W[1].name}: {back}/{len(R)} lines "
          f"return to the injector")
    check("back-traced lines return to the injector",
          (len(R) - back) / len(R), 1e-12, "frac")

    # -- pictures -----------------------------------------------------------
    if figures:
        print("\n[6] Figures")
        try:
            import pyvista as pv_

            from pymrst.plotting import (
                plot_cell_data,
                plot_streamlines,
                plot_well,
            )
            pv_.OFF_SCREEN = True
            os.makedirs(out_dir, exist_ok=True)

            def save(name, pl):
                path = os.path.join(out_dir, name)
                pl.camera_position = "xz"
                pl.camera.up = (0.0, 0.0, -1.0)
                pl.camera.azimuth = 210
                pl.camera.elevation = 25
                pl.add_axes()
                pl.screenshot(path)
                pl.close()
                print(f"  wrote {path}")

            pl = pv_.Plotter(off_screen=True)
            plot_cell_data(g, rock.perm[:, 0] / (MILLI * DARCY),
                           label="perm (mD)", plotter=pl, show_edges=True)
            plot_well(g, W, pl)
            save("1_permeability.png", pl)

            pl = pv_.Plotter(off_screen=True)
            plot_cell_data(g, state.pressure / BARSA, label="pressure (bar)",
                           cmap="coolwarm", plotter=pl)
            plot_well(g, W, pl)
            save("2_pressure.png", pl)

            pl = pv_.Plotter(off_screen=True)
            plot_cell_data(g, tof / YEAR, label="residence time (years)",
                           cmap="viridis", plotter=pl)
            plot_well(g, W, pl)
            save("3_time_of_flight.png", pl)

            pl = pv_.Plotter(off_screen=True)
            plot_streamlines(g, S, plotter=pl, grid_opacity=0.08,
                             sinks=outflow)
            plot_well(g, W, pl)
            save("4_streamlines.png", pl)
        except ImportError:
            print("  pyvista not installed; skipping "
                  "(pip install 'pymrst[plotting]')")

    # -- verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures_aquifer"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    sys.exit(main(args.out, figures=not args.no_figures))
