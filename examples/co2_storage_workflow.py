"""End-to-end CO2 storage workflow, rendered.

Takes a sloping aquifer with two structural domes and runs the whole
PyMRST CO2 chain:

  3D grid -> top-surface (VE) grid -> structural trap analysis
          -> VE simulation of injection and migration
          -> plots of caprock, traps, and the CO2 plume

Writes PNGs next to this file (or to --out) and prints the trapping
inventory. Run with no display; PyVista renders off-screen.
"""

import argparse
import os

import numpy as np
import pyvista as pv

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.co2lab import (
    VEFluid,
    VEModel,
    VEState,
    average_rock,
    co2_props,
    top_surface_grid,
    trap_analysis,
    volumes_of_traps,
)
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import make_rock
from pymrst.plotting import plot_caprock, plot_plume, plot_traps
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR


def caprock_depth(x, y):
    """A caprock rising toward +x with two domes on the way up."""
    return (1000.0 - 0.02 * x
            - 14.0 * np.exp(-(((x - 900.0) / 140.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2))
            - 18.0 * np.exp(-(((x - 1800.0) / 150.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2)))


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pv.OFF_SCREEN = True

    # --- geology ---------------------------------------------------------
    nx, ny, nz = 60, 24, 5
    g3 = cart_grid([nx, ny, nz], [2500.0, 1000.0, 25.0])
    x, y = g3.nodes.coords[:, 0], g3.nodes.coords[:, 1]
    g3.nodes.coords[:, 2] += caprock_depth(x, y)
    g3 = compute_geometry(g3)
    rock3 = make_rock(g3, 300 * MILLI * DARCY, 0.22)

    ts = top_surface_grid(g3)
    rock2 = average_rock(rock3, ts)
    print(f"3D grid: {g3.cells.num} cells  ->  VE grid: {ts.g.cells.num} "
          f"cells ({g3.cells.num / ts.g.cells.num:.0f}x cheaper)")

    # --- where can CO2 be trapped? ---------------------------------------
    res = trap_analysis(ts)
    caps = volumes_of_traps(ts, res, poro=rock2.poro)
    print(f"\nStructural traps found: {res.n_traps}")
    for i, v in enumerate(caps, start=1):
        cells = np.flatnonzero(res.traps == i)
        xc = ts.g.cells.centroids[cells, 0].mean()
        print(f"  trap {i}: spill depth {res.trap_z[i - 1]:7.1f} m, "
              f"capacity {v / 1e6:5.2f} Mm3  (x ~ {xc:.0f} m)")

    # --- simulate injection, then migration ------------------------------
    co2 = co2_props()
    t_res = 273.15 + 45.0
    rho_g = float(co2.rho(np.array([120 * BARSA]), np.array([t_res]))[0])
    mu_g = float(co2.mu(np.array([120 * BARSA]), np.array([t_res]))[0])
    print(f"\nCO2 at reservoir conditions: rho = {rho_g:.0f} kg/m3, "
          f"mu = {mu_g * 1e6:.1f} uPa.s")

    fluid = VEFluid(rhoWS=1020.0, rhoGS=rho_g, mu_w=8e-4, mu_g=mu_g,
                    srw=0.2, srg=0.1, krW_max=1.0, krG_max=1.0,
                    c_water=4.3e-10, c_gas=1e-9, p_ref=1.2e7)

    # inject downdip (deep end), so CO2 must migrate up through both domes
    inj = int(np.argmin(ts.g.cells.centroids[:, 0]
                        + 1e3 * np.abs(ts.g.cells.centroids[:, 1] - 500.0)))
    rate = 0.05e9 / YEAR / fluid.rhoGS          # 0.05 Mt CO2 / year
    model = VEModel(ts, rock2, fluid, well_cells=np.array([inj]),
                    well_rate=rate, well_WI=np.array([2e-11]))

    p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * ts.z
    st = VEState(pressure=p0, sG=np.zeros(ts.g.cells.num),
                 sGmax=np.zeros(ts.g.cells.num),
                 qGs=np.array([rate]), bhp=np.array([p0[inj] + 5e5]))

    n_inj = 12
    for k in range(n_inj):
        st, ok, _ = model.step(st, 60.0 * DAY)
        assert ok, f"injection step {k} failed"
    injected = model.stored_co2(st)
    print(f"\nInjected {injected / 1e6:.2f} Mm3 (surface) over "
          f"{n_inj * 60 / 365:.1f} years")

    h_inj = model.plume_height(st)

    # shut in and let the plume migrate updip
    quiet = VEModel(ts, rock2, fluid)
    st = VEState(pressure=st.pressure, sG=st.sG, sGmax=st.sGmax,
                 qGs=np.zeros(0), bhp=np.zeros(0))
    for k in range(15):
        st, ok, _ = quiet.step(st, 1.0 * YEAR)
        assert ok, f"migration step {k} failed"
    h_mig = quiet.plume_height(st)
    print(f"CO2 after 15 years of migration: "
          f"{quiet.stored_co2(st) / 1e6:.2f} Mm3 still in place")

    x_before = float(ts.g.cells.centroids[:, 0] @ h_inj / h_inj.sum())
    x_after = float(ts.g.cells.centroids[:, 0] @ h_mig / h_mig.sum())
    print(f"plume centre of mass moved updip: x = {x_before:.0f} m "
          f"-> {x_after:.0f} m")

    # --- pictures --------------------------------------------------------
    plot_caprock(ts, filename=os.path.join(out_dir, "1_caprock.png"),
                 show=False)
    plot_traps(ts, res, poro=rock2.poro,
               filename=os.path.join(out_dir, "2_traps.png"), show=False)
    plot_plume(ts, h_inj,
               filename=os.path.join(out_dir, "3_plume_after_injection.png"),
               show=False)
    plot_plume(ts, h_mig,
               filename=os.path.join(out_dir, "4_plume_after_migration.png"),
               show=False)
    print(f"\nWrote 4 figures to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"))
    main(ap.parse_args().out)
