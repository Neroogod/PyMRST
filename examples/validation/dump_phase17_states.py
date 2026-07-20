"""Generate the VE states that Phase 17 validates the CO2 inventory against.

Writes states_p17.mat for dump_phase17_reference.m to read back, so MRST's
massTrappingDistributionVEADI is fed exactly the states PyMRST computed and
the comparison isolates the inventory arithmetic from the simulator (which
Phase 9 already validated).
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.gridprocessing import cart_grid, compute_geometry      # noqa: E402
from pymrst.modules.co2lab.trapping import trap_analysis           # noqa: E402
from pymrst.modules.co2lab.ve import (                             # noqa: E402
    VEFluid,
    VEModel,
    VEState,
    average_rock,
    top_surface_grid,
)
from pymrst.params.gravity_const import NORM_GRAVITY               # noqa: E402
from pymrst.params.rock import make_rock                           # noqa: E402
from pymrst.utils.units import DARCY, DAY, MILLI, YEAR             # noqa: E402

NX, NY, NZ = 60, 24, 5
LX, LY, LZ = 2500.0, 1000.0, 25.0
PORO, PERM = 0.22, 300 * MILLI * DARCY


def caprock(x, y):
    return (1000.0 - 0.02 * x
            - 14.0 * np.exp(-(((x - 900.0) / 140.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2))
            - 18.0 * np.exp(-(((x - 1800.0) / 150.0) ** 2
                              + ((y - 500.0) / 260.0) ** 2)))


def build():
    g = cart_grid([NX, NY, NZ], [LX, LY, LZ])
    x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
    g.nodes.coords[:, 2] += caprock(x, y)
    g = compute_geometry(g)
    rock = make_rock(g, PERM, PORO)
    ts = top_surface_grid(g)
    return ts, average_rock(rock, ts)


def build_asymmetric():
    """The same aquifer with the domes moved off the symmetry axis.

    The symmetric caprock is degenerate for the trap analysis: mirror-image
    cells sit at bit-identical depths, so which of them counts as a summit is
    decided by round-off in the grid construction, and MATLAB and Python
    legitimately disagree. Offsetting the domes removes the ties and lets the
    catchments be compared exactly.
    """
    g = cart_grid([NX, NY, NZ], [LX, LY, LZ])
    x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
    g.nodes.coords[:, 2] += (
        1000.0 - 0.02 * x
        - 14.0 * np.exp(-(((x - 900.0) / 140.0) ** 2
                          + ((y - 430.0) / 240.0) ** 2))
        - 18.0 * np.exp(-(((x - 1800.0) / 155.0) ** 2
                          + ((y - 560.0) / 275.0) ** 2)))
    g = compute_geometry(g)
    rock = make_rock(g, PERM, PORO)
    ts = top_surface_grid(g)
    return ts, average_rock(rock, ts)


def main():
    ts, rock2 = build()
    res = trap_analysis(ts)
    fluid = VEFluid(rhoWS=1020.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
                    srw=0.2, srg=0.2, krW_max=1.0, krG_max=1.0,
                    c_water=4.3e-10, c_gas=1e-9, p_ref=1.2e7)

    inj = int(np.argmin(ts.g.cells.centroids[:, 0]
                        + 1e3 * np.abs(ts.g.cells.centroids[:, 1] - 500.0)))
    rate = 0.05e9 / YEAR / fluid.rhoGS
    model = VEModel(ts, rock2, fluid, well_cells=np.array([inj]),
                    well_rate=rate, well_WI=np.array([2e-11]))

    p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * ts.z
    st = VEState(pressure=p0, sG=np.zeros(ts.g.cells.num),
                 sGmax=np.zeros(ts.g.cells.num),
                 qGs=np.array([rate]), bhp=np.array([p0[inj] + 5e5]))

    states = []
    for _ in range(12):
        st, ok, _ = model.step(st, 60.0 * DAY)
        assert ok
        states.append(st)

    quiet = VEModel(ts, rock2, fluid)
    st = VEState(pressure=st.pressure, sG=st.sG, sGmax=st.sGmax,
                 qGs=np.zeros(0), bhp=np.zeros(0))
    for _ in range(20):
        st, ok, _ = quiet.step(st, 1.0 * YEAR)
        assert ok
        states.append(st)

    # keep a handful, spanning injection and migration
    keep = [0, 5, 11, 15, 21, 31]
    p = np.column_stack([states[k].pressure for k in keep])
    sg = np.column_stack([states[k].sG for k in keep])
    sgmax = np.column_stack([states[k].sGmax for k in keep])

    h = np.zeros_like(p)
    hmax = np.zeros_like(p)
    for j, k in enumerate(keep):
        hh, hm = fluid.heights(states[k].sG, states[k].sGmax, ts.H)
        h[:, j], hmax[:, j] = hh, hm

    out = Path(__file__).with_name("states_p17.mat")
    scipy.io.savemat(out, {
        "p": p, "sG": sg, "sGmax": sgmax, "h": h, "hmax": hmax,
        "poro": np.asarray(rock2.poro, float).ravel(),
        "trap_regions": res.trap_regions.astype(float),
        "trap_z": np.asarray(res.trap_z, float).ravel(),
        "z": ts.z, "H": ts.H, "area": ts.area,
        "srw": fluid.srw, "srg": fluid.srg, "rhoGS": fluid.rhoGS,
        "c_gas": fluid.c_gas, "c_water": fluid.c_water, "p_ref": fluid.p_ref,
        "n_traps": res.n_traps,
    })
    print(f"Wrote {out.name}: {len(keep)} states, {res.n_traps} traps, "
          f"{ts.g.cells.num} VE cells")


if __name__ == "__main__":
    main()
