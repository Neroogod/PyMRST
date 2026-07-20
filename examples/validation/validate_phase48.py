"""Phase 48: Fetkovich analytic aquifers, against MATLAB ad-core.

An analytic aquifer is the water body around a reservoir, modelled as one lumped
tank rather than gridded: it feeds water in through its boundary cells as they
are produced, and its own pressure falls in response. MRST's `AquiferModel`
implements the ECLIPSE Fetkovich model, and this phase ports it.

`dump_phase48_reference.m` isolates the aquifer physics - it does *not* run a
coupled reservoir solve. It probes MRST's `computeAquiferFluxes` at two steps
(dt = 0, where the Fetkovich relaxation coefficient is exactly 1, and a large
dt, where it is meaningfully below 1) and then draws the aquifer down over ten
steps with `updateAfterConvergence`, against a fixed reservoir state. Crucially
it saves the *water pressure and shrinkage MRST evaluated at the connections*,
so PyMRST replays its own formulas on the identical inputs - the physics is
compared, not the two codes' fluid libraries.

Three things are checked:

- the influx formula, `q = alpha J (p_aq - p_w + rho_w g dz) coef`, at both dt;
- the Fetkovich coefficient, which the large-dt sample forces below 1;
- the after-convergence update, `p_aq -= dt Q/(C V)`, `V_aq -= dt Q`, over the
  whole trajectory.

A note on MRST 2024a: `computeInitAquifer.m` (the aquifer-pressure equilibration
helper) references an undefined variable `V_aq` on its last line and cannot run
at all, so it is not part of this comparison; PyMRST's own
`equilibrium_aquifer_pressure` is checked against its analytic optimum in the
module self-tests instead.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections,
    compute_aquifer_fluxes,
    fetkovich_coefficient,
    update_aquifer_state,
)

HERE = Path(__file__).parent
REF = HERE / "ref_p48_aquifer.mat"


def build_connections(ref) -> AquiferConnections:
    """The connection table, from the dumped aquifer parameters."""
    aquid = np.atleast_1d(np.asarray(ref["aquid"], float)).astype(int) - 1
    return AquiferConnections(
        aquid=aquid,
        cells=np.atleast_1d(np.asarray(ref["conncells"], float)).astype(int) - 1,
        j=np.atleast_1d(ref["J"]),
        c=np.atleast_1d(ref["C_conn"]),
        alpha=np.atleast_1d(ref["alpha"]),
        depth_conn=np.atleast_1d(ref["depthconn"]),
        depth_aq=np.full(aquid.size, float(ref["depthaq"])),
        c_aq=np.atleast_1d(np.asarray(ref["C_aq"], float)),
    )


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
        print("ref_p48_aquifer.mat missing (run dump_phase48_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    conn = build_connections(ref)
    p_w = np.atleast_1d(ref["pW"]).astype(float)
    b_w = np.atleast_1d(ref["bW"]).astype(float)
    rho_ws = float(ref["rhoWS"])
    g = float(ref["gz"])
    p_aq0 = np.atleast_1d(np.asarray(ref["p_aq0"], float))
    v_aq0 = np.atleast_1d(np.asarray(ref["V_aq0"], float))

    failed = 0

    print("\nInflux, dt = 0 (Fetkovich coefficient is exactly 1):")
    q0 = compute_aquifer_fluxes(conn, p_aq0, v_aq0, p_w, b_w, rho_ws, g, 0.0)
    failed += compare("q(dt=0)", q0, ref["q_dt0"], 1e-10)

    print("\nInflux, large dt (the coefficient drops below 1):")
    dt_big = float(ref["DT_BIG"])
    q_big = compute_aquifer_fluxes(conn, p_aq0, v_aq0, p_w, b_w, rho_ws, g,
                                   dt_big)
    failed += compare("q(dt=big)", q_big, ref["q_dtbig"], 1e-10)
    coef = fetkovich_coefficient(conn.c, conn.map_to_conn(v_aq0), conn.j, dt_big)
    below = bool(np.all(coef < 0.99))
    print(f"  {'ok  ' if below else 'FAIL'} coefficient is actually exercised: "
          f"coef = {np.array2string(coef, precision=4)} (< 0.99)")
    failed += 0 if below else 1

    print("\nDrawdown trajectory (updateAfterConvergence over the steps):")
    dt_u = float(ref["dtU"])
    nstep = int(ref["nstep"])
    p_aq, v_aq = p_aq0.copy(), v_aq0.copy()
    p_traj, v_traj, q_traj = [], [], []
    for _ in range(nstep):
        q = compute_aquifer_fluxes(conn, p_aq, v_aq, p_w, b_w, rho_ws, g, dt_u)
        q_traj.append(np.asarray(q, float))
        p_aq, v_aq = update_aquifer_state(conn, p_aq, v_aq, q, dt_u)
        p_traj.append(p_aq[0])
        v_traj.append(v_aq[0])

    failed += compare("flux history", np.vstack(q_traj), ref["q_traj"], 1e-10)
    failed += compare("aquifer pressure", np.array(p_traj), ref["p_traj"], 1e-10)
    failed += compare("aquifer volume", np.array(v_traj), ref["v_traj"], 1e-10)

    print("\nFetkovich aquifers match MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
