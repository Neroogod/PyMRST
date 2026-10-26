"""Compare PyMRST Phase 3 transport results against MATLAB MRST exports.

Explicit transport uses identical fixed substeps, so saturations must match
to round-off. Implicit transport runs the same Newton algorithm to the same
tolerance (1e-6), so saturations are compared at that level.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import add_source, gravity_vector, make_rock, pore_volume
from pymrst.modules.incomp import (
    explicit_transport,
    implicit_transport,
    incomp_tpfa_ad as incomp_tpfa,
    init_simple_fluid,
)
from pymrst.solvers import compute_trans, init_res_sol
from pymrst.utils.units import CENTI, DARCY, DAY, MILLI, POISE

HERE = Path(__file__).parent
FLUID = init_simple_fluid(mu=np.array([1.0, 10.0]) * CENTI * POISE,
                          rho=[1000.0, 700.0], n=[2.0, 2.0])


def check(name: str, py: np.ndarray, ml: np.ndarray, tol: float) -> int:
    diff = float(np.max(np.abs(py - np.asarray(ml, dtype=float).ravel())))
    ok = diff <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(tol {tol:.1e})")
    return 0 if ok else 1


def validate_bl(ref) -> int:
    g = compute_geometry(cart_grid([100, 1, 1], [100.0, 1.0, 1.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
    ht = compute_trans(g, rock)
    rate = float(pore_volume(g, rock).sum()) / (100 * DAY)
    src = add_source(None, np.array([0]), rate, sat=np.array([[1.0, 0.0]]))
    src = add_source(src, np.array([99]), -rate, sat=np.array([[0.0, 1.0]]))
    st0 = init_res_sol(g, 0.0, s0=np.array([[0.0, 1.0]]))
    st0 = incomp_tpfa(st0, g, ht, FLUID, src=src)

    failed = check("p0", st0.pressure, ref["p0"], 1e-9 * 2e7)
    failed += check("flux0", st0.flux, ref["flux0"], 1e-9 * rate)

    tf = 30 * DAY
    st_e = incomp_tpfa(init_res_sol(g, 0.0, s0=np.array([[0.0, 1.0]])),
                       g, ht, FLUID, src=src)
    st_e = explicit_transport(st_e, g, tf, rock, FLUID, src=src,
                              dt=0.25 * DAY)
    failed += check("sE (explicit, fixed dt)", st_e.s[:, 0], ref["sE"], 1e-10)

    st_i = incomp_tpfa(init_res_sol(g, 0.0, s0=np.array([[0.0, 1.0]])),
                       g, ht, FLUID, src=src)
    st_i = implicit_transport(st_i, g, tf, rock, FLUID, src=src)
    failed += check("sI (implicit)", st_i.s[:, 0], ref["sI"], 5e-6)
    return failed


def validate_qfs(ref) -> int:
    g = compute_geometry(cart_grid([32, 32, 1], [500.0, 500.0, 10.0]))
    kvals = (10 + (7 * np.arange(g.cells.num)) % 100) * MILLI * DARCY
    rock = make_rock(g, kvals[:, None], 0.2)
    ht = compute_trans(g, rock)
    rate = 5e-3
    src = add_source(None, np.array([0]), rate, sat=np.array([[1.0, 0.0]]))
    src = add_source(src, np.array([g.cells.num - 1]), -rate,
                     sat=np.array([[0.0, 1.0]]))
    state = init_res_sol(g, 0.0, s0=np.array([[0.0, 1.0]]))
    for _ in range(2):
        state = incomp_tpfa(state, g, ht, FLUID, src=src)
        state = implicit_transport(state, g, 100 * DAY, rock, FLUID, src=src)
    failed = check("sQ (2 sequential cycles)", state.s[:, 0], ref["sQ"], 2e-5)
    failed += check("pQ", state.pressure, ref["pQ"],
                    1e-6 * float(np.abs(ref["pQ"]).max()))
    return failed


def validate_gravity(ref) -> int:
    g = compute_geometry(cart_grid([1, 1, 20], [1.0, 1.0, 20.0]))
    rock = make_rock(g, 1000 * MILLI * DARCY, 0.3)
    fluid = init_simple_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 700.0],
                              n=[2.0, 2.0])
    state = init_res_sol(g, 0.0, s0=np.array([[0.5, 0.5]]))
    state.flux = np.zeros(g.faces.num)
    state = implicit_transport(state, g, 50 * DAY, rock, fluid,
                               gravity=gravity_vector(3))
    return check("sG (gravity segregation)", state.s[:, 0], ref["sG"], 5e-6)


CASES = {
    "ref_p3_bl.mat": validate_bl,
    "ref_p3_qfs.mat": validate_qfs,
    "ref_p3_gravity.mat": validate_gravity,
}


def main() -> int:
    total_failed, missing = 0, 0
    for name, fn in CASES.items():
        path = HERE / name
        if not path.exists():
            print(f"{name}: missing (run dump_phase3_reference.m in MATLAB)")
            missing += 1
            continue
        print(f"{name}:")
        total_failed += fn(scipy.io.loadmat(path, squeeze_me=True))
    if missing == len(CASES):
        print("\nNo reference files found - nothing validated.")
        return 1
    print("\nAll compared arrays match!" if total_failed == 0
          else f"\n{total_failed} comparisons FAILED")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
