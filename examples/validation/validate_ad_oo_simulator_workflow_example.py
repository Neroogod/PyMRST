"""Validate MRST simulatorWorkflowExample: the port in
`ad_blackoil/examples/simulator_workflow_example.py`.

The reference is MRST 2024a running the script without its figures
(`dump_simulator_workflow_example_reference.m`). `makeModel3` perturbs the
grid with MATLAB's `randn`, so MRST's two draws are saved with the reference
and passed to the port, which then builds MRST's grid; they are also the
example's default. Every stage is
compared:
- the grid, rock, wells, schedule, initial state and fine operators;
- the fine run: Newton iterations identical at every report step, and
  pressures, water saturations, bhps and surface rates to `RUN_RTOL`;
- the partitions: `partitionUI`, the fault-split grid and `processPartition`
  identical;
- `upscaleModelTPFA`, `upscaleSchedule`, `upscaleState`: pore volumes,
  transmissibilities, coarse rock, initial state, well cells and indices;
- the simply upscaled run, as the fine one;
- `upscaleTrans`: the flow-based transmissibilities and well indices to
  `TRANS_RTOL` (the fine pressure solves' round-off), and the run with them;
- `computeTOFandTracer` for all three final states: times of flight and
  tracers to `DIAG_RTOL`, injector/producer split and partitions identical.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
SETUP_RTOL = 1e-12
RUN_RTOL = 1e-8
TRANS_RTOL = 1e-9
DIAG_RTOL = 1e-9


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.abs(a - b).max()/max(np.abs(b).max(), 1e-300))


def _close(name: str, actual, expected, tol: float) -> bool:
    err = _rel(actual, expected)
    return report(err < tol, f"{name} rel. diff {err:.1e}")


def _wells(tag: str, W, cells_ref, wi_ref, tol: float) -> bool:
    ok = True
    for i, w in enumerate(W):
        ok &= report(np.array_equal(np.atleast_1d(w.cells) + 1, np.atleast_1d(cells_ref[i])),
                     f"{tag} {w.name}: cells identical")
        ok &= _close(f"{tag} {w.name}: WI", np.ravel(w.WI), np.ravel(wi_ref[i]), tol)
    return ok


def _run(tag: str, ws, states, rep, ref) -> bool:
    its = np.array([r["Iterations"] for r in rep["ControlstepReports"]])
    nmini = np.array([len(r["StepReports"]) for r in rep["ControlstepReports"]])
    ok = report(np.array_equal(its, np.atleast_1d(ref.its)),
                f"{tag}: Newton iterations identical at every step ({its.sum()}, "
                f"MRST {np.sum(ref.its)})")
    ok &= report(np.array_equal(nmini, np.atleast_1d(ref.nmini)), f"{tag}: ministeps identical")
    ok &= _close(f"{tag}: pressure", [s["pressure"] for s in states], ref.p, RUN_RTOL)
    ok &= _close(f"{tag}: water saturation", [np.asarray(s["s"])[:, 0] for s in states], ref.sw,
                 RUN_RTOL)
    for key in ("bhp", "qWs", "qOs"):
        ok &= _close(f"{tag}: well {key}", [[getattr(w, key) for w in row] for row in ws],
                     getattr(ref, key), RUN_RTOL)
    return ok


def _diagnostics(tag: str, D, ref) -> bool:
    ok = report(np.array_equal(D.inj + 1, np.atleast_1d(ref.inj))
                and np.array_equal(D.prod + 1, np.atleast_1d(ref.prod)),
                f"{tag}: injectors and producers identical")
    ok &= _close(f"{tag}: time of flight", D.tof, ref.tof, DIAG_RTOL)
    ok &= _close(f"{tag}: injector tracer", D.itracer.ravel(), np.ravel(ref.itracer), DIAG_RTOL)
    ok &= _close(f"{tag}: producer tracers", D.ptracer, ref.ptracer, DIAG_RTOL)
    ok &= report(np.array_equal(D.ipart + 1, ref.ipart) and np.array_equal(D.ppart + 1, ref.ppart),
                 f"{tag}: injector and producer partitions identical")
    return ok


def main() -> int:
    from pymrst.modules.ad_blackoil.examples.simulator_workflow_example import (
        MATLAB_DEFAULT_DRAWS,
        simulator_workflow_example,
    )
    from pymrst.params.gravity.gravity import isolated_gravity

    ref = loadmat(HERE/"ref_simulator_workflow_example.mat", squeeze_me=True,
                  struct_as_record=False)
    with isolated_gravity():
        out = simulator_workflow_example(plot=False, draws=(ref["draws_a"], ref["draws_s"]))
    g, setup, part, up = ref["grid"], ref["setup"], ref["part"], ref["upscaled"]
    ok = report(np.array_equal(MATLAB_DEFAULT_DRAWS[0], np.ravel(ref["draws_a"]))
                and np.array_equal(MATLAB_DEFAULT_DRAWS[1], ref["draws_s"]),
                "the example's default makeModel3 draws are MATLAB's, bit for bit")
    G = out.G
    ok &= report(G.cells.num == g.ncells and G.faces.num == g.nfaces
                and int(np.sum(np.asarray(G.faces.tag) > 0)) == g.nfault,
                f"grid: {G.cells.num} cells, {G.faces.num} faces, "
                f"{int(np.sum(np.asarray(G.faces.tag) > 0))} fault faces as MRST")
    ok &= _close("grid: cell volumes", G.cells.volumes, g.volumes, SETUP_RTOL)
    ok &= _close("grid: cell depths", G.cells.centroids[:, 2], g.depth, SETUP_RTOL)
    ok &= _close("rock: permeability", out.rock.perm, g.perm, SETUP_RTOL)
    ok &= _wells("fine", out.W, ref["wells"].cells, ref["wells"].WI, SETUP_RTOL)
    ok &= _close("schedule: time steps", out.timesteps, setup.timesteps, SETUP_RTOL)
    ok &= _close("initial pressure", out.state0["pressure"], setup.p0, SETUP_RTOL)
    ok &= _close("initial saturations", out.state0["s"], setup.s0, SETUP_RTOL)
    ok &= _close("fine transmissibilities", out.model.operators.T, setup.T, SETUP_RTOL)
    ok &= _close("fine pore volumes", out.model.operators.pv, setup.pv, SETUP_RTOL)
    ok &= _run("fine run", out.wellSols, out.states, out.report, ref["fine"])

    ok &= report(np.array_equal(np.asarray(out.p0) + 1, part.p0), "partitionUI identical")
    ok &= report(out.G_fault.faces.num == part.nfaces_fault,
                 f"fault-split grid: {out.G_fault.faces.num} faces as MRST")
    ok &= report(np.array_equal(np.asarray(out.p) + 1, part.p),
                 f"processPartition identical ({np.max(out.p) + 1} blocks)")

    G_c = out.G_c
    ok &= report(G_c.cells.num == up.ncells and G_c.faces.num == up.nfaces,
                 f"coarse grid: {G_c.cells.num} blocks, {G_c.faces.num} faces as MRST")
    ok &= _close("coarse pore volumes", out.model_c.operators.pv, up.pv, SETUP_RTOL)
    ok &= _close("coarse transmissibilities", out.model_c.operators.T, up.T, SETUP_RTOL)
    ok &= _close("coarse permeability", out.rock_c.perm, up.perm, SETUP_RTOL)
    ok &= _close("coarse porosity", out.rock_c.poro, up.poro, SETUP_RTOL)
    ok &= _close("coarse initial pressure", out.state0_c["pressure"], up.p0, SETUP_RTOL)
    ok &= _close("coarse initial saturations", out.state0_c["s"], up.s0, SETUP_RTOL)
    ok &= _wells("upscaleSchedule", out.schedule_c.controls[0].W, up.Wcells, up.WI, SETUP_RTOL)
    ok &= _run("coarse run", out.wellSols_c, out.states_c, out.report_c, ref["coarse"])

    ok &= _close("upscaleTrans: transmissibilities", out.T_c, up.T_c, TRANS_RTOL)
    ok &= _wells("upscaleTrans", out.W_c, up.W2cells, up.WI2, TRANS_RTOL)
    ok &= _close("flow-upscaled model: transmissibilities", out.model_c2.operators.T, up.T2,
                 TRANS_RTOL)
    ok &= _run("flow-upscaled run", out.wellSols_c2, out.states_c2, out.report_c2, ref["coarse2"])

    ok &= _diagnostics("fine diagnostics", out.D, ref["Dfine"])
    ok &= _diagnostics("coarse diagnostics", out.D_c, ref["Dc"])
    ok &= _diagnostics("flow-upscaled diagnostics", out.D_c2, ref["Dc2"])
    print("\nsimulatorWorkflowExample matches MRST" if ok
          else "\nsimulatorWorkflowExample does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
