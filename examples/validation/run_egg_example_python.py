"""Run the eggExample port without its figures and save what
`validate_ad_oo_egg_example.py` compares (tens of minutes)."""

from pathlib import Path
from time import perf_counter

import numpy as np

from pymrst.modules.ad_blackoil.examples.egg.egg_example import egg_example
from pymrst.params.gravity.gravity import isolated_gravity

HERE = Path(__file__).resolve().parent


def main() -> None:
    timer = perf_counter()
    with isolated_gravity():
        out = egg_example(plot=False)
    walltime = perf_counter() - timer
    reports = out.report["ControlstepReports"]
    keep = np.unique(np.r_[np.arange(10, len(out.states) + 1, 10), len(out.states)]) - 1
    np.savez(HERE/"py_egg_example.npz", bhp=out.bhp, orat=out.orat, wrat=out.wrat, T=out.T,
             its=np.asarray(out.report["Iterations"]),
             nmini=np.array([len(r["StepReports"]) for r in reports]),
             keep=keep + 1,
             p=np.array([out.states[k]["pressure"] for k in keep]),
             sw=np.array([np.asarray(out.states[k]["s"])[:, 0] for k in keep]),
             p0=np.asarray(out.state0["pressure"]), s0=np.asarray(out.state0["s"]),
             lsolver=type(out.nonlinear.LinearSolver).__name__,
             backend=type(out.model.AutoDiffBackend).__name__,
             names=np.array([w.name for w in out.W]), inj=out.inj + 1, prod=out.prod + 1,
             walltime=walltime)
    print(f"saved after {walltime:.0f} s")


if __name__ == "__main__":
    main()
