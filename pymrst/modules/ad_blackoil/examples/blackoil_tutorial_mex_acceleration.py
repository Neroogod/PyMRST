"""Port note for MRST ``blackoilTutorialMexAcceleration.m``."""

from __future__ import annotations

import time

from pymrst.modules.ad_blackoil.examples.simulator_workflow_example import (
    run_simulator_workflow_example,
)


def run_blackoil_tutorial_mex_acceleration():
    """Run the baseline Python/SciPy path and report wall time.

    MRST's example toggles MATLAB MEX acceleration. PyMRST does not use that
    backend; sparse linear algebra is provided by SciPy.
    """
    t0 = time.perf_counter()
    result = run_simulator_workflow_example()
    return {"backend": "scipy", "seconds": time.perf_counter() - t0, "result": result}


if __name__ == "__main__":
    out = run_blackoil_tutorial_mex_acceleration()
    print(f"{out['backend']} backend: {out['seconds']:.3f} s")
