"""Phase 113: native SPE1 K-value compositional schedule solve.

This is the next layer after phases 110--112:

* phase 110 validates SPE1 black-oil -> pseudo-compositional state conversion;
* phase 111 validates MRST ``convertRSRVtoKValue`` table parity;
* phase 112 validates the K-value EOS backend inside reduced native flow solves;
* phase 113 runs a SPE1 report step with PyMRST's native compositional models.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.examples.run_compositional_spe1 import (
    run_native_compositional_spe1,
)


def _check_result(out, label: str) -> None:
    if not out.native:
        raise AssertionError(f"{label}: expected a native compositional run")
    if out.mixture.names != ("PseudoOil", "PseudoGas"):
        raise AssertionError(f"{label}: wrong pseudo-component mixture")
    if len(out.compositional_states) != 2:
        raise AssertionError(f"{label}: expected two report states")
    st = out.compositional_states[-1]
    if not np.all(np.isfinite(st.pressure)):
        raise AssertionError(f"{label}: non-finite pressure")
    if not np.all(np.isfinite(st.z)):
        raise AssertionError(f"{label}: non-finite composition")
    if not np.allclose(st.z.sum(axis=1), 1.0):
        raise AssertionError(f"{label}: compositions drifted off simplex")
    if st.sW is None or not np.all((st.sW >= 0.0) & (st.sW <= 1.0)):
        raise AssertionError(f"{label}: invalid water saturation")
    if np.asarray(out.iterations).max() <= 0:
        raise AssertionError(f"{label}: solve did not report Newton work")


def main() -> None:
    overall = run_native_compositional_spe1(nsteps=2, natural=False, min_dt=6.0 * 3600.0)
    _check_result(overall, "overall")
    natural = run_native_compositional_spe1(nsteps=2, natural=True, min_dt=6.0 * 3600.0)
    _check_result(natural, "natural")
    print(
        "phase113 native SPE1 compositional solve passed "
        f"(overall iterations {overall.iterations.tolist()}, "
        f"natural iterations {natural.iterations.tolist()})"
    )


if __name__ == "__main__":
    main()
