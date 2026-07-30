"""Validate ``crossFlowMixture`` and ``calculatePhaseRate`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.statefunctions.facility.component_mixing import (
    crossflow_composition,
)
from pymrst.modules.ad_core.statefunctions.facility.well_phase_flux import (
    calculate_phase_rate,
)
from pymrst.utils.adi import value


def _inputs():
    wells = [
        SimpleNamespace(compi=np.array([0.8, 0.2])),
        SimpleNamespace(compi=np.array([0.1, 0.9])),
    ]
    perf2well = np.array([0, 0, 1, 1])
    mapping = SimpleNamespace(
        wells=wells,
        W=wells,
        perf2well=perf2well,
        is_injector=np.array([True, False]),
        perforation_sum=sp.csr_array(
            (np.ones(4), (perf2well, np.arange(4))), shape=(2, 4)
        ),
    )
    mobility = [
        np.array([2.0, 1.0, 3.0, 2.0]),
        np.array([1.0, 4.0, 2.0, 1.0]),
    ]
    return wells, mapping, mobility


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_facility_crossflow.mat",
        squeeze_me=True,
    )
    wells, mapping, mobility = _inputs()
    mixture = crossflow_composition(
        ref["flux"], ref["topside"], mapping
    )
    mixture_conservative = crossflow_composition(
        ref["flux"], ref["topside"], mapping,
        conserve_component_amount=True,
    )
    phase_rate = np.column_stack([
        value(rate) for rate in calculate_phase_rate(
            ref["tdp"], mobility, mapping, wells, allow_cross_flow=True
        )
    ])
    phase_rate_no_crossflow = np.column_stack([
        value(rate) for rate in calculate_phase_rate(
            ref["tdp"], mobility, mapping, wells, allow_cross_flow=False
        )
    ])
    checks = {
        "crossFlowMixture": (mixture, ref["mixture"]),
        "crossFlowMixture conservative": (
            mixture_conservative, ref["mixture_conservative"]
        ),
        "calculatePhaseRate crossflow": (phase_rate, ref["phase_rate"]),
        "calculatePhaseRate no crossflow": (
            phase_rate_no_crossflow, ref["phase_rate_no_crossflow"]
        ),
    }
    failed = False
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
