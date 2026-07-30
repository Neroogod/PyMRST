"""Phase 433: compare computeSourcesAndBoundaryConditionsAD with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.utils import compute_sources_and_boundary_conditions_ad
from pymrst.params.wells_and_bc import BoundaryCondition, Source

REF = Path(__file__).with_name("ref_p433_sources_bc.mat")


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase433_reference.m first")
        return 1
    ref = loadmat(REF)
    grid = SimpleNamespace(
        cells=SimpleNamespace(num=2, centroids=ref["cell_centroids"]),
        faces=SimpleNamespace(
            neighbors=ref["neighbors"].astype(np.int64) - 1,
            centroids=ref["face_centroids"],
        ),
    )
    model = SimpleNamespace(
        G=grid,
        operators=SimpleNamespace(t_all=ref["T_all"].ravel()),
        fluid=SimpleNamespace(rho_s=np.array([1000.0, 800.0])),
        gravity_magnitude=0.0,
    )
    pressure = [np.array([100.0, 90.0]), np.array([100.0, 90.0])]
    sat = [np.array([0.2, 0.4]), np.array([0.8, 0.6])]
    mob = [np.array([1.0, 2.0]), np.array([3.0, 2.0])]
    rho = [np.array([1100.0, 1200.0]), np.array([760.0, 720.0])]
    bc = BoundaryCondition(
        face=ref["bc_faces"].astype(np.int64).ravel() - 1,
        type=["pressure", "flux"],
        value=np.array([110.0, -8.0]),
        sat=np.array([[0.75, 0.25], [0.1, 0.9]]),
    )
    src = Source(
        cell=np.array([0, 1]), rate=np.array([5.0, -4.0]),
        sat=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    src_out, bc_out = compute_sources_and_boundary_conditions_ad(
        model, pressure, sat, mob, rho, None, {"src": src, "bc": bc}
    )
    checks = {
        "source mass": (np.column_stack(src_out.phaseMass), ref["src_mass"]),
        "source volume": (np.column_stack(src_out.phaseVolume), ref["src_vol"]),
        "source mapping": (src_out.mapping.toarray(), ref["src_map"]),
        "source cells": (src_out.sourceCells + 1, ref["src_cells"].ravel()),
        "boundary mass": (np.column_stack(bc_out.phaseMass), ref["bc_mass"]),
        "boundary volume": (np.column_stack(bc_out.phaseVolume), ref["bc_vol"]),
        "boundary mapping": (bc_out.mapping.toarray(), ref["bc_map"]),
        "boundary cells": (bc_out.sourceCells + 1, ref["bc_cells"].ravel()),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual), np.asarray(expected)
        err = np.max(np.abs(actual - expected)) if actual.size else 0.0
        ok = actual.shape == expected.shape and err < 1e-10
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d|={err:.3e}")
    print("\ncomputeSourcesAndBoundaryConditionsAD matches MRST!"
          if not failed else f"\n{failed} checks FAILED")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
