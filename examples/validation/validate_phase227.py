"""Phase 227: PyMRST psideh against a direct MATLAB MRST dump."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.params import psideh  # noqa: E402


HERE = Path(__file__).parent
REF = HERE / "ref_p227_psideh.mat"


class SimpleWOFluid:
    def __init__(self, mu, rho):
        self.mu = np.asarray(mu, dtype=float)
        self.rho = np.asarray(rho, dtype=float)

    def properties(self, state):
        return self.mu.copy(), self.rho.copy()

    def relperm(self, sat):
        return np.asarray(sat, dtype=float)


def _string(raw) -> str:
    arr = np.asarray(raw, dtype=object).ravel()
    if arr.size == 0:
        return ""
    value = arr[0]
    if isinstance(value, np.ndarray):
        return "".join(str(v) for v in value.ravel())
    return str(value)


def _strings(raw) -> list[str]:
    arr = np.asarray(raw, dtype=object).ravel()
    out: list[str] = []
    for value in arr:
        if isinstance(value, np.ndarray):
            out.append("".join(str(v) for v in value.ravel()))
        else:
            out.append(str(value))
    return out


def compare(name, py, ref, rtol=0.0, atol=0.0) -> int:
    py = np.asarray(py)
    ref = np.asarray(ref)
    if py.dtype.kind in {"U", "S", "O"} or ref.dtype.kind in {"U", "S", "O"}:
        ok = list(py.ravel()) == list(ref.ravel())
        diff = 0.0 if ok else np.inf
    elif py.shape != ref.shape and py.size == ref.size:
        ref = ref.reshape(py.shape)
        diff = float(np.max(np.abs(py.astype(float) - ref.astype(float)))) if py.size else 0.0
        scale = max(float(np.max(np.abs(ref.astype(float)))) if ref.size else 0.0, 1.0)
        ok = diff <= atol + rtol * scale
    elif py.shape != ref.shape:
        ok = False
        diff = np.inf
    else:
        diff = float(np.max(np.abs(py.astype(float) - ref.astype(float)))) if py.size else 0.0
        scale = max(float(np.max(np.abs(ref.astype(float)))) if ref.size else 0.0, 1.0)
        ok = diff <= atol + rtol * scale
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
        f"(rtol {rtol:.1e}, atol {atol:.1e})"
    )
    return 0 if ok else 1


def _omega(mu, rho, sat) -> float:
    mob = np.asarray(sat, dtype=float) / np.asarray(mu, dtype=float)
    return float(np.sum(mob * np.asarray(rho, dtype=float)) / np.sum(mob))


def main() -> int:
    if not REF.exists():
        print("ref_p227_psideh.mat missing (run dump_phase227_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    dims = np.asarray(ref["dims"], dtype=int).ravel()
    pdims = np.asarray(ref["pdims"], dtype=float).ravel()
    side = _string(ref["side"])
    sat = np.asarray(ref["sat"], dtype=float).ravel()
    fluid = SimpleWOFluid(ref["mu"], ref["rho"])
    gravity = np.asarray(ref["gravity_vector"], dtype=float).ravel()

    # MATLAB/MRST logical ranges are 1-based. PyMRST uses Python's 0-based
    # logical indexing for the same cell axes.
    i1 = [int(np.asarray(ref["i1"]).item()) - 1]
    i2 = [int(np.asarray(ref["i2"]).item()) - 1]
    normal_range = [int(np.asarray(ref["normal_range"]).item()) - 1]

    g = compute_geometry(cart_grid(dims, pdims))
    bc = psideh(
        None,
        g,
        side,
        fluid,
        i1,
        i2,
        sat=sat,
        search_range=normal_range,
        ref_position=np.asarray(ref["ref_position"], dtype=float).ravel(),
        ref_pressure=float(np.asarray(ref["ref_pressure"]).item()),
        gravity=gravity,
    )

    print("\nPhase 227: psideh direct MATLAB MRST regression")
    failures = 0
    failures += compare("grid dims", g.cartDims, ref["dims"], atol=0.0)
    failures += compare("selected faces", bc.face + 1, ref["bc_faces"], atol=0.0)
    failures += compare("face centroids", g.faces.centroids[bc.face], ref["face_centroids"], atol=1e-12)
    failures += compare("bc pressure values", bc.value, ref["bc_values"], rtol=1e-13, atol=1e-8)
    failures += compare("bc saturation", bc.sat, ref["bc_sat"], atol=0.0)
    failures += compare("bc types", bc.type, _strings(ref["bc_types"]))
    failures += compare("mobility-weighted density", _omega(ref["mu"], ref["rho"], sat), ref["omega"], atol=1e-12)

    expected = (
        float(np.asarray(ref["ref_pressure"]).item())
        + float(np.asarray(ref["omega"]).item())
        * ((g.faces.centroids[bc.face] - np.asarray(ref["ref_position"], dtype=float).ravel()) @ gravity)
    )
    failures += compare("explicit hydrostatic formula", bc.value, expected, rtol=1e-13, atol=1e-8)

    if failures:
        print(f"\nPhase 227 validation failed ({failures} checks).")
    else:
        print("\nPyMRST psideh matches a direct MATLAB MRST psideh dump.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
