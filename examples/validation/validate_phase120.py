"""Phase 120: MRST well-control fallback equations vs PyMRST."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import init_variables_adi, value

HERE = Path(__file__).parent
REF = HERE / "ref_p120_well_control_fallback.mat"


def _mat_strings(raw) -> list[str]:
    arr = np.asarray(raw, dtype=object).ravel()
    out: list[str] = []
    for item in arr:
        if isinstance(item, np.ndarray):
            out.append("".join(str(x) for x in item.ravel()).strip())
        else:
            out.append(str(item).strip())
    return out


def _dummy_model(water: bool):
    model = object.__new__(OverallCompositionModel)
    model.water = bool(water)
    return model


def _well(name: str, control_type: str, target: float) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type=control_type,
        val=float(target),
        r=np.array([0.1], dtype=float),
        dir="z",
        WI=np.array([1.0], dtype=float),
        dZ=np.array([0.0], dtype=float),
        name=name,
        compi=np.array([1.0], dtype=float),
        refDepth=0.0,
        sign=-1,
    )


def compare(name: str, py, ref, rtol: float = 1.0e-12, atol: float = 1.0e-14) -> int:
    py = float(np.asarray(value(py), dtype=float).reshape(-1)[0])
    ref = float(np.asarray(ref, dtype=float).reshape(-1)[0])
    diff = abs(py - ref)
    ok = diff <= atol + rtol * max(abs(ref), 1.0)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: PyMRST={py:.12e}, MRST={ref:.12e}, diff={diff:.3e}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p120_well_control_fallback.mat missing (run dump_phase120_reference.m)")
        return 1

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    names = _mat_strings(ref["names"])
    types = _mat_strings(ref["types"])
    water = np.asarray(ref["water"], dtype=bool).ravel()
    targets = np.asarray(ref["targets"], dtype=float).ravel()
    p_bh = np.asarray(ref["pBH"], dtype=float).ravel()
    qw = np.asarray(ref["qw"], dtype=float).ravel()
    qo = np.asarray(ref["qo"], dtype=float).ravel()
    qg = np.asarray(ref["qg"], dtype=float).ravel()
    eq = np.asarray(ref["eq"], dtype=float).ravel()

    print("\nPhase 120: well-control fallback equations vs MATLAB")
    failures = 0
    for i, name in enumerate(names):
        model = _dummy_model(bool(water[i]))
        w = _well(name, types[i], targets[i])
        if water[i]:
            qw_ad, qo_ad, qg_ad, bhp_ad = init_variables_adi(
                np.array([qw[i]]),
                np.array([qo[i]]),
                np.array([qg[i]]),
                np.array([p_bh[i]]),
            )
            ctrl = model._well_control_equation(w, qo_ad, qg_ad, bhp_ad, qw_ad)
        else:
            qo_ad, qg_ad, bhp_ad = init_variables_adi(
                np.array([qo[i]]),
                np.array([qg[i]]),
                np.array([p_bh[i]]),
            )
            ctrl = model._well_control_equation(w, qo_ad, qg_ad, bhp_ad)
        failures += compare(name, ctrl, eq[i])

    if failures:
        print(f"\nPhase 120 validation failed ({failures} checks).")
    else:
        print("\nWell-control fallback equations match MATLAB.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
