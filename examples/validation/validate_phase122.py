"""Phase 122: water compositional sequential transport step vs MATLAB.

Checks the water-active `OverallCompositionModel.transport_step` path against
MRST's `TransportOverallCompositionModel`, with pressure and total flux frozen
from the MATLAB pressure solve.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p122_comptransport_water.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def compare(name, py, ml, atol, scale=1.0, unit="") -> int:
    py = np.asarray(py, float)
    ml = np.asarray(ml, float)
    diff = float(np.abs(py - ml).max())
    ok = diff <= atol
    tag = "ok  " if ok else "FAIL"
    print(f"  {tag} {name}: max abs diff = {diff / scale:.3e} {unit} "
          f"(atol {atol / scale:.1e} {unit})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print("ref_p122_comptransport_water.mat missing (run dump_phase122_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    p_guess = np.atleast_1d(ref["p_guess"]).astype(float)
    sw0 = np.atleast_1d(ref["sW0"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    model = OverallCompositionModel(
        g, rock, mix, temperature=T0, water=True,
        n_kr=(2.0, 2.0, 2.0),
    )
    st0 = model.initial_state(p0, z0, sW=sw0)
    st_guess = st0.copy()
    st_guess.pressure = p_guess.copy()

    stp, ok_p, its_p = model.pressure_step(st_guess, dt)
    assert ok_p, "PyMRST pressure_step did not converge"

    stp_transport = st0.copy()
    stp_transport.pressure = np.atleast_1d(ref["p_pressure"]).astype(float)
    stp_transport.z = np.atleast_2d(ref["z_pressure"]).astype(float)
    s_pressure = np.asarray(ref["s_pressure"], float)
    stp_transport.sW = s_pressure[:, 0].copy()
    stp_transport.sT = np.ones(nx)
    flux = np.asarray(ref["flux_pressure"], float)
    internal = np.where(np.abs(flux).sum(axis=1) > 0)[0]
    stp_transport.total_flux = flux[internal].sum(axis=1)

    stt, ok_t, its_t = model.transport_step(st0, stp_transport, dt)
    assert ok_t, "PyMRST transport_step did not converge"

    print(f"\n{nx} cells, water sequential transport step at {dt / DAY:.0f} days:")
    failed = 0
    failed += compare("pressure state", stp.pressure, ref["p_pressure"],
                      5e-1 * BARSA, BARSA, "bar")
    failed += compare("frozen composition", stp.z, ref["z_pressure"], 1e-12)
    assert stp.sW is not None
    failed += compare("frozen water saturation", stp.sW, s_pressure[:, 0], 1e-12)
    failed += compare("transport pressure frozen", stt.pressure,
                      ref["p_transport"], 1e-12 * BARSA, BARSA, "bar")
    failed += compare("transport sT", stt.sT, ref["sT_transport"], 2e-5)
    failed += compare("transport composition", stt.z, ref["z_transport"], 5e-7)

    assert stt.sW is not None
    _, _, s_o, s_g, _, _ = model._flow_props0(stt)
    s_scaled = np.column_stack([stt.sW * stt.sT, s_o * stt.sT, s_g * stt.sT])
    failed += compare("transport phase saturations", s_scaled,
                      ref["s_transport"], 2e-5)

    print(f"  ok   PyMRST pressure/transport iterations {its_p}/{its_t} "
          f"(MRST {int(ref['itsP'])}/{int(ref['itsT'])})")

    print("\nWater compositional sequential transport step matches MATLAB!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
