"""Phase 123: sequential no-water transport with wells vs MATLAB.

MRST's TransportOverallCompositionModel freezes pressure, face flux, and
wellSol streams from the pressure solve. It then solves only cell transport
variables (`sT` and independent overall compositions). This validator feeds the
exact MATLAB pressure-step handoff into PyMRST and compares the transport result.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.model import OverallCompositionModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

HERE = Path(__file__).parent
REF = HERE / "ref_p123_comptransport_wells.mat"


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


def as_2d(x, ncols: int) -> np.ndarray:
    arr = np.asarray(x, float)
    return arr.reshape((-1, ncols))


def main() -> int:
    if not REF.exists():
        print("ref_p123_comptransport_wells.mat missing (run dump_phase123_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    p_guess = np.atleast_1d(ref["p_guess"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    wells = add_well(
        None, g, rock, np.array([0]), well_type="rate", val=5e-5,
        compi=np.array([0.0, 1.0]), sign=1,
        wi=np.atleast_1d(ref["inj_WI"]).astype(float),
    )
    wells[0].components = np.array([0.90, 0.05, 0.05])
    wells = add_well(
        wells, g, rock, np.array([nx - 1]), well_type="bhp",
        val=55 * BARSA, compi=np.array([0.0, 1.0]), sign=-1,
        wi=np.atleast_1d(ref["prod_WI"]).astype(float),
    )
    wells[1].components = np.array([0.50, 0.30, 0.20])
    model = OverallCompositionModel(g, rock, mix, temperature=T0, wells=wells)

    st0 = model.initial_state(p0, z0)
    st_guess = st0.copy()
    st_guess.pressure = p_guess.copy()
    stp, ok_p, its_p = model.pressure_step(st_guess, dt)
    assert ok_p, "PyMRST pressure_step did not converge"

    stp_transport = st0.copy()
    stp_transport.pressure = np.atleast_1d(ref["p_pressure"]).astype(float)
    stp_transport.z = as_2d(ref["z_pressure"], mix.n)
    stp_transport.qOs = np.atleast_1d(ref["well_qOs_pressure"]).astype(float)
    stp_transport.qGs = np.atleast_1d(ref["well_qGs_pressure"]).astype(float)
    stp_transport.bhp = np.atleast_1d(ref["well_bhp_pressure"]).astype(float)
    stp_transport.sT = np.ones(nx)
    flux = np.asarray(ref["flux_pressure"], float)
    internal = np.where(np.abs(flux).sum(axis=1) > 0)[0]
    stp_transport.total_flux = flux[internal].sum(axis=1)
    stp_transport.well_flux = as_2d(ref["well_flux_pressure"], 2)
    stp_transport.well_components = as_2d(ref["well_components_pressure"], mix.n)

    stt, ok_t, its_t = model.transport_step(st0, stp_transport, dt)
    assert ok_t, "PyMRST transport_step did not converge"

    print(f"\n{nx} cells, sequential well transport step at {dt / DAY:.0f} days:")
    failed = 0
    failed += compare("pressure state", stp.pressure, ref["p_pressure"],
                      1.0 * BARSA, BARSA, "bar")
    failed += compare("pressure-step well flux", stp.well_flux,
                      as_2d(ref["well_flux_pressure"], 2), 5e-4)
    failed += compare("transport pressure frozen", stt.pressure,
                      ref["p_transport"], 1e-12 * BARSA, BARSA, "bar")
    failed += compare("transport sT", stt.sT, ref["sT_transport"], 2e-5)
    failed += compare("transport composition", stt.z, as_2d(ref["z_transport"], mix.n), 1e-6)

    _, _, s_o, s_g, _, _ = model._flow_props0(stt)
    s_scaled = np.column_stack([s_o * stt.sT, s_g * stt.sT])
    failed += compare("transport phase saturations", s_scaled,
                      as_2d(ref["s_transport"], 2), 2e-5)
    failed += compare("transport qOs", stt.qOs,
                      ref["well_qOs_transport"], 5e-8)
    failed += compare("transport qGs", stt.qGs,
                      ref["well_qGs_transport"], 5e-8)
    failed += compare("transport bhp frozen", stt.bhp,
                      ref["well_bhp_transport"], 1e-12 * BARSA, BARSA, "bar")

    print(f"  ok   PyMRST pressure/transport iterations {its_p}/{its_t} "
          f"(MRST {int(ref['itsP'])}/{int(ref['itsT'])})")
    print("\nCompositional sequential well transport matches MATLAB!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
