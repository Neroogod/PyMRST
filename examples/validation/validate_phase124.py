"""Phase 124: water sequential transport with wells vs MATLAB."""

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
REF = HERE / "ref_p124_comptransport_water_wells.mat"


def build_mixture(ref) -> Mixture:
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.atleast_1d(ref["Tcrit"]).astype(float),
        Pcrit=np.atleast_1d(ref["Pcrit"]).astype(float),
        Vcrit=np.atleast_1d(ref["Vcrit"]).astype(float),
        acentric=np.atleast_1d(ref["acf"]).astype(float),
        molarMass=np.atleast_1d(ref["mw"]).astype(float),
    )


def as_2d(x, ncols: int) -> np.ndarray:
    return np.asarray(x, float).reshape((-1, ncols))


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
        print("ref_p124_comptransport_water_wells.mat missing (run dump_phase124_reference.m)")
        return 1
    ref = scipy.io.loadmat(REF, squeeze_me=True)

    mix = build_mixture(ref)
    nx = int(ref["nx"])
    T0 = float(ref["T0"])
    p0 = np.atleast_1d(ref["p0"]).astype(float)
    sW0 = np.atleast_1d(ref["sW0"]).astype(float)
    z0 = np.atleast_1d(ref["z0"]).astype(float)
    dt = float(ref["dt"])

    g = compute_geometry(cart_grid([nx, 1, 1], [80.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    wells = add_well(
        None, g, rock, np.array([0]), well_type="rate", val=5e-5,
        compi=np.array([0.50, 0.25, 0.25]), sign=1,
        wi=np.atleast_1d(ref["inj_WI"]).astype(float),
    )
    wells[0].components = np.array([0.50, 0.30, 0.20])
    wells = add_well(
        wells, g, rock, np.array([nx - 1]), well_type="bhp",
        val=55 * BARSA, compi=np.array([0.20, 0.40, 0.40]), sign=-1,
        wi=np.atleast_1d(ref["prod_WI"]).astype(float),
    )
    wells[1].components = np.array([0.50, 0.30, 0.20])
    model = OverallCompositionModel(
        g, rock, mix, temperature=T0, water=True, wells=wells,
        n_kr=(2.0, 2.0, 2.0),
    )

    st0 = model.initial_state(p0, z0, sW=sW0)

    stp_transport = st0.copy()
    stp_transport.pressure = np.atleast_1d(ref["p_pressure"]).astype(float)
    stp_transport.z = as_2d(ref["z_pressure"], mix.n)
    s_pressure = as_2d(ref["s_pressure"], 3)
    stp_transport.sW = s_pressure[:, 0]
    stp_transport.qWs = np.atleast_1d(ref["well_qWs_pressure"]).astype(float)
    stp_transport.qOs = np.atleast_1d(ref["well_qOs_pressure"]).astype(float)
    stp_transport.qGs = np.atleast_1d(ref["well_qGs_pressure"]).astype(float)
    stp_transport.bhp = np.atleast_1d(ref["well_bhp_pressure"]).astype(float)
    stp_transport.sT = np.ones(nx)
    flux = np.asarray(ref["flux_pressure"], float)
    internal = np.where(np.abs(flux).sum(axis=1) > 0)[0]
    stp_transport.total_flux = flux[internal].sum(axis=1)
    stp_transport.well_flux = as_2d(ref["well_flux_pressure"], 3)
    stp_transport.well_components = as_2d(ref["well_components_pressure"], mix.n)

    stt, ok_t, its_t = model.transport_step(st0, stp_transport, dt)
    assert ok_t, "PyMRST transport_step did not converge"

    print(f"\n{nx} cells, water sequential well transport step at {dt / DAY:.0f} days:")
    failed = 0
    failed += compare("transport pressure frozen", stt.pressure,
                      ref["p_transport"], 1e-12 * BARSA, BARSA, "bar")
    failed += compare("transport sT", stt.sT, ref["sT_transport"], 3e-5)
    s_ref = as_2d(ref["s_transport"], 3)
    failed += compare("transport sW", stt.sW, s_ref[:, 0] / np.asarray(ref["sT_transport"], float), 3e-5)
    failed += compare("transport composition", stt.z, as_2d(ref["z_transport"], mix.n), 2e-6)

    _, _, s_o, s_g, _, _ = model._flow_props0(stt)
    s_scaled = np.column_stack([stt.sW * stt.sT, s_o * stt.sT, s_g * stt.sT])
    failed += compare("transport phase saturations", s_scaled,
                      s_ref, 3e-5)
    failed += compare("transport qWs", stt.qWs,
                      ref["well_qWs_transport"], 5e-8)
    failed += compare("transport qOs", stt.qOs,
                      ref["well_qOs_transport"], 5e-8)
    failed += compare("transport qGs", stt.qGs,
                      ref["well_qGs_transport"], 5e-8)
    failed += compare("transport bhp frozen", stt.bhp,
                      ref["well_bhp_transport"], 1e-12 * BARSA, BARSA, "bar")

    print(f"  ok   PyMRST/MRST transport iterations {its_t}/{int(ref['itsT'])}")
    print("\nWater compositional sequential well transport matches MATLAB!"
          if failed == 0 else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
