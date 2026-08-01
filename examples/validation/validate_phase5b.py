"""Compare PyMRST's SPE 1 black-oil simulation against MATLAB ad-blackoil.

Both sides run the first 30 report steps of BENCH_SPE1.DATA fully
implicitly with dissolved gas and tight tolerances.
"""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_props.props.init_deck_adi_fluid import init_deck_blackoil_fluid
from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock

HERE = Path(__file__).parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
N_STEPS = 30


def run_python(order):
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = None
    for name in order:                        # match MATLAB well order
        wd = deck["wells"][name]
        wells = add_well(
            wells, g, rock, wd["cells"], well_type=wd["type"], val=wd["val"],
            radius=wd["radius"], compi=wd["compi"], sign=wd["sign"],
            name=name, ref_depth=wd["ref_depth"],
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )
    model = ThreePhaseBlackOilModelAD(g, rock, fluid, wells)
    st = initial_state_spe1(model, deck)

    hist = {k: [] for k in ("p", "sw", "sg", "rs", "qWs", "qOs", "qGs",
                            "bhp")}
    for i, dt in enumerate(deck["tsteps"][:N_STEPS]):
        st, ok, its = model.step(st, dt)
        assert ok, f"step {i} failed"
        hist["p"].append(st.pressure.copy())
        hist["sw"].append(st.s[:, 0].copy())
        hist["sg"].append(st.s[:, 2].copy())
        hist["rs"].append(st.rs.copy())
        hist["qWs"].append(st.qWs.copy())
        hist["qOs"].append(st.qOs.copy())
        hist["qGs"].append(st.qGs.copy())
        hist["bhp"].append(st.bhp.copy())
    return model, {k: np.vstack(v) for k, v in hist.items()}


def main() -> int:
    path = HERE / "ref_p5b_spe1.mat"
    if not path.exists():
        print("ref_p5b_spe1.mat missing (run dump_phase5b_reference.m)")
        return 1
    ref = scipy.io.loadmat(path, squeeze_me=True)
    order = ("PRODUCER", "INJECTOR") if bool(ref["isProdFirst"]) \
        else ("INJECTOR", "PRODUCER")

    model, py = run_python(order)

    wi_py = np.concatenate([w.WI for w in model.wells])
    checks = [
        ("WI", wi_py, ref["WI"], 1e-10),
        ("pressure", py["p"], ref["p_hist"], 2e-5),
        ("sW", py["sw"], ref["sw_hist"], 2e-5),
        ("sG", py["sg"], ref["sg_hist"], 2e-5),
        ("rs", py["rs"], ref["rs_hist"], 2e-5),
        ("qWs", py["qWs"], ref["qWs"], 2e-5),
        ("qOs", py["qOs"], ref["qOs"], 2e-5),
        ("qGs", py["qGs"], ref["qGs"], 2e-5),
        ("bhp", py["bhp"], ref["bhp"], 2e-5),
    ]
    failed = 0
    for name, a, b, rtol in checks:
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        scale = max(np.abs(b).max(), 1e-300)
        diff = float(np.abs(a - b).max())
        ok = diff <= rtol * scale
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = "
              f"{diff:.3e} (rel {diff / scale:.2e}, rtol {rtol:.0e})")
        failed += 0 if ok else 1

    print("\nSPE 1 matches MATLAB!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
