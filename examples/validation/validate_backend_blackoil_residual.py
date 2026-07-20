"""Validate Torch backend live-oil residual against PyMRST's SPE1 equations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.modules.ad_blackoil import (
    BackendBOState,
    BackendBlackOilResidual,
    ThreePhaseBlackOilModelAD,
    init_backend_blackoil_fluid,
    initial_state_spe1,
)
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.utils.adi import init_variables_adi, value

HERE = Path(__file__).resolve().parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def _build_spe1():
    deck = read_spe1_deck(DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    fluid = init_deck_blackoil_fluid(deck)
    wells = []
    for name in ("INJECTOR", "PRODUCER"):
        wd = deck["wells"][name]
        wells = add_well(
            wells,
            g,
            rock,
            wd["cells"],
            well_type=wd["type"],
            val=wd["val"],
            radius=wd["radius"],
            compi=wd["compi"],
            sign=wd["sign"],
            name=name,
            ref_depth=wd["ref_depth"],
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )
    model = ThreePhaseBlackOilModelAD(g, rock, fluid, wells, gravity=NORM_GRAVITY)
    backend_fluid = init_backend_blackoil_fluid(deck, backend="torch", device="cpu")
    backend_model = BackendBlackOilResidual(
        g,
        rock,
        backend_fluid,
        wells,
        gravity=NORM_GRAVITY,
        backend="torch",
        device="cpu",
        dtype="float64",
    )
    return deck, model, backend_model


def _reference_residual(model, st0, st, dt):
    st1 = st.status == 1
    x_val = np.where(st1, st.rs, st.s[:, 2])
    p, sw, x, qws, qos, qgs, bhp = init_variables_adi(
        st.pressure,
        st.s[:, 0],
        x_val,
        st.qWs,
        st.qOs,
        st.qGs,
        st.bhp,
    )
    eqs = model.assemble(st0, st, p, sw, x, qws, qos, qgs, bhp, dt)
    return np.concatenate([np.ravel(value(e)) for e in eqs])


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip backend black-oil residual: torch is not installed")
        return 0

    deck, ref_model, backend_model = _build_spe1()
    st = initial_state_spe1(ref_model, deck)
    bst = BackendBOState.from_bo_state(st)
    x0 = torch.as_tensor(backend_model.pack(bst), dtype=torch.float64)
    dt = float(deck["tsteps"][0])

    ref = _reference_residual(ref_model, st, st, dt)
    got = backend_model.residual(bst, bst, dt)(x0).detach().cpu().numpy()
    diff = np.abs(got - ref)
    scale = np.maximum(np.abs(ref), 1.0)
    max_abs = float(diff.max())
    max_rel = float((diff / scale).max())
    ok = max_abs <= 1e-8 and max_rel <= 1e-10
    print(
        f"{'ok  ' if ok else 'FAIL'} SPE1 backend black-oil residual: "
        f"max abs {max_abs:.3e}, max rel {max_rel:.3e}"
    )

    x_req = x0.clone().requires_grad_(True)
    r = backend_model.residual(bst, bst, dt)(x_req)
    probe = torch.sum(r[: ref_model.g.cells.num] ** 2)
    grad = torch.autograd.grad(probe, x_req)[0]
    finite = bool(torch.isfinite(grad).all().detach().cpu())
    print(f"{'ok  ' if finite else 'FAIL'} Torch autograd through residual")
    return 0 if ok and finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
