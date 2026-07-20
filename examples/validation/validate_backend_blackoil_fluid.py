"""Validate backend SPE1 black-oil fluid tables against PyMRST's fluid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.modules.ad_blackoil import init_backend_blackoil_fluid
from pymrst.modules.ad_props import init_deck_blackoil_fluid
from pymrst.modules.deckformat import read_spe1_deck

HERE = Path(__file__).resolve().parent
DECK = HERE.parents[2] / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def _tonp(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip backend black-oil fluid: torch is not installed")
        return 0

    deck = read_spe1_deck(DECK)
    ref = init_deck_blackoil_fluid(deck)
    be = init_backend_blackoil_fluid(deck, backend="torch", device="cpu")

    p = np.linspace(deck["pressure"].min(), deck["pressure"].max() * 1.15, 9)
    sw = np.linspace(0.12, 0.7, 9)
    sg = np.linspace(0.0, 0.4, 9)
    so = np.maximum(1.0 - sw - sg, 1e-4)
    rs = np.linspace(deck["rs"].min() * 0.8, deck["rs"].max() * 1.2, 9)
    sat = sg > 0.05
    pt = torch.as_tensor(p, dtype=torch.float64)
    swt = torch.as_tensor(sw, dtype=torch.float64)
    sgt = torch.as_tensor(sg, dtype=torch.float64)
    sot = torch.as_tensor(so, dtype=torch.float64)
    rst = torch.as_tensor(rs, dtype=torch.float64)

    checks = [
        ("bW", _tonp(be.b_w(pt)), ref.b_w(p)),
        ("muW", _tonp(be.mu_w(pt)), ref.mu_w(p)),
        ("pvMult", _tonp(be.pv_mult(pt)), ref.pv_mult(p)),
        ("bG", _tonp(be.b_g(pt)), ref.b_g(p)),
        ("muG", _tonp(be.mu_g(pt)), ref.mu_g(p)),
        ("rsSat", _tonp(be.rs_sat(pt)), ref.rs_sat(p)),
        ("bO", _tonp(be.b_o(pt, rst, sat)), ref.b_o(p, rs, sat)),
        ("muO", _tonp(be.mu_o(pt, rst, sat)), ref.mu_o(p, rs, sat)),
        ("krW", _tonp(be.kr_w(swt)), ref.kr_w(sw)),
        ("krG", _tonp(be.kr_g(sgt)), ref.kr_g(sg)),
        ("krO", _tonp(be.kr_o(swt, sot, sgt)), ref.kr_o(sw, so, sg)),
    ]
    failed = 0
    for name, a, b in checks:
        diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        scale = max(float(np.max(np.abs(b))), 1.0)
        ok = diff <= 1e-10 * scale
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {diff:.3e}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
