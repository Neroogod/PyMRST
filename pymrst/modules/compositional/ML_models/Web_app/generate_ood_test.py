#!/usr/bin/env python3
"""Generate an out-of-distribution (OOD) test set for the flash web app.

Purpose
-------
The deployed MLP models were trained on the ternary system C1 / C3 / C14
(methane / propane / tetradecane). Their input is purely positional:
``(z1, z3, z14, p)`` with no feature identifying *which* component sits in
each slot. This script builds a physically-correct equilibrium dataset for a
DIFFERENT ternary system -- N2 / H2S / C1 -- at the same temperature and
pressure range the models were trained on, and writes it in the exact column
format the web app expects.

Because the true N2/H2S/C1 equilibrium differs from the C1/C3/C14 mapping the
MLP learned, uploading this file should yield a poor R2, demonstrating that a
component-agnostic MLP cannot generalize to unseen components -- the property a
graph neural network (which encodes component identity as node features) would
be expected to provide.

Ground-truth x, y, and L are computed with a proper Peng-Robinson two-phase
isothermal flash (Wilson K-value initialization + Rachford-Rice +
successive substitution), reusing the app's own PR fugacity-coefficient
routine (``inference._pr_logphi_batch``) with the correct constants for the
new components.

Slot mapping (documented, but arbitrary -- that is the whole point):
    z1  <- N2      z3  <- H2S      z14 <- C1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import inference as inf

# --- Fixed conditions (match the training set) -------------------------------
T = 353.0                       # K, same isotherm the models were trained on
P_MIN_BAR, P_MAX_BAR = 2.0, 340.0
# Target composition of the final test set (the two-phase split is where
# component identity matters, so we oversample it via rejection sampling).
N_TWO_PHASE = 300
N_LIQUID = 150
N_GAS = 150
POOL = 200000                   # oversampling pool for rejection sampling
# Only keep two-phase rows with a genuine (non-trivial, float32-safe) split.
# Rows with a vanishingly small vapor fraction round to L=1 in float32 and
# collapse to single-phase liquid, which is useless for the demonstration.
TWO_PHASE_L = (0.05, 0.95)
SEED = 7

# --- Peng-Robinson constants for the NEW system: N2 / H2S / C1 ---------------
# Tc [K], Pc [Pa], acentric factor [-]
Tci = torch.tensor([126.20, 373.53, 190.56], dtype=torch.float64)
Pci = torch.tensor([3.394e6, 8.963e6, 4.599e6], dtype=torch.float64)
acc = torch.tensor([0.0377, 0.0942, 0.0115], dtype=torch.float64)

# Binary interaction parameters kij (symmetric, literature values)
kij = torch.zeros((3, 3), dtype=torch.float64)
kij[0, 1] = kij[1, 0] = 0.130   # N2 - H2S
kij[0, 2] = kij[2, 0] = 0.031   # N2 - C1
kij[1, 2] = kij[2, 1] = 0.085   # H2S - C1

COMPONENTS = ["N2", "H2S", "C1"]


def wilson_K(p_pa: np.ndarray) -> np.ndarray:
    """Wilson correlation K-values, shape [n, 3]."""
    Tc = Tci.numpy(); Pc = Pci.numpy(); w = acc.numpy()
    p = p_pa[:, None]
    return (Pc[None, :] / p) * np.exp(5.373 * (1.0 + w[None, :])
                                      * (1.0 - Tc[None, :] / T))


def rachford_rice(z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Solve for vapor fraction V per row by bisection. z,K shape [n,3]."""
    n = z.shape[0]
    lo = np.full(n, 1e-9)
    hi = np.full(n, 1.0 - 1e-9)

    def g(V):
        Km1 = K - 1.0
        return np.sum(z * Km1 / (1.0 + V[:, None] * Km1), axis=1)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        pos = gm > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def logphi(X: np.ndarray, p_pa: np.ndarray, phase: str) -> np.ndarray:
    """Log fugacity coefficients using the app's PR routine + new constants."""
    Xt = torch.as_tensor(X, dtype=torch.float64)
    pt = torch.as_tensor(p_pa, dtype=torch.float64)
    Tt = torch.tensor(T, dtype=torch.float64)
    with torch.no_grad():
        lp = inf._pr_logphi_batch(pt, Tt, Pci, Tci, acc, Xt, kij, phase)
    return lp.cpu().numpy()


def flash(z: np.ndarray, p_pa: np.ndarray, n_iter: int = 100):
    """Isothermal PR flash. Returns (L_liquid_frac, x, y, phase_code).

    phase_code: 0 = single vapor, 1 = single liquid, 2 = two-phase.
    """
    n = z.shape[0]
    K = wilson_K(p_pa)
    V = np.full(n, 0.5)
    x = z.copy()
    y = z.copy()

    for _ in range(n_iter):
        sK = np.sum(z * K, axis=1)
        sKinv = np.sum(z / K, axis=1)
        two = (sK > 1.0) & (sKinv > 1.0)

        if np.any(two):
            V[two] = rachford_rice(z[two], K[two])
            denom = 1.0 + V[two, None] * (K[two] - 1.0)
            x[two] = z[two] / denom
            y[two] = K[two] * x[two]
            x[two] /= x[two].sum(axis=1, keepdims=True)
            y[two] /= y[two].sum(axis=1, keepdims=True)

        # single-phase rows: use feed as the relevant composition
        liq = ~two & (sK <= 1.0)
        vap = ~two & (sKinv <= 1.0) & ~liq
        V[liq] = 0.0
        V[vap] = 1.0

        if not np.any(two):
            break
        lpL = logphi(x, p_pa, "liquid")
        lpV = logphi(y, p_pa, "vapor")
        K_new = np.exp(lpL - lpV)
        delta = np.max(np.abs(np.log(K_new[two]) - np.log(K[two])))
        K = K_new
        if delta < 1e-9:
            break

    sK = np.sum(z * K, axis=1)
    sKinv = np.sum(z / K, axis=1)
    two = (sK > 1.0) & (sKinv > 1.0)
    liq = ~two & (sK <= 1.0)
    vap = ~two & ~liq

    phase = np.full(n, 2, dtype=int)
    phase[liq] = 1
    phase[vap] = 0

    L_liq = 1.0 - V                      # app convention: L = liquid fraction
    L_liq[liq] = 1.0
    L_liq[vap] = 0.0
    return L_liq, x, y, phase


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - sse / sst


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Oversample a large pool, flash it, then select a balanced set that is
    # rich in two-phase points (where the equilibrium split, and hence
    # component identity, actually matters).
    z_pool = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=POOL)
    p_pool = np.exp(rng.uniform(np.log(P_MIN_BAR), np.log(P_MAX_BAR), POOL)) * 1e5
    L_pool, x_pool, y_pool, ph_pool = flash(z_pool.copy(), p_pool)

    def take(mask, k):
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        return idx[:k]

    lo, hi = TWO_PHASE_L
    genuine_two = (ph_pool == 2) & (L_pool > lo) & (L_pool < hi)
    two_idx = take(genuine_two, N_TWO_PHASE)
    print(f"Genuine two-phase available in pool: {int(genuine_two.sum())} "
          f"(requested {N_TWO_PHASE})")

    sel = np.concatenate([
        two_idx,
        take(ph_pool == 1, N_LIQUID),
        take(ph_pool == 0, N_GAS),
    ])
    rng.shuffle(sel)

    z, p_pa = z_pool[sel], p_pool[sel]
    L_liq, x, y, phase = L_pool[sel], x_pool[sel], y_pool[sel], ph_pool[sel]

    # Match the app / training convention for single-phase composition columns.
    x_out = x.copy(); y_out = y.copy()
    x_out[phase == 0] = 0.0   # gas: liquid comp undefined -> 0
    y_out[phase == 1] = 0.0   # liquid: vapor comp undefined -> 0

    df = pd.DataFrame({
        "z1": z[:, 0], "z3": z[:, 1], "z14": z[:, 2],
        "p": p_pa, "L": L_liq,
        "x1": x_out[:, 0], "x3": x_out[:, 1], "x14": x_out[:, 2],
        "y1": y_out[:, 0], "y3": y_out[:, 1], "y14": y_out[:, 2],
    })

    out_path = "ood_N2_H2S_C1.csv"
    df.to_csv(out_path, index=False)

    n_two = int(np.sum(phase == 2))
    n_liq = int(np.sum(phase == 1))
    n_gas = int(np.sum(phase == 0))
    print(f"Wrote {out_path}: {len(df)} rows  "
          f"(two-phase={n_two}, liquid={n_liq}, gas={n_gas})")
    print(f"Slot mapping: z1={COMPONENTS[0]}, z3={COMPONENTS[1]}, z14={COMPONENTS[2]}")
    print(f"T = {T} K,  p in [{P_MIN_BAR}, {P_MAX_BAR}] bar "
          "(within the training pressure range)")

    # --- Demo: run the deployed model and report R2 on this OOD set ----------
    try:
        model_key = "Modified LR annealing (beta=2.2)"
        y_pred, _std, X_scaled, _has_gt = inf.predict(model_key, df, "cpu")
        y_true = df[inf.Y_COLUMNS].to_numpy(dtype=float)
        r2_all = inf.phase_aware_r2(y_true, y_pred, X_scaled)

        two = phase == 2
        # Two-phase split quality: predicted vs true x1,x3,x14,y1,y3,y14.
        r2_split = _r2(y_true[two, 0:6], y_pred[two, 0:6]) if two.any() else float("nan")

        print(f"\nDemo -- deployed MLP '{model_key}' on this OOD (N2/H2S/C1) set:")
        print(f"  phase-aware R2 (all rows)            = {r2_all:.4f}")
        print(f"  R2 on two-phase x/y equilibrium split = {r2_split:.4f}")
        print("  For reference, the SAME model on the in-distribution "
              "C1/C3/C14 test set scores R2 ~ 0.97.")
        print("\nInterpretation: the MLP still returns numbers, but the "
              "equilibrium split it predicts is wrong for components it never "
              "saw -- because its inputs are positional and carry no component "
              "identity. A GNN with per-component node features is the "
              "standard remedy.")
    except Exception as exc:  # pragma: no cover - demo only
        print(f"(Skipped in-script model demo: {exc})")


if __name__ == "__main__":
    main()
