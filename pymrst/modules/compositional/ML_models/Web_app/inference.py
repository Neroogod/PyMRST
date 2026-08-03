#!/usr/bin/env python3
"""Core inference + metrics backend for the flash-calculation web app.

Loads a 7-fold ensemble of FlashNet models for a chosen training scheme,
runs them on user-supplied feed data, and reports predictions plus
physics-consistency / accuracy metrics.

Key design choices for deployment (differ from the offline benchmark):

  * Scalers are fit ONCE on the bundled reference set (reference/test_data.csv)
    and held fixed -- they are NOT refit on user uploads (which may be tiny).
    MinMax scaling is valid across schemes because every model shares the same
    physical input/output ranges (compositions in [0, 1], the same DOE pressure
    range, L in [0, 1]).
  * Predictions are the mean across the 7 fold models; the per-fold spread is
    returned as an uncertainty estimate.
  * Metrics adapt to whether ground-truth equilibrium compositions are present:
      - R2 requires ground truth (predicted vs true outputs).
      - CompRMSE / InterRMSE are self-consistency residuals (no labels needed).
      - FugRMSE is a thermodynamic residual on the predictions; its sample
        gating/weighting uses the true L/x/y when available, else the
        predicted L/x/y.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

_HERE = Path(__file__).resolve().parent
REFERENCE_CSV = _HERE / "reference" / "test_data.csv"

INPUT_DIM = 6
OUTPUT_DIM = 7
HIDDEN_WIDTH = 256
HIDDEN_DEPTH = 4
N_FOLDS = 7

X_COLUMNS = ["z1", "z3", "z14", "p"]
Y_COLUMNS = ["x1", "x3", "x14", "y1", "y3", "y14", "L"]
GT_COLUMNS = ["x1", "x3", "x14", "y1", "y3", "y14"]

# --- Thermodynamic constants (3-component Peng-Robinson) ---------------------
_RR = 8.3144621  # universal gas constant, J/(mol K)
TinK = 353.0
Pci = torch.tensor([4.604e6, 4.249e6, 1.620e6], dtype=torch.float64)
Tci = torch.tensor([190.58, 369.82, 723.00], dtype=torch.float64)
acc = torch.tensor([0.0108, 0.1520, 0.8000], dtype=torch.float64)
dij = torch.zeros((3, 3), dtype=torch.float64)
dij[0, 1] = dij[1, 0] = 0.00683
dij[0, 2] = dij[2, 0] = 0.01230
dij[1, 2] = dij[2, 1] = 0.00082

# --- Selectable component sets for the fugacity metric -----------------------
# The models are component-agnostic (positional inputs), but the iso-fugacity
# residual needs the actual Peng-Robinson constants of the components under
# test. The training system is C1/C3/C14; the OOD demo system is N2/H2S/C1.
COMPONENT_SETS: Dict[str, Dict[str, object]] = {
    "C1 / C3 / C14 (training)": {
        "labels": ["C1", "C3", "C14"],
        "Pci": [4.604e6, 4.249e6, 1.620e6],
        "Tci": [190.58, 369.82, 723.00],
        "acc": [0.0108, 0.1520, 0.8000],
        "kij": {(0, 1): 0.00683, (0, 2): 0.01230, (1, 2): 0.00082},
    },
    "N2 / H2S / C1 (OOD demo)": {
        "labels": ["N2", "H2S", "C1"],
        "Pci": [3.394e6, 8.963e6, 4.599e6],
        "Tci": [126.20, 373.53, 190.56],
        "acc": [0.0377, 0.0942, 0.0115],
        "kij": {(0, 1): 0.130, (0, 2): 0.031, (1, 2): 0.085},
    },
}
DEFAULT_COMPONENT_SET = "C1 / C3 / C14 (training)"


def component_constants(component_set: str = DEFAULT_COMPONENT_SET
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (Pci, Tci, acc, dij) tensors for a named component set."""
    if component_set not in COMPONENT_SETS:
        raise KeyError(f"Unknown component set '{component_set}'. "
                       f"Options: {list(COMPONENT_SETS)}")
    cfg = COMPONENT_SETS[component_set]
    pc = torch.tensor(cfg["Pci"], dtype=torch.float64)
    tc = torch.tensor(cfg["Tci"], dtype=torch.float64)
    ac = torch.tensor(cfg["acc"], dtype=torch.float64)
    d = torch.zeros((3, 3), dtype=torch.float64)
    for (i, j), v in cfg["kij"].items():
        d[i, j] = d[j, i] = float(v)
    return pc, tc, ac, d


# --- Model registry: display name -> (subdir, checkpoint filename template) --
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "DOE optimum (Run 276)": {"dir": "models/doe276", "template": "model_run276_fold{fold}.pt"},
    "ReLoBRaLo": {"dir": "models/relobralo", "template": "model_runtest_fold{fold}.pt"},
    "Modified LR annealing (beta=1)": {"dir": "models/beta1", "template": "model_runtest_beta1_fold{fold}.pt"},
    "Modified LR annealing (beta=1.2)": {"dir": "models/beta1.2", "template": "model_runtest_beta1.2_fold{fold}.pt"},
    "Modified LR annealing (beta=1.5)": {"dir": "models/beta1.5", "template": "model_runtest_beta1.5_fold{fold}.pt"},
    "Modified LR annealing (beta=2)": {"dir": "models/beta2", "template": "model_runtest_beta2_fold{fold}.pt"},
    "Modified LR annealing (beta=2.2)": {"dir": "models/beta2.2", "template": "model_runtest_beta2.2_fold{fold}.pt"},
    "Modified LR annealing (beta=2.5)": {"dir": "models/beta2.5", "template": "model_runtest_beta2.5_fold{fold}.pt"},
    "Modified LR annealing (beta=3)": {"dir": "models/beta3", "template": "model_runtest_beta3_fold{fold}.pt"},
}


class FlashNet(nn.Module):
    """Feed-forward flash-calculation network (ReLU hidden, sigmoid output)."""

    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                 width: int = HIDDEN_WIDTH, depth: int = HIDDEN_DEPTH) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.ReLU()]
            dim = width
        layers += [nn.Linear(dim, output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def phase_flags_from_L(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gas, liquid, two-phase) one-hot flags from the phase label L."""
    gas = (L == 0).astype(np.float32)
    liq = (L == 1).astype(np.float32)
    two = ((gas == 0) & (liq == 0)).astype(np.float32)
    return gas, liq, two


# -----------------------------------------------------------------------------
# Vectorised Peng-Robinson log-fugacity coefficients (batched float64).
# -----------------------------------------------------------------------------
def _pr_logphi_batch(p: torch.Tensor, TinK_: torch.Tensor, pCrit: torch.Tensor,
                     tCrit: torch.Tensor, accFact: torch.Tensor, X: torch.Tensor,
                     deltaij: torch.Tensor, phase: str) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.float64)
    X = torch.as_tensor(X, dtype=torch.float64)
    X = X / (X.sum(dim=1, keepdim=True) + 1e-18)

    Tr = TinK_ / tCrit
    kappa = (-0.26992 * accFact + 1.54226) * accFact + 0.37464
    mask = (accFact > 0.5) & (kappa > 1.07829)
    kappa_alt = ((0.01667 * accFact - 0.1644) * accFact + 1.485) * accFact + 0.3796
    kappa = torch.where(mask, kappa_alt, kappa)
    sqrtAlpha = 1.0 + kappa * (1.0 - torch.sqrt(Tr))
    alpha = sqrtAlpha * sqrtAlpha
    aa = alpha * (0.457235 * _RR * _RR) * tCrit * tCrit / pCrit
    bb = 0.077796 * _RR * tCrit / pCrit
    aij = (1.0 - deltaij) * torch.sqrt(torch.outer(aa, aa))

    bMix = X @ bb
    aMix = torch.einsum("ni,ij,nj->n", X, aij, X)
    A = aMix * p / (_RR * _RR * TinK_ * TinK_)
    B = bMix * p / (_RR * TinK_)

    a2 = -(1.0 - B)
    a1 = A - 3.0 * B * B - 2.0 * B
    a0 = -B * (A - B * (1.0 + B))
    pp = (3.0 * a1 - a2 * a2) / 3.0
    qq = (2.0 * a2 ** 3 - 9.0 * a2 * a1 + 27.0 * a0) / 27.0
    disc = (qq / 2.0) ** 2 + (pp / 3.0) ** 3
    has3 = disc < 0.0

    POS_EPS, eps = 1e-12, 1e-30
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    u = torch.sign(-qq / 2.0 + sqrt_disc) * torch.abs(-qq / 2.0 + sqrt_disc + 1e-30) ** (1.0 / 3.0)
    v = torch.sign(-qq / 2.0 - sqrt_disc) * torch.abs(-qq / 2.0 - sqrt_disc + 1e-30) ** (1.0 / 3.0)
    z_one = (u + v) - a2 / 3.0
    z_one_safe = torch.where(z_one > POS_EPS, z_one, 1.0 + B)

    r = torch.sqrt(torch.clamp(-pp / 3.0, min=0.0) + 1e-30)
    denom = torch.sqrt(torch.clamp(-(pp / 3.0) ** 3, min=1e-30))
    phi = torch.acos(torch.clamp((-qq / 2.0) / denom, -1.0, 1.0))
    y1 = 2.0 * r * torch.cos(phi / 3.0)
    y2 = 2.0 * r * torch.cos((phi + 2.0 * torch.pi) / 3.0)
    y3 = 2.0 * r * torch.cos((phi + 4.0 * torch.pi) / 3.0)
    z_roots = torch.stack([y1 - a2 / 3.0, y2 - a2 / 3.0, y3 - a2 / 3.0], dim=0)
    big = 1e30
    z_min = torch.where(z_roots > POS_EPS, z_roots, torch.full_like(z_roots, big)).min(dim=0).values
    z_max = torch.where(z_roots > POS_EPS, z_roots, torch.full_like(z_roots, -big)).max(dim=0).values

    if phase.lower() == "liquid":
        z_sel = torch.where(has3, z_min, z_one_safe)
    elif phase.lower() == "vapor":
        z_sel = torch.where(has3, z_max, z_one_safe)
    else:
        raise ValueError("phase must be 'liquid' or 'vapor'")

    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
    xaij = X @ aij
    term = 2.0 * xaij / (aMix[:, None] + eps) - bb[None, :] / (bMix[:, None] + eps)
    logPhi = (bb[None, :] / (bMix[:, None] + eps) * (z_sel[:, None] - 1.0)
              - torch.log(z_sel - B + eps)[:, None]
              - (A / (2.0 * sqrt2 * (B + eps)))[:, None] * term
              * torch.log((z_sel + 2.414213562373095 * B)
                          / (z_sel - 0.414213562373095 * B + eps) + eps)[:, None])
    return logPhi


# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_reference_scalers() -> Tuple[MinMaxScaler, MinMaxScaler]:
    """Fit (pressure, output) MinMax scalers once on the bundled reference set."""
    if not REFERENCE_CSV.is_file():
        raise FileNotFoundError(
            f"Reference CSV not found: {REFERENCE_CSV}. It is required to fix the "
            "input/output scaling used by every model.")
    df = pd.read_csv(REFERENCE_CSV, engine="python", on_bad_lines="skip")
    gas, _liq, two = phase_flags_from_L(df["L"].to_numpy(dtype=np.float32))
    df["Gas"] = gas
    df["2phase"] = two
    df.loc[df["Gas"] > 0.5, ["x1", "x3", "x14"]] = 0.0
    df.loc[(df["Gas"] < 0.5) & (df["2phase"] < 0.5), ["y1", "y3", "y14"]] = 0.0
    X = df[X_COLUMNS + ["Gas", "2phase"]].to_numpy(dtype=np.float32)
    Y = df[Y_COLUMNS].to_numpy(dtype=np.float32)
    scP = MinMaxScaler().fit(X[:, [3]])
    scY = MinMaxScaler().fit(Y)
    return scP, scY


@functools.lru_cache(maxsize=len(MODEL_REGISTRY))
def load_ensemble(model_key: str, device_str: str = "cpu") -> Tuple[FlashNet, ...]:
    """Load and cache the 7-fold FlashNet ensemble for a given scheme."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[model_key]
    device = torch.device(device_str)
    models: List[FlashNet] = []
    for fold in range(1, N_FOLDS + 1):
        path = _HERE / cfg["dir"] / cfg["template"].format(fold=fold)
        if not path.is_file():
            continue
        m = FlashNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError(f"No checkpoints found for '{model_key}' in {cfg['dir']}.")
    return tuple(models)


def available_models() -> List[str]:
    """Return the registry keys whose checkpoint folders actually exist."""
    out = []
    for key, cfg in MODEL_REGISTRY.items():
        first = _HERE / cfg["dir"] / cfg["template"].format(fold=1)
        if first.is_file():
            out.append(key)
    return out


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------
def prepare_inputs(df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """Validate the uploaded frame and build the model input matrix.

    Returns (X, has_ground_truth) where X is [N, 6] = [z1,z3,z14,p,Gas,2phase]
    (raw, unscaled) and has_ground_truth indicates that true equilibrium
    compositions are present.
    """
    df = df.copy()
    missing = [c for c in X_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input column(s): {missing}. "
                         f"Required: {X_COLUMNS} plus phase info (L or Gas/2phase).")

    if "Gas" in df.columns and "2phase" in df.columns:
        gas = df["Gas"].to_numpy(dtype=np.float32)
        two = df["2phase"].to_numpy(dtype=np.float32)
    elif "L" in df.columns:
        gas, _liq, two = phase_flags_from_L(df["L"].to_numpy(dtype=np.float32))
    else:
        raise ValueError("Provide phase info: either an 'L' column (0=gas, 1=liquid, "
                         "fraction=two-phase) or explicit 'Gas' and '2phase' flag columns.")

    z = df[["z1", "z3", "z14"]].to_numpy(dtype=np.float32)
    p = df[["p"]].to_numpy(dtype=np.float32)
    X = np.concatenate([z, p, gas[:, None], two[:, None]], axis=1).astype(np.float32)

    has_gt = all(c in df.columns for c in GT_COLUMNS) and ("L" in df.columns)
    return X, has_gt


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
def predict(model_key: str, df: pd.DataFrame, device_str: str = "cpu"
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run the ensemble. Returns (y_pred_mean, y_pred_std, X_scaled, has_gt).

    y_pred_* are unscaled [N, 7] arrays (x1,x3,x14,y1,y3,y14,L).
    X_scaled is the model input with pressure scaled (col 3), as the metric
    helpers expect.
    """
    scP, scY = get_reference_scalers()
    models = load_ensemble(model_key, device_str)
    X, has_gt = prepare_inputs(df)

    X_scaled = X.copy()
    X_scaled[:, [3]] = scP.transform(X[:, [3]])
    xt = torch.from_numpy(X_scaled).to(torch.device(device_str))

    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(scY.inverse_transform(m(xt).cpu().numpy()))
    stacked = np.stack(preds, axis=0)            # [folds, N, 7]
    y_pred = stacked.mean(axis=0)
    y_std = stacked.std(axis=0)
    return y_pred, y_std, X_scaled, has_gt


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def phase_aware_r2(y_true: np.ndarray, y_pred: np.ndarray, X_scaled: np.ndarray) -> float:
    gas = X_scaled[:, 4] > 0.5
    two = X_scaled[:, 5] > 0.5
    liq = (~gas) & (~two)
    m = np.zeros_like(y_true, dtype=bool)
    m[gas, 3:7] = True
    m[liq, 0:3] = True
    m[liq, 6] = True
    m[two, :] = True
    yt, yp = y_true[m], y_pred[m]
    sse = np.sum((yt - yp) ** 2)
    sst = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
    return float(1.0 - sse / sst)


def comp_rmse_eval(y_pred: np.ndarray, X_scaled: np.ndarray) -> float:
    two = X_scaled[:, 5] > 0.5
    if not np.any(two):
        return float("nan")
    x_pred = y_pred[two, 0:3]
    y_v = y_pred[two, 3:6]
    nx = x_pred.sum(axis=1) - 1.0
    ny = y_v.sum(axis=1) - 1.0
    return float(np.sqrt(np.mean(nx ** 2 + ny ** 2)))


def interphase_rmse_eval(y_pred: np.ndarray, X_scaled: np.ndarray) -> float:
    gas = X_scaled[:, 4] > 0.5
    two = X_scaled[:, 5] > 0.5
    liq = (~gas) & (~two)
    x_pred = y_pred[:, 0:3]
    y_v = y_pred[:, 3:6]
    L_pred = np.clip(y_pred[:, 6], 0.0, 1.0)[:, None]
    z = X_scaled[:, 0:3]
    z = z / (z.sum(axis=1, keepdims=True) + 1e-12)
    errs = []
    if np.any(two):
        z_hat = L_pred[two] * x_pred[two] + (1.0 - L_pred[two]) * y_v[two]
        errs.extend(np.sum((z_hat - z[two]) ** 2, axis=1))
    if np.any(liq):
        errs.extend(np.sum((x_pred[liq] - z[liq]) ** 2, axis=1))
    if np.any(gas):
        errs.extend(np.sum((y_v[gas] - z[gas]) ** 2, axis=1))
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(errs)))


def fug_rmse_eval(y_pred: np.ndarray, X_scaled: np.ndarray, scP: MinMaxScaler,
                  y_true: Optional[np.ndarray] = None, max_n: int = 15000,
                  gate_L: Tuple[float, float] = (0.02, 0.98), gate_xy: float = 0.05,
                  seed: int = 123,
                  pcrit: Optional[torch.Tensor] = None,
                  tcrit: Optional[torch.Tensor] = None,
                  accfac: Optional[torch.Tensor] = None,
                  deltaij: Optional[torch.Tensor] = None) -> Tuple[float, int]:
    """Weighted iso-fugacity residual on gated two-phase samples.

    Gating/weighting uses the ground truth when ``y_true`` is given, otherwise
    it falls back to the model's own predictions (so the metric is available
    for unlabeled uploads). The residual itself always uses the predictions.

    The Peng-Robinson constants (``pcrit``, ``tcrit``, ``accfac``, ``deltaij``)
    default to the C1/C3/C14 training system, but should be supplied for the
    actual components under test so the residual is thermodynamically valid.
    """
    if pcrit is None:
        pcrit, tcrit, accfac, deltaij = Pci, Tci, acc, dij
    rng = np.random.default_rng(seed)
    two = X_scaled[:, 5] > 0.5
    idx = np.where(two)[0]
    if idx.size == 0:
        return float("nan"), 0

    gate_src = y_true if y_true is not None else y_pred
    L_g = np.clip(gate_src[idx, 6], 0.0, 1.0)
    xg = gate_src[idx, 0:3]
    yg = gate_src[idx, 3:6]
    xg = xg / (xg.sum(axis=1, keepdims=True) + 1e-12)
    yg = yg / (yg.sum(axis=1, keepdims=True) + 1e-12)
    xy_norm = np.linalg.norm(xg - yg, axis=1)
    lo, hi = gate_L
    keep = (L_g > lo) & (L_g < hi) & (xy_norm > gate_xy)
    idx = idx[keep]
    if idx.size == 0:
        return float("nan"), 0
    if idx.size > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)

    L_w = np.clip(gate_src[idx, 6], 0.0, 1.0)
    w = np.clip(4.0 * L_w * (1.0 - L_w), 0.0, 1.0)

    x_pred = y_pred[idx, 0:3]
    y_v = y_pred[idx, 3:6]
    x_pred = x_pred / (x_pred.sum(axis=1, keepdims=True) + 1e-12)
    y_v = y_v / (y_v.sum(axis=1, keepdims=True) + 1e-12)
    p = X_scaled[idx, 3] * scP.data_range_[0] + scP.data_min_[0]

    TinK_t = torch.tensor(float(TinK), dtype=torch.float64)
    p_t = torch.as_tensor(p, dtype=torch.float64)
    xt = torch.as_tensor(x_pred, dtype=torch.float64)
    yt = torch.as_tensor(y_v, dtype=torch.float64)
    with torch.no_grad():
        logPhiL = _pr_logphi_batch(p_t, TinK_t, pcrit, tcrit, accfac, xt, deltaij, "liquid")
        logPhiV = _pr_logphi_batch(p_t, TinK_t, pcrit, tcrit, accfac, yt, deltaij, "vapor")
        diff = xt * torch.exp(logPhiL) - yt * torch.exp(logPhiV)
        e = torch.mean(diff ** 2, dim=1).cpu().numpy()
    return float(np.sqrt((w * e).sum() / (w.sum() + 1e-12))), int(idx.size)


def compute_metrics(y_pred: np.ndarray, X_scaled: np.ndarray, has_gt: bool,
                    y_true: Optional[np.ndarray] = None,
                    component_set: str = DEFAULT_COMPONENT_SET) -> Dict[str, object]:
    """Build the metrics dict, including only R2 when ground truth is present.

    ``component_set`` selects the Peng-Robinson constants used by the
    iso-fugacity residual, so it matches the actual components under test.
    """
    scP, _scY = get_reference_scalers()
    gt = y_true if has_gt else None
    pc, tc, ac, d = component_constants(component_set)
    fug, n_fug = fug_rmse_eval(y_pred, X_scaled, scP, y_true=gt,
                               pcrit=pc, tcrit=tc, accfac=ac, deltaij=d)
    metrics: Dict[str, object] = {
        "FugRMSE": fug,
        "CompRMSE": comp_rmse_eval(y_pred, X_scaled),
        "InterRMSE": interphase_rmse_eval(y_pred, X_scaled),
        "n_two_phase_fug": n_fug,
        "fug_gating": "ground-truth" if has_gt else "predicted",
        "fug_components": component_set,
        "has_ground_truth": has_gt,
    }
    if has_gt and y_true is not None:
        metrics["R2"] = phase_aware_r2(y_true, y_pred, X_scaled)
    return metrics


def predictions_dataframe(df_in: pd.DataFrame, y_pred: np.ndarray,
                          y_std: np.ndarray) -> pd.DataFrame:
    """Return the input frame with predicted columns (and fold std) appended."""
    pred_cols = [f"{c}_pred" for c in Y_COLUMNS]
    std_cols = [f"{c}_std" for c in Y_COLUMNS]
    out = df_in.reset_index(drop=True).copy()
    pred_df = pd.DataFrame(y_pred, columns=pred_cols)
    std_df = pd.DataFrame(y_std, columns=std_cols)
    return pd.concat([out, pred_df, std_df], axis=1)


def reconstruct_z(y_pred: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    """Reconstruct feed composition z from predictions (z = L*x + (1-L)*y)."""
    gas = X_scaled[:, 4] > 0.5
    two = X_scaled[:, 5] > 0.5
    liq = (~gas) & (~two)
    x_pred = y_pred[:, 0:3]
    y_v = y_pred[:, 3:6]
    L_pred = np.clip(y_pred[:, 6], 0.0, 1.0)[:, None]
    z = X_scaled[:, 0:3]
    z = z / (z.sum(axis=1, keepdims=True) + 1e-12)
    z_recon = z.copy()
    if np.any(two):
        z_recon[two] = L_pred[two] * x_pred[two] + (1.0 - L_pred[two]) * y_v[two]
    if np.any(liq):
        z_recon[liq] = x_pred[liq]
    if np.any(gas):
        z_recon[gas] = y_v[gas]
    return z_recon
