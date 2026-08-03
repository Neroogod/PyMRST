"""Reusable inference adapter for Victory's isothermal C1/C3/C14 flash MLPs.

This module is deliberately independent of the Streamlit web app.  It exposes
the seven-fold ensemble as an experimental value-level predictor; it does not
register as, or replace, a PyMRST EOS backend.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from pymrst.modules.compositional.eos import Mixture

MODEL_ROOT = Path(__file__).resolve().parent
WEB_ROOT = MODEL_ROOT / "Web_app"
REFERENCE_CSV = WEB_ROOT / "reference" / "test_data.csv"
TEMPERATURE_K = 353.0
INPUT_COLUMNS = ("z1", "z3", "z14", "p", "Gas", "2phase")
OUTPUT_COLUMNS = ("x1", "x3", "x14", "y1", "y3", "y14", "L")

SCHEMES = {
    "beta2.2": (MODEL_ROOT, "model_runtest_beta2.2_fold{fold}.pt"),
    "doe276": (WEB_ROOT / "models" / "doe276", "model_run276_fold{fold}.pt"),
    "relobralo": (WEB_ROOT / "models" / "relobralo", "model_runtest_fold{fold}.pt"),
    **{
        f"beta{beta}": (
            WEB_ROOT / "models" / f"beta{beta}",
            f"model_runtest_beta{beta}_fold{{fold}}.pt",
        )
        for beta in ("1", "1.2", "1.5", "2", "2.5", "3")
    },
}


@dataclass(frozen=True)
class VictoryPrediction:
    """Unscaled ensemble prediction and fold uncertainty."""

    x: np.ndarray
    y: np.ndarray
    L: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    folds: np.ndarray
    elapsed_seconds: float
    phase_flags: np.ndarray


def victory_mixture() -> Mixture:
    """C1/C3/C14 mixture used to train the supplied models.

    The value-level PR flash consumes Tc, Pc, acentric factors and BIC.  Vc and
    molar mass are supplied for a complete mixture record but do not affect the
    flash comparison.
    """

    bic = np.array(
        [[0.0, 0.00683, 0.01230], [0.00683, 0.0, 0.00082], [0.01230, 0.00082, 0.0]],
        dtype=float,
    )
    return Mixture(
        names=("Methane", "n-Propane", "n-Tetradecane"),
        Tcrit=np.array([190.58, 369.82, 723.00]),
        Pcrit=np.array([4.604e6, 4.249e6, 1.620e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4, 8.20e-4]),
        acentric=np.array([0.0108, 0.1520, 0.8000]),
        molarMass=np.array([0.0160428, 0.04409562, 0.198388]),
        bic=bic,
        name="Victory C1/C3/C14 at 353 K",
    )


def phase_flags_from_liquid_fraction(L: np.ndarray) -> np.ndarray:
    """Return ``[Gas, 2phase]`` flags using the training convention."""

    L = np.asarray(L, dtype=float).reshape(-1)
    gas = L == 0.0
    liquid = L == 1.0
    two = ~(gas | liquid)
    return np.column_stack([gas, two]).astype(np.float32)


class VictoryFlashEnsemble:
    """Load and evaluate one of Victory's seven-fold FlashNet ensembles."""

    def __init__(
        self,
        scheme: str = "beta2.2",
        *,
        device: str = "cpu",
        reference_csv: str | Path = REFERENCE_CSV,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Victory inference requires Torch; install PyMRST with .[ml]") from exc
        if scheme not in SCHEMES:
            raise ValueError(f"unknown Victory scheme {scheme!r}; choose from {sorted(SCHEMES)}")
        self.scheme = scheme
        self.device = torch.device(device)
        self.reference_csv = Path(reference_csv)
        self._torch = torch
        self._nn = nn
        self.pressure_min, self.pressure_max, self.output_min, self.output_max = (
            _reference_scaling_bounds(self.reference_csv)
        )
        self.models = self._load_models()

    def _new_model(self):
        nn = self._nn
        layers = []
        dim = 6
        for _ in range(4):
            layers.extend([nn.Linear(dim, 256), nn.ReLU()])
            dim = 256
        layers.extend([nn.Linear(dim, 7), nn.Sigmoid()])
        return nn.Sequential(*layers)

    def _load_models(self) -> tuple:
        torch = self._torch
        folder, template = SCHEMES[self.scheme]
        models = []
        for fold in range(1, 8):
            path = folder / template.format(fold=fold)
            if not path.is_file():
                raise FileNotFoundError(f"missing Victory fold checkpoint: {path}")
            model = self._new_model().to(self.device)
            state = torch.load(path, map_location=self.device, weights_only=True)
            # Web-app checkpoints use a wrapper named `net`; this adapter's
            # Sequential is the wrapper itself, so remove that stable prefix.
            state = {k.removeprefix("net."): v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        return tuple(models)

    @property
    def pressure_range(self) -> tuple[float, float]:
        return self.pressure_min, self.pressure_max

    def predict(
        self,
        pressure: np.ndarray,
        z: np.ndarray,
        *,
        liquid_fraction_for_flags: np.ndarray | None = None,
        phase_flags: np.ndarray | None = None,
        temperature: float | np.ndarray = TEMPERATURE_K,
        strict_domain: bool = True,
    ) -> VictoryPrediction:
        """Evaluate all folds and return their unscaled mean and spread.

        Victory's network requires externally supplied gas/two-phase flags.
        Exactly one of ``liquid_fraction_for_flags`` or ``phase_flags`` must be
        given.  Temperature is checked even though it is not a network input.
        """

        p, z = _validate_states(pressure, z, temperature, self.pressure_range, strict_domain)
        if (liquid_fraction_for_flags is None) == (phase_flags is None):
            raise ValueError("provide exactly one of liquid_fraction_for_flags or phase_flags")
        if phase_flags is None:
            flags = phase_flags_from_liquid_fraction(liquid_fraction_for_flags)
        else:
            flags = np.asarray(phase_flags, dtype=np.float32)
            if flags.shape != (p.size, 2):
                raise ValueError(f"phase_flags must have shape {(p.size, 2)}, got {flags.shape}")
            if np.any((flags != 0.0) & (flags != 1.0)) or np.any(flags.sum(axis=1) > 1.0):
                raise ValueError("phase_flags must be valid binary [Gas, 2phase] rows")
        if flags.shape[0] != p.size:
            raise ValueError("phase flag count must match pressure rows")

        pscaled = (p - self.pressure_min) / (self.pressure_max - self.pressure_min)
        inputs = np.column_stack([z, pscaled, flags]).astype(np.float32)
        xt = self._torch.from_numpy(inputs).to(self.device)
        predictions = []
        start = perf_counter()
        with self._torch.no_grad():
            for model in self.models:
                scaled = model(xt).detach().cpu().numpy()
                predictions.append(scaled * (self.output_max - self.output_min) + self.output_min)
        elapsed = perf_counter() - start
        folds = np.stack(predictions, axis=0)
        mean = folds.mean(axis=0)
        std = folds.std(axis=0)
        return VictoryPrediction(
            x=mean[:, :3],
            y=mean[:, 3:6],
            L=mean[:, 6],
            mean=mean,
            std=std,
            folds=folds,
            elapsed_seconds=elapsed,
            phase_flags=flags,
        )


def _reference_scaling_bounds(path: Path) -> tuple[float, float, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Victory reference scaler data not found: {path}")
    output_names = list(OUTPUT_COLUMNS)
    out_min = np.full(7, np.inf, dtype=np.float32)
    out_max = np.full(7, -np.inf, dtype=np.float32)
    pmin = np.float32(np.inf)
    pmax = np.float32(-np.inf)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            L = np.float32(row["L"])
            values = np.array([row[name] for name in output_names], dtype=np.float32)
            if L == 0.0:
                values[:3] = 0.0
            elif L == 1.0:
                values[3:6] = 0.0
            out_min = np.minimum(out_min, values)
            out_max = np.maximum(out_max, values)
            pv = np.float32(row["p"])
            pmin = min(pmin, pv)
            pmax = max(pmax, pv)
    return float(pmin), float(pmax), out_min, out_max


def _validate_states(pressure, z, temperature, pressure_range, strict_domain):
    p = np.asarray(pressure, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if z.shape != (p.size, 3):
        raise ValueError(f"z must have shape {(p.size, 3)}, got {z.shape}")
    T = np.asarray(temperature, dtype=float)
    T = np.broadcast_to(T, p.shape)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(T)):
        raise ValueError("pressure, temperature and composition must be finite")
    if np.any(p <= 0.0) or np.any(z < 0.0):
        raise ValueError("pressure must be positive and compositions nonnegative")
    if not np.allclose(z.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-6):
        raise ValueError("feed compositions must sum to one")
    if not np.allclose(T, TEMPERATURE_K, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"Victory models are isothermal and support only {TEMPERATURE_K:g} K")
    # Bounds originate from sklearn fitted to float32 reference columns. Allow
    # one small float32-scale margin so the original float64 CSV extrema are
    # not rejected after the scaler bounds round inward.
    pressure_margin = max(1.0, 2.0e-6 * (pressure_range[1] - pressure_range[0]))
    if strict_domain and (
        np.any(p < pressure_range[0] - pressure_margin)
        or np.any(p > pressure_range[1] + pressure_margin)
    ):
        raise ValueError(f"pressure lies outside trained/reference range {pressure_range} Pa")
    return p, z


def load_reference_rows(
    path: str | Path = REFERENCE_CSV,
    *,
    rows: int | None = None,
    seed: int = 42,
    phase: str = "all",
) -> dict[str, np.ndarray]:
    """Load a deterministic reference subset without requiring pandas."""

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding=None)
    phase_masks = {
        "all": np.ones(data.size, dtype=bool),
        "gas": data["L"] == 0.0,
        "liquid": data["L"] == 1.0,
        "two-phase": (data["L"] > 0.0) & (data["L"] < 1.0),
    }
    if phase not in phase_masks:
        raise ValueError(f"unknown phase {phase!r}; choose from {tuple(phase_masks)}")
    indices = np.flatnonzero(phase_masks[phase])
    if rows is not None and rows < indices.size:
        indices = np.sort(np.random.default_rng(seed).choice(indices, size=rows, replace=False))
    d = data[indices]
    return {
        "index": indices.astype(np.int64),
        "p": np.asarray(d["p"], dtype=float),
        "T": np.full(indices.size, TEMPERATURE_K),
        "z": np.column_stack([d["z1"], d["z3"], d["z14"]]),
        "L": np.asarray(d["L"], dtype=float),
        "x": np.column_stack([d["x1"], d["x3"], d["x14"]]),
        "y": np.column_stack([d["y1"], d["y3"], d["y14"]]),
        "Z_L": np.asarray(d["Z_L"], dtype=float),
        "Z_V": np.asarray(d["Z_V"], dtype=float),
    }
