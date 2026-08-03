"""Compare Victory's ensemble with PyMRST, MATLAB MRST and reference flashes.

Run ``python3 -m pymrst.modules.compositional.ML_models.compare_with_pymrst
--help`` for the experimental CLI.  This module never changes the production
flash dispatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from pymrst.modules.compositional.eos import flash, fugacity_coefficients
from pymrst.modules.compositional.ML_models.victory_flash import (
    OUTPUT_COLUMNS,
    TEMPERATURE_K,
    VictoryFlashEnsemble,
    load_reference_rows,
    victory_mixture,
)

PHASE_NAMES = np.array(["gas", "liquid", "two-phase"])


class ConsoleProgress:
    """Dependency-free terminal progress bar for the comparison stages."""

    def __init__(self, enabled: bool = True, width: int = 28) -> None:
        self.enabled = enabled
        self.width = width
        self.started = perf_counter()

    def stage(self, label: str) -> None:
        self.started = perf_counter()
        if self.enabled:
            print(f"\n{label}", flush=True)

    def update(self, completed: int, total: int, label: str = "") -> None:
        if not self.enabled:
            return
        total = max(int(total), 1)
        completed = min(max(int(completed), 0), total)
        fraction = completed / total
        filled = int(self.width * fraction)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = perf_counter() - self.started
        eta = elapsed * (total - completed) / completed if completed else float("nan")
        eta_text = _format_duration(eta) if np.isfinite(eta) else "--"
        suffix = f" {label}" if label else ""
        print(
            f"\r[{bar}] {100*fraction:6.2f}%  {completed:,}/{total:,}  "
            f"elapsed {_format_duration(elapsed)}  ETA {eta_text}{suffix}",
            end="\n" if completed == total else "",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass(frozen=True)
class FlashValues:
    name: str
    L: np.ndarray
    x: np.ndarray
    y: np.ndarray
    Z_L: np.ndarray | None
    Z_V: np.ndarray | None
    phase: np.ndarray
    elapsed_seconds: float | None = None
    uncertainty: np.ndarray | None = None

    @property
    def outputs(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.L])


def phase_codes(L: np.ndarray, tolerance: float = 1.0e-8) -> np.ndarray:
    L = np.asarray(L, dtype=float).reshape(-1)
    return np.where(L <= tolerance, 0, np.where(L >= 1.0 - tolerance, 1, 2))


def reference_values(data: dict[str, np.ndarray]) -> FlashValues:
    return FlashValues(
        "reference", data["L"], data["x"], data["y"], data["Z_L"], data["Z_V"],
        phase_codes(data["L"], tolerance=0.0),
    )


def run_victory(data: dict[str, np.ndarray], scheme="beta2.2", device="cpu"):
    model = VictoryFlashEnsemble(scheme, device=device)
    # The supplied reference phase label is an input to this regression model.
    # This is intentionally explicit and disclosed in every generated report.
    model.predict(
        data["p"][:1], data["z"][:1], liquid_fraction_for_flags=data["L"][:1]
    )  # warm-up
    prediction = model.predict(
        data["p"], data["z"], liquid_fraction_for_flags=data["L"]
    )
    values = FlashValues(
        "victory", prediction.L, prediction.x, prediction.y, None, None,
        phase_codes(prediction.L), prediction.elapsed_seconds, prediction.std,
    )
    return values, model, prediction


def run_pymrst(
    data: dict[str, np.ndarray],
    *,
    batch_size: int = 1000,
    progress: ConsoleProgress | None = None,
) -> tuple[FlashValues, int]:
    mixture = victory_mixture()
    flash(mixture, data["p"][:1], data["T"][:1], data["z"][:1])  # warm-up
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = data["p"].size
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ("V", "x", "y", "Z_L", "Z_V")}
    max_iterations = 0
    start = perf_counter()
    for lower in range(0, count, batch_size):
        upper = min(lower + batch_size, count)
        result = flash(
            mixture,
            data["p"][lower:upper],
            data["T"][lower:upper],
            data["z"][lower:upper],
        )
        for name in chunks:
            chunks[name].append(np.asarray(getattr(result, name)))
        max_iterations = max(max_iterations, int(result.iterations))
        if progress is not None:
            progress.update(upper, count, "PyMRST flash")
    elapsed = perf_counter() - start
    combined = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    L = 1.0 - combined["V"]
    values = FlashValues(
        "pymrst", L, combined["x"], combined["y"], combined["Z_L"], combined["Z_V"],
        phase_codes(L), elapsed,
    )
    return values, max_iterations


def state_digest(data: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("index", "p", "T", "z"):
        a = np.ascontiguousarray(data[key])
        digest.update(key.encode("ascii"))
        digest.update(str(a.dtype).encode("ascii"))
        digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        digest.update(a.tobytes())
    return digest.hexdigest()


def export_mrst_input(path: str | Path, data: dict[str, np.ndarray]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mix = victory_mixture()
    digest = state_digest(data)
    savemat(
        path,
        {
            "p": data["p"].reshape(-1, 1),
            "T": data["T"].reshape(-1, 1),
            "z": data["z"],
            "reference_L": data["L"].reshape(-1, 1),
            "reference_x": data["x"],
            "reference_y": data["y"],
            "reference_Z_L": data["Z_L"].reshape(-1, 1),
            "reference_Z_V": data["Z_V"].reshape(-1, 1),
            "sample_indices_zero_based": data["index"].reshape(-1, 1),
            "component_names": np.asarray(mix.names, dtype=object),
            "Tcrit": mix.Tcrit.reshape(1, -1),
            "Pcrit": mix.Pcrit.reshape(1, -1),
            "Vcrit": mix.Vcrit.reshape(1, -1),
            "acentric": mix.acentric.reshape(1, -1),
            "molar_mass": mix.molarMass.reshape(1, -1),
            "bic": mix.bic,
            "input_digest": digest,
            "schema_version": "victory-flash-comparison-v1",
        },
        do_compression=True,
    )
    return digest


def load_mrst_values(path: str | Path, expected_digest: str, expected_rows: int) -> FlashValues:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    digest = str(raw.get("input_digest", "")).strip()
    if digest != expected_digest:
        raise ValueError(f"MRST input digest mismatch: expected {expected_digest}, got {digest}")
    L = np.asarray(raw["L"], dtype=float).reshape(-1)
    x = np.asarray(raw["x"], dtype=float).reshape(-1, 3)
    y = np.asarray(raw["y"], dtype=float).reshape(-1, 3)
    if L.size != expected_rows or x.shape != (expected_rows, 3) or y.shape != (expected_rows, 3):
        raise ValueError("MRST result shapes do not match the exported input batch")
    if not np.all(np.isfinite(np.column_stack([x, y, L]))):
        raise ValueError("MRST result contains nonfinite flash outputs")
    return FlashValues(
        "mrst", L, x, y,
        np.asarray(raw["Z_L"], dtype=float).reshape(-1),
        np.asarray(raw["Z_V"], dtype=float).reshape(-1),
        phase_codes(L), float(np.asarray(raw.get("flash_elapsed_seconds", np.nan)).reshape(())),
    )


def scalar_errors(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return {k: float("nan") for k in ("bias", "mae", "rmse", "median_ae", "p95_ae", "p99_ae", "max_ae", "r2")} | {"count": 0}
    delta = a[valid] - b[valid]
    absolute = np.abs(delta)
    denom = np.sum((b[valid] - np.mean(b[valid])) ** 2)
    return {
        "count": int(valid.sum()),
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "median_ae": float(np.median(absolute)),
        "p95_ae": float(np.quantile(absolute, 0.95)),
        "p99_ae": float(np.quantile(absolute, 0.99)),
        "max_ae": float(np.max(absolute)),
        "r2": float(1.0 - np.sum(delta * delta) / denom) if denom > 0.0 else float("nan"),
    }


def _assessment_phase(values: FlashValues, reference_phase: np.ndarray) -> np.ndarray:
    # Victory's class is the supplied reference class, not a prediction.
    return reference_phase if values.name == "reference" or values.name.startswith("victory") else values.phase


def pair_metrics(a: FlashValues, b: FlashValues, reference_phase: np.ndarray, subset=None) -> list[dict[str, Any]]:
    rows = []
    phase_a = _assessment_phase(a, reference_phase)
    phase_b = _assessment_phase(b, reference_phase)
    selected = np.ones(reference_phase.size, dtype=bool) if subset is None else np.asarray(subset, bool)
    for column, name in enumerate(OUTPUT_COLUMNS):
        if name.startswith("x"):
            mask = selected & (phase_a != 0) & (phase_b != 0)
        elif name.startswith("y"):
            mask = selected & (phase_a != 1) & (phase_b != 1)
        else:
            mask = selected
        rows.append({"left": a.name, "right": b.name, "quantity": name, **scalar_errors(a.outputs[mask, column], b.outputs[mask, column])})
    if a.Z_L is not None and b.Z_L is not None:
        mask = reference_phase != 0
        rows.append({"left": a.name, "right": b.name, "quantity": "Z_L", **scalar_errors(a.Z_L[mask], b.Z_L[mask])})
    if a.Z_V is not None and b.Z_V is not None:
        mask = reference_phase != 1
        rows.append({"left": a.name, "right": b.name, "quantity": "Z_V", **scalar_errors(a.Z_V[mask], b.Z_V[mask])})
    return rows


def phase_confusion(actual: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    return [[int(np.sum((actual == i) & (predicted == j))) for j in range(3)] for i in range(3)]


def physical_metrics(values: FlashValues, data: dict[str, np.ndarray]) -> dict[str, Any]:
    # Victory is a phase-conditional regressor and must be assessed under the
    # externally supplied reference class. PyMRST/MRST classify independently,
    # so their own phase result defines which composition is physically present.
    classification = phase_codes(data["L"], tolerance=0.0) if values.name in {"reference", "victory"} else values.phase
    gas, liquid, two = classification == 0, classification == 1, classification == 2
    zhat = data["z"].copy()
    zhat[two] = values.L[two, None] * values.x[two] + (1.0 - values.L[two, None]) * values.y[two]
    zhat[liquid] = values.x[liquid]
    zhat[gas] = values.y[gas]
    out = values.outputs
    result: dict[str, Any] = {
        "nonfinite_count": int(np.size(out) - np.isfinite(out).sum()),
        "out_of_bounds_count": int(np.sum((out < 0.0) | (out > 1.0))),
        "x_closure_rmse": float(np.sqrt(np.mean((values.x.sum(axis=1) - 1.0) ** 2))),
        "y_closure_rmse": float(np.sqrt(np.mean((values.y.sum(axis=1) - 1.0) ** 2))),
        "material_balance_rmse": float(np.sqrt(np.mean((zhat - data["z"]) ** 2))),
        "material_balance_max": float(np.max(np.abs(zhat - data["z"]))),
        "phase_counts": {name: int(np.sum(classification == i)) for i, name in enumerate(PHASE_NAMES)},
    }
    if np.any(two):
        mix = victory_mixture()
        xl = values.x[two] / values.x[two].sum(axis=1, keepdims=True)
        yv = values.y[two] / values.y[two].sum(axis=1, keepdims=True)
        lnl, _ = fugacity_coefficients(mix, data["p"][two], data["T"][two], xl, vapour=False)
        lnv, _ = fugacity_coefficients(mix, data["p"][two], data["T"][two], yv, vapour=True)
        logf_delta = np.log(np.maximum(xl, 1e-300)) + lnl - np.log(np.maximum(yv, 1e-300)) - lnv
        result["log_fugacity_rmse"] = float(np.sqrt(np.mean(logf_delta * logf_delta)))
        result["log_fugacity_max"] = float(np.max(np.abs(logf_delta)))
    else:
        result["log_fugacity_rmse"] = result["log_fugacity_max"] = float("nan")
    return result


def compare_all(methods: list[FlashValues], data: dict[str, np.ndarray]):
    reference_phase = phase_codes(data["L"], tolerance=0.0)
    pairs = []
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            pairs.extend(pair_metrics(left, right, reference_phase))
    return pairs


def phase_stratified_metrics(methods, data):
    reference_phase = phase_codes(data["L"], tolerance=0.0)
    rows = []
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            for code, phase_name in enumerate(PHASE_NAMES):
                for row in pair_metrics(left, right, reference_phase, reference_phase == code):
                    rows.append({"reference_phase": phase_name, **row})
    return rows


def worst_cases(methods, data, limit=10):
    reference_phase = phase_codes(data["L"], tolerance=0.0)
    rows = []
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            phase_left = _assessment_phase(left, reference_phase)
            phase_right = _assessment_phase(right, reference_phase)
            for column, quantity in enumerate(OUTPUT_COLUMNS):
                mask = np.ones(reference_phase.size, dtype=bool)
                if quantity.startswith("x"):
                    mask &= (phase_left != 0) & (phase_right != 0)
                elif quantity.startswith("y"):
                    mask &= (phase_left != 1) & (phase_right != 1)
                candidates = np.flatnonzero(mask)
                if candidates.size == 0:
                    continue
                errors = np.abs(left.outputs[candidates, column] - right.outputs[candidates, column])
                order = candidates[np.argsort(errors)[-limit:][::-1]]
                for rank, row_index in enumerate(order, 1):
                    rows.append({
                        "left": left.name, "right": right.name, "quantity": quantity,
                        "rank": rank, "sample_index": int(data["index"][row_index]),
                        "p_Pa": float(data["p"][row_index]), "z1": float(data["z"][row_index, 0]),
                        "z3": float(data["z"][row_index, 1]), "z14": float(data["z"][row_index, 2]),
                        "reference_phase": str(PHASE_NAMES[reference_phase[row_index]]),
                        "left_value": float(left.outputs[row_index, column]),
                        "right_value": float(right.outputs[row_index, column]),
                        "absolute_error": float(abs(left.outputs[row_index, column] - right.outputs[row_index, column])),
                    })
    return rows


def fold_metrics(victory_prediction, reference: FlashValues, reference_phase):
    rows = []
    for fold_index, outputs in enumerate(victory_prediction.folds, 1):
        fold = FlashValues(
            f"victory_fold{fold_index}", outputs[:, 6], outputs[:, :3], outputs[:, 3:6],
            None, None, reference_phase,
        )
        for row in pair_metrics(reference, fold, reference_phase):
            rows.append({"fold": fold_index, **row})
    return rows


def uncertainty_error_correlation(victory: FlashValues, reference: FlashValues, reference_phase):
    result = {}
    if victory.uncertainty is None:
        return result
    for column, quantity in enumerate(OUTPUT_COLUMNS):
        mask = np.ones(reference_phase.size, dtype=bool)
        if quantity.startswith("x"):
            mask &= reference_phase != 0
        elif quantity.startswith("y"):
            mask &= reference_phase != 1
        uncertainty = victory.uncertainty[mask, column]
        error = np.abs(victory.outputs[mask, column] - reference.outputs[mask, column])
        if uncertainty.size > 1 and np.std(uncertainty) > 0 and np.std(error) > 0:
            result[quantity] = float(np.corrcoef(uncertainty, error)[0, 1])
        else:
            result[quantity] = float("nan")
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_reports(output: Path, data, methods, pair_rows, phase_rows, worst_rows, fold_rows, summary, include_predictions=True):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
    keys = ["left", "right", "quantity", "count", "bias", "mae", "rmse", "median_ae", "p95_ae", "p99_ae", "max_ae", "r2"]
    with (output / "per_output_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(pair_rows)
    with (output / "phase_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reference_phase", *keys])
        writer.writeheader(); writer.writerows(phase_rows)
    worst_keys = ["left", "right", "quantity", "rank", "sample_index", "p_Pa", "z1", "z3", "z14", "reference_phase", "left_value", "right_value", "absolute_error"]
    with (output / "worst_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=worst_keys)
        writer.writeheader(); writer.writerows(worst_rows)
    with (output / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", *keys])
        writer.writeheader(); writer.writerows(fold_rows)
    if include_predictions:
        columns = ["index", "p", "T", "z1", "z3", "z14", "reference_phase"]
        arrays = [data["index"], data["p"], data["T"], *data["z"].T, PHASE_NAMES[phase_codes(data["L"], 0.0)]]
        for method in methods:
            for j, name in enumerate(OUTPUT_COLUMNS):
                columns.append(f"{method.name}_{name}"); arrays.append(method.outputs[:, j])
            if method.uncertainty is not None:
                for j, name in enumerate(OUTPUT_COLUMNS):
                    columns.append(f"{method.name}_{name}_std"); arrays.append(method.uncertainty[:, j])
        with (output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(columns); writer.writerows(zip(*arrays))
    lines = ["# Victory / PyMRST / MRST flash comparison", "", f"Rows: {len(data['p'])}", "", "> Victory receives reference-derived Gas/2phase flags; this experiment does not measure Victory stability classification.", "", "## Timings", ""]
    for method in methods:
        if method.elapsed_seconds is not None:
            lines.append(f"- {method.name}: {method.elapsed_seconds:.6g} s")
    lines.extend(["", "## Physical consistency", ""])
    for name, metrics in summary["physical_metrics"].items():
        lines.append(f"- {name}: material RMSE={metrics['material_balance_rmse']:.6g}, log-fugacity RMSE={metrics['log_fugacity_rmse'] if metrics['log_fugacity_rmse'] is not None else 'n/a'}")
    lines.extend(["", "See `per_output_metrics.csv`, `predictions.csv`, and `summary.json` for complete results.", ""])
    (output / "comparison.md").write_text("\n".join(lines))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="beta2.2")
    parser.add_argument("--rows", default="10000", help="integer sample count or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=("all", "gas", "liquid", "two-phase"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--mrst-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000, help="PyMRST states per progress batch (default: 1000)")
    parser.add_argument("--no-progress", action="store_true", help="disable terminal stage and progress output")
    parser.add_argument("--no-predictions", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    progress = ConsoleProgress(not args.no_progress)
    count = None if args.rows.lower() == "all" else int(args.rows)
    progress.stage("[1/6] Loading the bundled reference states")
    load_kwargs = {"rows": count, "seed": args.seed, "phase": args.phase}
    if args.reference_csv is not None:
        load_kwargs["path"] = args.reference_csv
    data = load_reference_rows(**load_kwargs)
    if data["p"].size == 0:
        raise ValueError("selected reference subset is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    progress.update(len(data["p"]), len(data["p"]), "reference states loaded")
    progress.stage("[2/6] Exporting the digest-protected MRST input fixture")
    digest = export_mrst_input(args.output / "inputs.mat", data)
    progress.update(1, 1, "inputs.mat written")
    ref = reference_values(data)
    progress.stage("[3/6] Running the seven-fold Victory ensemble")
    victory, ensemble, ensemble_prediction = run_victory(data, args.model, args.device)
    progress.update(len(data["p"]), len(data["p"]), "7/7 folds complete")
    progress.stage(f"[4/6] Running PyMRST in batches of {args.batch_size:,} states")
    pymrst, pymrst_iterations = run_pymrst(
        data, batch_size=args.batch_size, progress=progress
    )
    methods = [ref, victory, pymrst]
    if args.mrst_results is not None:
        methods.append(load_mrst_values(args.mrst_results, digest, len(data["p"])))
    progress.stage("[5/6] Calculating phase-aware errors and physical residuals")
    pair_rows = compare_all(methods, data)
    phase_rows = phase_stratified_metrics(methods, data)
    worst_rows = worst_cases(methods, data)
    fold_rows = fold_metrics(ensemble_prediction, ref, ref.phase)
    physical = {method.name: physical_metrics(method, data) for method in methods}
    progress.update(len(methods), len(methods), f"{len(methods)} methods evaluated")
    ref_phase = ref.phase
    summary = {
        "schema_version": "victory-flash-comparison-v1",
        "rows": int(len(data["p"])),
        "seed": args.seed,
        "phase_filter": args.phase,
        "input_digest": digest,
        "model": args.model,
        "folds": len(ensemble.models),
        "device": args.device,
        "temperature_K": TEMPERATURE_K,
        "pressure_range_Pa": [float(data["p"].min()), float(data["p"].max())],
        "victory_phase_flags_from_reference": True,
        "victory_uncertainty_error_correlation": uncertainty_error_correlation(victory, ref, ref.phase),
        "timings_seconds": {m.name: m.elapsed_seconds for m in methods},
        "pymrst_flash_iterations": pymrst_iterations,
        # Victory is excluded: it receives the reference class as an input, so
        # reporting its inferred-L threshold as classification accuracy would
        # be circular and misleading.
        "phase_confusion_vs_reference": {m.name: phase_confusion(ref_phase, m.phase) for m in methods if m.name not in {"reference", "victory"}},
        "physical_metrics": physical,
        "python": sys.version,
        "platform": platform.platform(),
        "mrst_included": any(m.name == "mrst" for m in methods),
    }
    progress.stage("[6/6] Writing comparison artifacts")
    write_reports(args.output, data, methods, pair_rows, phase_rows, worst_rows, fold_rows, summary, not args.no_predictions)
    progress.update(1, 1, "reports complete")
    print(json.dumps(_json_safe(summary), indent=2))
    print(f"MRST input fixture: {args.output / 'inputs.mat'}")
    if not summary["mrst_included"]:
        print("MRST results not supplied; run scripts/matlab/run_mrst_victory_flash_reference.m and rerun with --mrst-results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
