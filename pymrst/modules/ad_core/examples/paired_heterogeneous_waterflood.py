"""Runnable 3D heterogeneous waterflood paired with an MRST MATLAB example.

The matching MATLAB script is ``pairedHeterogeneousWaterflood.m`` in MRST's
``modules/ad-core/examples``. Both cases use the same grid, deterministic rock,
fluid, wells, controls, and report steps. Results are saved with matching array
names so they can be compared without translating simulator-specific objects.

Run from the PyMRST repository root:

    python pymrst/modules/ad_core/examples/paired_heterogeneous_waterflood.py

The default output is ``paired_waterflood_pymrst_results.npz`` in the current
directory. Use ``--output PATH`` to choose another location. After running the
MATLAB counterpart, validate every common result array with:

    python pymrst/modules/ad_core/examples/paired_heterogeneous_waterflood.py \
        --compare-matlab paired_waterflood_mrst_results.mat
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core import simulate_schedule_ad
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


@dataclass(frozen=True)
class WaterfloodResult:
    """Portable numerical output from the paired waterflood."""

    time_days: np.ndarray
    pressure: np.ndarray
    sw: np.ndarray
    qws: np.ndarray
    qos: np.ndarray
    bhp: np.ndarray
    newton_iterations: np.ndarray
    grid_dims: np.ndarray
    cell_centroids: np.ndarray
    permeability_md: np.ndarray
    porosity: np.ndarray


def _column_cells(nx: int, ny: int, nz: int, i: int, j: int) -> np.ndarray:
    """Zero-based cells in the vertical column ``(i, j)``."""
    return i + nx * j + nx * ny * np.arange(nz, dtype=np.int64)


def build_paired_waterflood():
    """Build the common PyMRST/MRST model, initial state, and timesteps."""
    nx, ny, nz = 15, 15, 3
    g = compute_geometry(
        cart_grid([nx, ny, nz], [750.0, 750.0, 30.0])
    )

    # Deterministic geology: layered permeability plus smooth areal channels.
    cell = np.arange(g.cells.num, dtype=np.int64)
    i = cell % nx
    j = (cell // nx) % ny
    k = cell // (nx * ny)
    layer_multiplier = np.array([1.0, 0.35, 1.8])[k]
    channel = 1.0 + 0.55 * np.sin((i + 1) * 0.63) * np.cos((j + 1) * 0.41)
    permeability_md = np.clip(120.0 * layer_multiplier * channel, 15.0, None)
    porosity = np.clip(
        0.16 + 0.055 * (permeability_md / permeability_md.max()), 0.16, 0.24
    )
    rock = make_rock(
        g, (permeability_md * MILLI * DARCY)[:, None], porosity[:, None]
    )

    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 4.0]) * CENTI * POISE,
        rho=[1000.0, 780.0],
        n=[2, 2],
        c=[1.0e-6 / BARSA, 8.0e-5 / BARSA],
        p_ref=250.0 * BARSA,
    )

    injector_cells = _column_cells(nx, ny, nz, 1, 1)
    producer_cells = _column_cells(nx, ny, nz, nx - 2, ny - 2)
    injection_rate = 180.0 / DAY  # 180 reservoir m^3/day
    wells = add_well(
        None, g, rock, injector_cells,
        well_type="rate", val=injection_rate,
        compi=np.array([1.0, 0.0]), sign=1, name="INJ",
    )
    wells = add_well(
        wells, g, rock, producer_cells,
        well_type="bhp", val=180.0 * BARSA,
        compi=np.array([0.0, 1.0]), sign=-1, name="PROD",
    )

    model = TwoPhaseOilWaterModelAD(g, rock, fluid, wells)
    state0 = model.initial_state(250.0 * BARSA, 0.15)
    timesteps = np.full(24, 30.0 * DAY)
    return g, rock, model, state0, timesteps, permeability_md, porosity


def run_paired_waterflood() -> WaterfloodResult:
    """Run two years of fully implicit waterflood simulation."""
    (
        g, _rock, model, state0, timesteps, permeability_md, porosity,
    ) = build_paired_waterflood()
    states, iterations = simulate_schedule_ad(state0, model, timesteps)
    if len(states) != len(timesteps):
        raise RuntimeError(
            f"simulation returned {len(states)} of {len(timesteps)} states"
        )

    return WaterfloodResult(
        time_days=np.cumsum(timesteps) / DAY,
        pressure=np.vstack([state.pressure for state in states]),
        sw=np.vstack([state.sW for state in states]),
        qws=np.vstack([state.qWs for state in states]),
        qos=np.vstack([state.qOs for state in states]),
        bhp=np.vstack([state.bhp for state in states]),
        newton_iterations=np.asarray(iterations, dtype=np.int64),
        grid_dims=np.array([15, 15, 3], dtype=np.int64),
        cell_centroids=np.asarray(g.cells.centroids, dtype=float),
        permeability_md=permeability_md,
        porosity=porosity,
    )


def save_result(result: WaterfloodResult, output: str | Path) -> Path:
    """Save every result field as a named NumPy array."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **vars(result))
    return output


def compare_with_mrst(
    result: WaterfloodResult, matlab_file: str | Path
) -> bool:
    """Compare the portable histories with the paired MATLAB result file."""
    from scipy.io import loadmat

    reference = loadmat(Path(matlab_file).expanduser().resolve(), squeeze_me=True)
    print("PyMRST/MRST full-history comparison")
    ok = True
    for name in (
        "time_days", "pressure", "sw", "qws", "qos", "bhp",
        "grid_dims", "cell_centroids", "permeability_md", "porosity",
    ):
        actual = np.asarray(getattr(result, name), dtype=float).squeeze()
        expected = np.asarray(reference[name], dtype=float).squeeze()
        same_shape = actual.shape == expected.shape
        if same_shape:
            error = float(np.max(np.abs(actual - expected)))
            scale = max(float(np.max(np.abs(expected))), 1.0e-300)
            passed = np.allclose(actual, expected, rtol=1.0e-8, atol=1.0e-10)
            print(
                f"  {'ok  ' if passed else 'FAIL'} {name:18s}"
                f" max abs {error:.3e}, max relative {error / scale:.3e}"
            )
        else:
            passed = False
            print(
                f"  FAIL {name:18s} shape {actual.shape}, "
                f"MRST shape {expected.shape}"
            )
        ok &= passed

    mrst_iterations = np.asarray(
        reference["newton_iterations"], dtype=np.int64
    ).ravel()
    print(
        "  info Newton iterations: "
        f"PyMRST {result.newton_iterations.sum()}, "
        f"MRST {mrst_iterations.sum()}"
    )
    return bool(ok)


def _print_summary(result: WaterfloodResult) -> None:
    print("Paired heterogeneous 3D waterflood (PyMRST)")
    print(" step   day    avg pressure [bar]   avg Sw   producer oil [m3/day]")
    for step in range(result.time_days.size):
        print(
            f"{step + 1:5d} {result.time_days[step]:6.0f}"
            f" {result.pressure[step].mean() / BARSA:21.3f}"
            f" {result.sw[step].mean():8.4f}"
            f" {-result.qos[step, 1] * DAY:23.3f}"
        )
    print(
        f"Final pressure range: "
        f"{result.pressure[-1].min() / BARSA:.2f}--"
        f"{result.pressure[-1].max() / BARSA:.2f} bar"
    )
    print(
        f"Final water saturation range: "
        f"{result.sw[-1].min():.4f}--{result.sw[-1].max():.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="paired_waterflood_pymrst_results.npz",
        help="destination .npz file (default: current directory)",
    )
    parser.add_argument(
        "--compare-matlab",
        metavar="MAT_FILE",
        help="compare all common histories with the paired MRST .mat result",
    )
    args = parser.parse_args(argv)
    result = run_paired_waterflood()
    path = save_result(result, args.output)
    _print_summary(result)
    print(f"Saved full result to {path}")
    if args.compare_matlab and not compare_with_mrst(
        result, args.compare_matlab
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
