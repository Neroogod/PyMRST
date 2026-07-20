"""Phase 233: showNaturalOverall2 runtime plot parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.book_ii.show_natural_overall2 import (  # noqa: E402
    run_show_natural_overall2,
    show_natural_overall2_runtime_plot,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    out = run_show_natural_overall2()
    plot = out["runtime_plot"]
    helper = show_natural_overall2_runtime_plot(out["iterations"], out["wall_time"])

    print("\nPhase 233: showNaturalOverall2 runtime plot parity")
    failures = 0
    failures += _check("runtime vectors have Natural/Overall rows",
                       out["iterations"].shape == (2,)
                       and out["wall_time"].shape == (2,))
    failures += _check("runtime plot uses MRST subplot layout",
                       plot["layout"] == (1, 2)
                       and plot["iterations"]["subplot"] == (1, 2, 1)
                       and plot["wall_time"]["subplot"] == (1, 2, 2))
    failures += _check("runtime plot uses MRST labels and titles",
                       plot["x_tick_labels"] == ("Natural", "Overall")
                       and plot["iterations"]["title"] == "Iterations"
                       and plot["wall_time"]["title"] == "Wall time")
    failures += _check("runtime plot bars match report values",
                       np.allclose(plot["iterations"]["bar"], out["iterations"])
                       and np.allclose(plot["wall_time"]["bar"], out["wall_time"])
                       and np.all(plot["wall_time"]["bar"] >= 0.0))
    failures += _check("standalone plot helper matches runner payload",
                       helper["x_tick_labels"] == plot["x_tick_labels"]
                       and np.allclose(helper["iterations"]["bar"], plot["iterations"]["bar"])
                       and np.allclose(helper["wall_time"]["bar"], plot["wall_time"]["bar"]))

    if failures:
        print(f"\nPhase 233 failed ({failures} checks).")
        return 1
    print("\nshowNaturalOverall2 now exposes MRST's iteration and wall-time bar plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
