"""Phase 241: 3D six-component example plot-section metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_example_3d_six_components import (  # noqa: E402
    run_compositional_example_3d_six_components,
    six_component_plot_sections,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    out = run_compositional_example_3d_six_components((2, 2, 2), nstep=1)
    sections = out["plot_sections"]
    helper = six_component_plot_sections(out)
    animation = sections["animation"]
    toolbar = sections["toolbar"]

    print("\nPhase 241: 3D six-component plot-section metadata parity")
    failures = 0
    failures += _check("runner returns MRST animation and toolbar sections",
                       set(sections) == {"animation", "toolbar"})
    failures += _check("animation carries MRST figure/subplot/view metadata",
                       animation["figure_position_offset"] == (0, -200, 350, 200)
                       and animation["subplot_columns"] == 3
                       and animation["subplot_rows"] == int(np.ceil(out["mixture"].n / 2.0))
                       and animation["view"] == (-30, 60)
                       and animation["drawnow"] is True)
    failures += _check("component plots carry MRST titles and color limits",
                       len(animation["component_plots"]) == out["mixture"].n
                       and tuple(p["title"] for p in animation["component_plots"]) == out["mixture"].names
                       and all(p["edge_color"] == "none" for p in animation["component_plots"])
                       and all(p["caxis"] == (0.0, 1.0) for p in animation["component_plots"]))
    failures += _check("pressure/oil/gas subplot metadata matches MRST tail plots",
                       tuple(p["field"] for p in animation["extra_plots"]) == ("pressure", "sO", "sG")
                       and tuple(p["title"] for p in animation["extra_plots"]) == ("Pressure", "sO", "sG")
                       and all(p["edge_color"] == "none" for p in animation["extra_plots"]))
    failures += _check("animation frames preserve per-state plot fields",
                       len(animation["frames"]) == 1
                       and animation["frames"][0]["fields"]["components"].shape == (8, 6)
                       and animation["frames"][0]["fields"]["pressure"].shape == (8,))
    failures += _check("toolbar carries MRST plotToolbar/view/axis metadata",
                       toolbar["figure"] == 1
                       and toolbar["states"] == out["states"]
                       and toolbar["grid_cells"] == out["grid"].cells.num
                       and toolbar["view"] == (-30, 60)
                       and toolbar["axis"] == "tight")
    failures += _check("standalone helper reproduces runner plot metadata",
                       helper["animation"]["component_plots"] == animation["component_plots"]
                       and helper["toolbar"]["axis"] == toolbar["axis"])

    if failures:
        print(f"\nPhase 241 failed ({failures} checks).")
        return 1
    print("\n3D six-component example now exposes MRST animation and toolbar metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
