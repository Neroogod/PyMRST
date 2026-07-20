"""Phase 229: 3D six-component example MRST plotting-frame parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.compositional_example_3d_six_components import (  # noqa: E402
    extract_plot_fields,
    run_compositional_example_3d_six_components,
    six_component_plot_frames,
    six_component_state_plot_fields,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 229: 3D six-component plotting-frame parity")
    case = run_compositional_example_3d_six_components((2, 2, 2), nstep=2)
    frames = case["plot_frames"]
    final_fields = extract_plot_fields(case)
    manual_frames = six_component_plot_frames(case)
    first_state_fields = six_component_state_plot_fields(case["states"][0])
    nc = case["grid"].cells.num
    ncomp = case["mixture"].n

    failures = 0
    failures += _check("one frame per report state",
                       len(frames["frames"]) == len(case["states"]) == 2)
    failures += _check("component names match benchmark mixture",
                       frames["component_names"] == case["mixture"].names
                       and len(frames["component_names"]) == ncomp)
    failures += _check("MRST subplot metadata is present",
                       frames["view"] == (-30, 60)
                       and frames["subplot_columns"] == 3
                       and frames["subplot_rows"] == int(np.ceil(ncomp / 2.0))
                       and frames["extra_fields"] == ("pressure", "sO", "sG"))
    first = frames["frames"][0]["fields"]
    failures += _check("component frame has one row per cell and component",
                       first["components"].shape == (nc, ncomp)
                       and np.allclose(first["components"].sum(axis=1), 1.0))
    failures += _check("pressure frame has one value per cell",
                       first["pressure"].shape == (nc,)
                       and np.all(np.isfinite(first["pressure"])))
    failures += _check("oil and gas saturation frames have one value per cell",
                       first["sO"].shape == (nc,)
                       and first["sG"].shape == (nc,))
    failures += _check("saturation frames are bounded",
                       np.all((first["sO"] >= -1e-12) & (first["sO"] <= 1.0 + 1e-12))
                       and np.all((first["sG"] >= -1e-12) & (first["sG"] <= 1.0 + 1e-12)))
    failures += _check("frame times match cumulative schedule time",
                       np.allclose(
                           [frame["time"] for frame in frames["frames"]],
                           case["times"],
                       ))
    failures += _check("final compatibility helper matches final frame",
                       np.allclose(final_fields["components"], frames["frames"][-1]["fields"]["components"])
                       and np.allclose(final_fields["pressure"], frames["frames"][-1]["fields"]["pressure"])
                       and np.allclose(final_fields["sO"], frames["frames"][-1]["fields"]["sO"])
                       and np.allclose(final_fields["sG"], frames["frames"][-1]["fields"]["sG"]))
    failures += _check("standalone frame helpers reproduce stored output",
                       np.allclose(first_state_fields["components"], first["components"])
                       and manual_frames["frames"][0]["fields"]["pressure"].shape == (nc,))

    if failures:
        print(f"\nPhase 229 validation failed ({failures} checks).")
    else:
        print("\n3D six-component example now exposes MRST-shaped plotting frames.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
