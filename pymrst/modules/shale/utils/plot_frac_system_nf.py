"""Hydraulic/natural fracture plot (MRST ``plotfracSystem_NF.m``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.modules.shale.utils.plot_frac_system import FractureSystemPlot, plot_frac_system


def plot_frac_system_nf(
    grid,
    fracture_planes,
    number_hydraulic_fractures: int,
    wells=(),
    *,
    fracture_list=None,
    label: bool = False,
    fracture_face_alpha: float = 0.5,
    hydraulic_color="yellow",
    natural_color="blue",
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    flip_z: bool = True,
    zscale: float = 1.0,
) -> FractureSystemPlot:
    """Render hydraulic and natural fractures with separate colors."""
    count = len(fracture_planes)
    nhf = int(number_hydraulic_fractures)
    if nhf < 0 or nhf > count:
        raise ValueError("number_hydraulic_fractures must lie between zero and fracture count")
    selected = (np.arange(count, dtype=np.int64) if fracture_list is None
                else np.asarray(fracture_list, dtype=np.int64).reshape(-1))

    colors = [hydraulic_color] * nhf + [natural_color] * (count - nhf)
    return plot_frac_system(
        grid, fracture_planes, wells=wells,
        fracture_list=selected, label=label,
        fracture_face_alpha=fracture_face_alpha,
        fracture_face_color=colors, well_color="red",
        plotter=plotter, filename=filename, show=show,
        flip_z=flip_z, zscale=zscale,
    )


plotfracSystem_NF = plot_frac_system_nf

__all__ = ["plot_frac_system_nf", "plotfracSystem_NF"]
