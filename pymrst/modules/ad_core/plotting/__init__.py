"""Plotting helpers (MRST `ad-core/plotting`).

`get_plot_after_step` is the after-step hook (which can stop a run), and
`simulation_runtime_panel` the progress panel it shows;
`plot_well_sols` is the interactive well-solution viewer, its controls built
from matplotlib widgets.
"""

from pymrst.modules.ad_core.plotting.get_plot_after_step import (
    getPlotAfterStep, get_plot_after_step, get_plot_after_step_ad,
)
from pymrst.modules.ad_core.plotting.simulation_runtime_panel import (
    simulationRuntimePanel, simulation_runtime_panel,
)
from pymrst.modules.ad_core.plotting.plot_well_sols import (
    plotWellSols, plot_well_sols,
)

__all__ = ["getPlotAfterStep", "get_plot_after_step", "get_plot_after_step_ad",
           "simulationRuntimePanel", "simulation_runtime_panel", "plotWellSols",
           "plot_well_sols"]
