"""Runnable shale simulation and validation examples."""

from pymrst.modules.shale.examples.data_loader import available_shale_datasets, load_shale_dataset
from pymrst.modules.shale.examples.contents import EXAMPLES, ShaleExample, shale_example_catalog
from pymrst.modules.shale.examples.compositional_3d import (
    Compositional3DCase,
    run_compositional_3d_case,
    setup_compositional_3d_case,
)
from pymrst.modules.shale.examples.gangi_plot import GangiCurve, gangi_plot, gangiPlot
from pymrst.modules.shale.examples.langmuir_plot import LangmuirCurve, langmuir_plot, langmuirPlot
from pymrst.modules.shale.examples.diffusion import run_diffusion_case, setup_diffusion_case
from pymrst.modules.shale.examples.gangi import run_gangi_case, setup_gangi_case
from pymrst.modules.shale.examples.sorption import run_sorption_case, setup_sorption_case
from pymrst.modules.shale.examples.stochastic_fractures import (
    StochasticFractureCase,
    setup_stochastic_fractures,
    stochasticFracs,
)
from pymrst.modules.shale.examples.eagle_ford import setup_eagle_ford, run_eagle_ford, eagleFord
from pymrst.modules.shale.examples.eagle_ford_edfm import (
    setup_eagle_ford_edfm, run_eagle_ford_edfm, eagleFordEDFM,
)
from pymrst.modules.shale.examples.eagle_ford_full import (
    setup_eagle_ford_full, run_eagle_ford_full, eagleFordfull,
)
from pymrst.modules.shale.examples.eagle_ford_edfm_full import (
    setup_eagle_ford_edfm_full, run_eagle_ford_edfm_full, eagleFordEDFMfull,
)
from pymrst.modules.shale.examples.eagle_ford_accelerated import (
    eagle_ford_accelerated_backend, setup_eagle_ford_accelerated,
    run_eagle_ford_accelerated, eagleFord_MEX,
)
from pymrst.modules.shale.examples.eagle_ford_edfm_accelerated import (
    eagle_ford_edfm_accelerated_backend, setup_eagle_ford_edfm_accelerated,
    run_eagle_ford_edfm_accelerated, eagleFordEDFM_MEX,
)

__all__ = [
    "available_shale_datasets", "load_shale_dataset",
    "EXAMPLES", "ShaleExample", "shale_example_catalog",
    "Compositional3DCase", "setup_compositional_3d_case", "run_compositional_3d_case",
    "GangiCurve", "gangi_plot", "gangiPlot",
    "LangmuirCurve", "langmuir_plot", "langmuirPlot",
    "setup_diffusion_case", "run_diffusion_case",
    "setup_gangi_case", "run_gangi_case",
    "setup_sorption_case", "run_sorption_case",
    "StochasticFractureCase", "setup_stochastic_fractures", "stochasticFracs",
    "setup_eagle_ford", "run_eagle_ford", "eagleFord",
    "setup_eagle_ford_edfm", "run_eagle_ford_edfm", "eagleFordEDFM",
    "setup_eagle_ford_full", "run_eagle_ford_full", "eagleFordfull",
    "setup_eagle_ford_edfm_full", "run_eagle_ford_edfm_full", "eagleFordEDFMfull",
    "eagle_ford_accelerated_backend", "setup_eagle_ford_accelerated",
    "run_eagle_ford_accelerated", "eagleFord_MEX",
    "eagle_ford_edfm_accelerated_backend", "setup_eagle_ford_edfm_accelerated",
    "run_eagle_ford_edfm_accelerated", "eagleFordEDFM_MEX",
]
