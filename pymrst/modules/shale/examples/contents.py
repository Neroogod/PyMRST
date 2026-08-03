"""Machine-readable counterpart of MRST shale ``examples/Contents.m``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShaleExample:
    owner: str
    module: str
    description: str
    section: str


EXAMPLES = (
    ShaleExample("GangiPlot", "gangi_plot", "Gangi permeability correction curve", "Figure 21"),
    ShaleExample("LangmuirPlot", "langmuir_plot", "Langmuir sorption isotherm", "Figure 17"),
    ShaleExample("StochasticFracs", "stochastic_fractures", "Stochastic fracture realization", "Figures 7-8"),
    ShaleExample("compositional3D", "compositional_3d", "HFM compositional waterflood", "Section 4"),
    ShaleExample("Sorption", "sorption", "Competitive sorption comparison", "Section 7.1"),
    ShaleExample("Diffusion", "diffusion", "Gas-diffusion comparison", "Section 7.2"),
    ShaleExample("Gangi", "gangi", "Stress-sensitive permeability comparison", "Section 7.3"),
    ShaleExample("EagleFord", "eagle_ford", "Eagle Ford pEDFM application", "Section 6"),
    ShaleExample("EagleFordEDFM", "eagle_ford_edfm", "Eagle Ford EDFM application", "Section 6"),
)


def shale_example_catalog() -> tuple[ShaleExample, ...]:
    """Return the ordered catalog documented by the MRST reference owner."""
    return EXAMPLES


__all__ = ["ShaleExample", "EXAMPLES", "shale_example_catalog"]
