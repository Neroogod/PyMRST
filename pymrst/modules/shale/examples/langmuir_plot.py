"""Langmuir sorption curve (MRST ``examples/langmuirPlot.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LangmuirCurve:
    pressure: np.ndarray
    sorbed_density: np.ndarray
    langmuir_pressure: float
    langmuir_density: float
    figure: object | None = None


def langmuir_plot(
    *,
    samples: int = 100,
    maximum_pressure: float = 10e7,
    langmuir_pressure: float = 1.077e7,
    langmuir_density: float = 3.0,
    plot: bool = False,
    show: bool = True,
    filename: str | Path | None = None,
) -> LangmuirCurve:
    """Compute, and optionally render, the reference Langmuir isotherm."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    if langmuir_pressure <= 0 or langmuir_density < 0:
        raise ValueError("Langmuir pressure must be positive and density non-negative")
    pressure = np.linspace(0.0, float(maximum_pressure), int(samples))
    density = pressure * float(langmuir_density) / (float(langmuir_pressure) + pressure)
    figure = None
    if plot or filename is not None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(4, 4))
        axes.plot(pressure, density, "bo-", linewidth=2)
        axes.set_xlabel("Pressure, Pa")
        axes.set_ylabel(r"$\rho_s$, kg/m³")
        if filename is not None:
            figure.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()
    return LangmuirCurve(
        pressure, density, float(langmuir_pressure), float(langmuir_density), figure,
    )


langmuirPlot = langmuir_plot

if __name__ == "__main__":
    # See `gangi_plot` for why `show=False` matters here.
    def test_langmuir_isotherm_rises_and_saturates():
        """Adsorption grows with pressure and approaches the Langmuir density.

        The saturating shape is the whole content of the isotherm. A missing
        denominator gives a straight line that still looks plausible plotted
        over a limited range.
        """
        curve = langmuir_plot(samples=80, plot=False, show=False)
        assert curve.sorbed_density.size == 80
        assert np.all(np.diff(curve.sorbed_density) >= -1e-12), "not monotone"
        assert np.all(curve.sorbed_density <= curve.langmuir_density + 1e-9)
        # Concave: the increments must shrink as pressure rises.
        steps = np.diff(curve.sorbed_density)
        assert np.all(np.diff(steps) <= 1e-12), "isotherm is not concave"
        print(f"ok test_langmuir_isotherm_rises_and_saturates "
              f"(0 to {curve.sorbed_density[-1]:.4f} of "
              f"{curve.langmuir_density:.1f})")

    def test_half_the_density_is_reached_at_the_langmuir_pressure():
        """The defining property of the isotherm, and an exact one."""
        curve = langmuir_plot(samples=400, plot=False, show=False)
        at_pl = np.interp(curve.langmuir_pressure, curve.pressure,
                          curve.sorbed_density)
        assert np.isclose(at_pl, curve.langmuir_density / 2.0, rtol=1e-3), at_pl
        print(f"ok test_half_the_density_is_reached_at_the_langmuir_pressure "
              f"({at_pl:.4f} against {curve.langmuir_density / 2:.4f})")

    def test_not_plotting_produces_no_figure():
        assert langmuir_plot(samples=5, plot=False, show=False).figure is None
        print("ok test_not_plotting_produces_no_figure")

    test_langmuir_isotherm_rises_and_saturates()
    test_half_the_density_is_reached_at_the_langmuir_pressure()
    test_not_plotting_produces_no_figure()
    print("\nAll tests passed!")


__all__ = ["LangmuirCurve", "langmuir_plot", "langmuirPlot"]
