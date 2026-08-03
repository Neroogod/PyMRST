"""Gangi permeability-correction curve (MRST ``examples/gangiPlot.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymrst.modules.shale.utils.gangi import gangi_fn


@dataclass(frozen=True)
class GangiCurve:
    pressure: np.ndarray
    multiplier: np.ndarray
    confining_pressure: float
    closure_pressure: float
    biot_coefficient: float
    roughness: float
    figure: object | None = None


def gangi_plot(
    *,
    samples: int = 50,
    maximum_pressure: float = 300e5,
    confining_pressure: float = 38e6,
    closure_pressure: float = 180e6,
    biot_coefficient: float = 0.5,
    roughness: float = 0.5,
    plot: bool = False,
    show: bool = True,
    filename: str | Path | None = None,
) -> GangiCurve:
    """Compute, and optionally render, MRST's reference Gangi curve."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    pressure = np.linspace(0.0, float(maximum_pressure), int(samples))
    multiplier = np.asarray(gangi_fn(
        pressure, confining_pressure, biot_coefficient, closure_pressure,
        roughness, 1.0, 1.0,
    ))
    figure = None
    if plot or filename is not None:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(8, 6))
        axes.plot(pressure, multiplier, "bo-", linewidth=2)
        axes.grid(True)
        axes.set_ylim(.15, .30)
        axes.set_xlabel("Pressure, Pa")
        axes.set_ylabel(r"$F_{app}$")
        if filename is not None:
            figure.savefig(filename, bbox_inches="tight")
        if show:
            plt.show()
    return GangiCurve(
        pressure, multiplier, float(confining_pressure), float(closure_pressure),
        float(biot_coefficient), float(roughness), figure,
    )


gangiPlot = gangi_plot

if __name__ == "__main__":
    # `plot=True` alone leaves `show=True`, which opens a window and waits.
    # Whether that blocks depends on the backend, so it blocks on some machines
    # and not others - which is how `stochastic_fractures` stalled the sweep for
    # half an hour. The curve is what this example is about, so assert that.
    def test_gangi_multiplier_rises_with_pore_pressure():
        """Gangi's law: raising pore pressure lowers effective stress, so the
        aperture opens and the permeability multiplier rises.

        Written the other way round first, which the data refuted: the curve
        climbs from 0.158 to 0.265 over 0-300 bar. A sign error on the
        effective stress would reverse it while leaving a plausible-looking
        plot, so the direction is the assertion.
        """
        curve = gangi_plot(samples=40, plot=False, show=False)
        assert curve.pressure.size == curve.multiplier.size == 40
        assert np.all(np.isfinite(curve.multiplier))
        assert np.all((curve.multiplier >= 0.0) & (curve.multiplier <= 1.0)), (
            curve.multiplier.min(), curve.multiplier.max())
        assert np.all(np.diff(curve.multiplier) >= -1e-12), "multiplier is not monotone"
        # Convex: the aperture opens faster as effective stress approaches zero.
        assert np.all(np.diff(np.diff(curve.multiplier)) >= -1e-12), "not convex"
        print(f"ok test_gangi_multiplier_rises_with_pore_pressure "
              f"({curve.multiplier[0]:.4f} to {curve.multiplier[-1]:.4f})")

    def test_a_higher_closure_pressure_keeps_the_aperture_open():
        """Closure pressure is the stress that shuts the fracture entirely, so
        a larger one leaves more aperture at the same effective stress.

        Measured, not assumed: 0.043-0.121 at 90 MPa against 0.308-0.417 at
        360 MPa.
        """
        soft = gangi_plot(samples=20, closure_pressure=90e6, plot=False, show=False)
        hard = gangi_plot(samples=20, closure_pressure=360e6, plot=False, show=False)
        assert np.all(hard.multiplier >= soft.multiplier - 1e-12)
        print("ok test_a_higher_closure_pressure_keeps_the_aperture_open")

    def test_too_few_samples_is_rejected():
        try:
            gangi_plot(samples=1, plot=False, show=False)
        except ValueError as error:
            assert "at least two" in str(error), error
        else:
            raise AssertionError("expected a single sample to be rejected")
        print("ok test_too_few_samples_is_rejected")

    def test_not_plotting_produces_no_figure():
        assert gangi_plot(samples=5, plot=False, show=False).figure is None
        print("ok test_not_plotting_produces_no_figure")

    test_gangi_multiplier_rises_with_pore_pressure()
    test_a_higher_closure_pressure_keeps_the_aperture_open()
    test_too_few_samples_is_rejected()
    test_not_plotting_produces_no_figure()
    print("\nAll tests passed!")


__all__ = ["GangiCurve", "gangi_plot", "gangiPlot"]
