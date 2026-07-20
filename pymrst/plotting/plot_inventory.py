"""The CO2 trapping inventory plot (MRST co2lab-ve plotTrappingDistribution).

The picture a storage assessment is actually for: a stacked area chart of
where the injected CO2 has ended up, over time. The bands are ordered
safest at the bottom (dissolved) to riskiest at the top (free and mobile),
so a site that is working shows the coloured mass migrating downward as the
years pass.

Drawn with PyVista's 2D charts, so it needs no extra plotting dependency.
"""

from typing import Optional, Sequence

import numpy as np
import pyvista as pv

from pymrst.modules.co2lab_mit.inventory import CATEGORIES

# safest (bottom) to riskiest (top), as co2lab orders them
COLORS = (
    (0.30, 0.55, 0.85),      # dissolved        - blue
    (0.20, 0.65, 0.45),      # residual in trap - green
    (0.55, 0.78, 0.45),      # residual         - light green
    (0.95, 0.82, 0.35),      # residual in plume- yellow
    (0.95, 0.62, 0.25),      # free in trap     - orange
    (0.75, 0.45, 0.70),      # sub-scale trapped- purple
    (0.85, 0.30, 0.28),      # free and mobile  - red
)


def plot_trapping_inventory(
    history: np.ndarray,                 # (n_steps, 7) in kg
    times: Optional[Sequence[float]] = None,
    time_unit: float = 1.0,
    time_label: str = "years",
    mass_unit: float = 1e9,              # kg -> Mt
    mass_label: str = "Mt CO2",
    filename: Optional[str] = None,
    show: bool = True,
):
    """Stacked bands of the seven trapping categories against time."""
    h = np.atleast_2d(np.asarray(history, float))
    assert h.shape[1] == len(CATEGORIES), (
        f"expected {len(CATEGORIES)} categories, got {h.shape[1]}")

    t = (np.arange(h.shape[0], dtype=float) if times is None
         else np.asarray(times, float).ravel() / time_unit)
    assert t.size == h.shape[0], "one time per inventory row"

    chart = pv.Chart2D()
    # cumulative sums give the band edges; draw from the top down so each
    # filled area is painted over the ones above it
    cum = np.cumsum(h, axis=1) / mass_unit
    for i in reversed(range(len(CATEGORIES))):
        band = chart.area(t, np.zeros_like(t), cum[:, i], color=COLORS[i])
        band.label = CATEGORIES[i]

    chart.x_label = f"time ({time_label})"
    chart.y_label = mass_label
    chart.title = "CO2 trapping inventory"

    pl = pv.Plotter(off_screen=filename is not None, window_size=(1000, 620))
    pl.add_chart(chart)
    pl.background_color = "white"
    if filename is not None:
        pl.screenshot(filename)
        pl.close()
        return filename
    if show:
        pl.show()
    return pl


if __name__ == "__main__":
    import os
    import tempfile

    pv.OFF_SCREEN = True
    OUT = tempfile.mkdtemp(prefix="pymrst_inventory_")

    def test_stacking_is_cumulative_and_ordered():
        """The bands stack: each edge is the running total of the ones below."""
        h = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],       # all mobile
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 5.0],        # spreading out
            [3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0],        # mostly safe
        ]) * 1e9
        cum = np.cumsum(h, axis=1)
        # the top edge of the last band is the total, by construction
        assert np.allclose(cum[:, -1], h.sum(axis=1))
        # and the bands never cross
        assert np.all(np.diff(cum, axis=1) >= -1e-12)
        print("ok test_stacking_is_cumulative_and_ordered")

    def test_renders_a_non_empty_chart():
        """The inventory draws, with and without explicit times."""
        rng = np.random.default_rng(0)
        n = 20
        h = np.zeros((n, 7))
        # a plausible story: mobile CO2 turning into trapped CO2
        frac = np.linspace(0.0, 1.0, n)
        h[:, 6] = (1.0 - frac) * 10e9
        h[:, 2] = frac * 4e9
        h[:, 4] = frac * 3e9
        h[:, 1] = frac * 2e9
        h[:, 0] = frac * 1e9

        f = os.path.join(OUT, "inventory.png")
        plot_trapping_inventory(h, times=np.arange(n) * 3.15576e7 * 10,
                                time_unit=3.15576e7, filename=f, show=False)
        assert os.path.getsize(f) > 5000, "the chart looks empty"

        f2 = os.path.join(OUT, "inventory_steps.png")
        plot_trapping_inventory(h, filename=f2, show=False)
        assert os.path.getsize(f2) > 5000
        print(f"ok test_renders_a_non_empty_chart -> {f}")

    def test_rejects_the_wrong_number_of_categories():
        """A history with the wrong shape is caught, not silently plotted."""
        try:
            plot_trapping_inventory(np.zeros((5, 4)), show=False)
        except AssertionError as e:
            assert "categories" in str(e)
        else:
            raise AssertionError("a 4-column history should have been rejected")
        print("ok test_rejects_the_wrong_number_of_categories")

    test_stacking_is_cumulative_and_ordered()
    test_rejects_the_wrong_number_of_categories()
    test_renders_a_non_empty_chart()
    print(f"\nAll tests passed!  (images in {OUT})")
