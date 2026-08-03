import numpy as np
import pytest

from pymrst.modules.ad_mechanics.examples.utils.plot_norne_poro_example import plot_norne_poro_example
from pymrst.utils.units import BARSA


def test_norne_plot_requires_states():
    with pytest.raises(ValueError, match="at least one"):
        plot_norne_poro_example(object(), [])


def test_norne_plot_builds_field_history_without_opening_window(monkeypatch):
    captured = {}

    def toolbar(grid, fields, **kwargs):
        captured.update(fields=fields, kwargs=kwargs)
        return "toolbar"

    monkeypatch.setattr("pymrst.plotting.extensions.plot_toolbar.plot_toolbar", toolbar)
    grid = type("Grid", (), {"griddim": 3})()
    mechanics = type("Mechanics", (), {"volumetric_strain_full": staticmethod(lambda u: np.array([u[0]]))})()
    model = type("Model", (), {"G": grid, "mech": mechanics})()
    states = [
        {"pressure": np.array([100 * BARSA]), "sW": np.array([0.2]), "u": np.array([1.0, 0.0, 0.0])},
        {"pressure": np.array([110 * BARSA]), "sW": np.array([0.3]), "u": np.array([2.0, 0.0, 0.0])},
    ]
    assert plot_norne_poro_example(model, states, show=False) == "toolbar"
    assert np.asarray(captured["fields"]["pressure [bar]"]).ravel() == pytest.approx([100, 110])
    assert np.asarray(captured["fields"]["div(u)"]).ravel() == pytest.approx([1, 2])
    assert captured["kwargs"]["show"] is False
