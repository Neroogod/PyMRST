import numpy as np
import pytest

from pymrst.modules.ad_mechanics.examples.utils.plot_2d_poro_example import (
    plot_2d_poro_example,
)


def test_plot_2d_rejects_empty_state_history():
    with pytest.raises(ValueError, match="at least one"):
        plot_2d_poro_example(object(), [])


def test_plot_2d_passes_flow_and_mechanics_fields_to_toolbar(monkeypatch):
    captured = {}

    def toolbar(grid, fields, **kwargs):
        captured.update(grid=grid, fields=fields, kwargs=kwargs)
        return "toolbar"

    monkeypatch.setattr("pymrst.plotting.extensions.plot_toolbar.plot_toolbar", toolbar)
    grid = type("Grid", (), {"griddim": 2, "cells": type("Cells", (), {"num": 2})()})()

    class Mechanics:
        @staticmethod
        def volumetric_strain_full(u):
            return np.array([u[0], u[2]])

    model = type("Model", (), {"G": grid, "mech": Mechanics()})()
    states = [
        {"pressure": np.array([1.0, 2.0]), "sW": np.array([0.2, 0.3]), "u": np.array([1.0, 0.0, 2.0, 0.0])},
        {"pressure": np.array([3.0, 4.0]), "sW": np.array([0.4, 0.5]), "u": np.array([2.0, 0.0, 3.0, 0.0])},
    ]
    result = plot_2d_poro_example(model, states, {"fluid_model": "oil water", "method": "fully coupled"}, show=False)
    assert result == "toolbar"
    assert captured["fields"]["pressure"].shape == (2, 2)
    assert captured["fields"]["volumetric strain"].shape == (2, 2)
    assert captured["kwargs"]["show"] is False
    assert "oil water" in captured["kwargs"]["title"]
