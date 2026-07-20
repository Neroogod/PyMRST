"""Initialize compositional states (MRST `initCompositionalState.m` layout)."""

from typing import Any

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.eos import flash
from pymrst.modules.compositional.flow_props import flow_props_from_flash
from pymrst.modules.compositional.model import CompositionalState


def _cell_count(model: Any) -> int:
    if hasattr(model, "g") and hasattr(model.g, "cells"):
        return int(model.g.cells.num)
    if hasattr(model, "G") and hasattr(model.G, "cells"):
        return int(model.G.cells.num)
    raise AttributeError("model must expose `g.cells.num` or `G.cells.num`")


def _broadcast_vector(values: Any, count: int, name: str) -> np.ndarray:
    arr = np.asarray(values, float)
    if arr.ndim == 0:
        return np.full(count, float(arr))
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full(count, float(arr[0]))
    if arr.size != count:
        raise ValueError(f"{name} must have length 1 or {count}, got {arr.size}")
    return arr.copy()


def _composition_array(values: Any, count: int, ncomp: int) -> np.ndarray:
    z = np.asarray(values, float)
    if z.ndim == 1:
        if z.size != ncomp:
            raise ValueError(f"composition must have {ncomp} entries, got {z.size}")
        z = np.tile(z, (count, 1))
    elif z.shape == (1, ncomp):
        z = np.repeat(z, count, axis=0)
    elif z.shape == (ncomp, count):
        z = z.T
    elif z.shape != (count, ncomp):
        raise ValueError(f"composition must be {(count, ncomp)} or {(ncomp, count)}, got {z.shape}")
    z = np.clip(z, 0.0, np.inf)
    totals = z.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("composition rows must have positive sum")
    return z / totals


def _water_saturation(saturation: Any, count: int) -> np.ndarray:
    if saturation is None:
        return np.zeros(count)
    sat = np.asarray(saturation, float)
    if sat.ndim == 0:
        return np.full(count, float(sat))
    if sat.ndim == 1:
        return _broadcast_vector(sat, count, "water saturation")
    if sat.shape[0] == 1:
        sat = np.repeat(sat, count, axis=0)
    if sat.shape[0] != count:
        raise ValueError(f"saturation must have {count} rows, got {sat.shape[0]}")
    return np.asarray(sat[:, 0], float)


def init_compositional_state(
    model: Any,
    pressure: Any,
    temperature: Any,
    saturation: Any,
    composition: Any,
    eos: Mixture | None = None,
) -> CompositionalState:
    """Initialize pressure, composition, phase split, and saturations.

    The returned :class:`CompositionalState` is the model's native state object.
    MRST-style flash outputs are attached as extra attributes: `T`, `x`, `y`,
    `L`, `Z_L`, `Z_V`, `s`, and `flash_result`.
    """
    mix = eos if eos is not None else model.mix
    count = _cell_count(model)
    p = _broadcast_vector(pressure, count, "pressure")
    t = _broadcast_vector(temperature, count, "temperature")
    z = _composition_array(composition, count, mix.n)

    s_w = _water_saturation(saturation, count) if getattr(model, "water", False) else None
    state = model.initial_state(p, z, sW=s_w)

    flash_result = flash(mix, p, t, z)
    liquid_fraction = 1.0 - flash_result.V
    flow_props = flow_props_from_flash(
        mix,
        p,
        t,
        flash_result.x,
        flash_result.y,
        liquid_fraction,
        flash_result.Z_L,
        flash_result.Z_V,
    )

    if getattr(model, "water", False):
        assert s_w is not None
        void = 1.0 - s_w
        saturations = np.column_stack((s_w, void * flow_props.sL, void * flow_props.sV))
    else:
        saturations = np.column_stack((flow_props.sL, flow_props.sV))

    state.T = t
    state.x = flash_result.x
    state.y = flash_result.y
    state.L = liquid_fraction
    state.Z_L = flash_result.Z_L
    state.Z_V = flash_result.Z_V
    state.s = saturations
    state.flash_result = flash_result
    return state


initCompositionalState = init_compositional_state


if __name__ == "__main__":
    from dataclasses import dataclass

    from pymrst.modules.compositional.eos import simple_mixture

    @dataclass
    class _Cells:
        num: int

    @dataclass
    class _Grid:
        cells: _Cells

    class _Model:
        def __init__(self, count: int, water: bool = False):
            self.g = _Grid(_Cells(count))
            self.mix = simple_mixture()
            self.water = water

        def initial_state(self, p, z, sW=None):
            return CompositionalState(
                pressure=np.asarray(p, float),
                z=np.asarray(z, float),
                qOs=np.zeros(0),
                qGs=np.zeros(0),
                bhp=np.zeros(0),
                sW=sW,
            )

    def test_initial_state_broadcasts_and_flashes():
        model = _Model(4)
        state = init_compositional_state(model, 50e5, 350.0, [], [0.5, 0.3, 0.2])
        assert state.pressure.shape == (4,)
        assert state.z.shape == (4, 3)
        assert state.x.shape == (4, 3)
        assert state.s.shape == (4, 2)
        assert np.allclose(state.s.sum(axis=1), 1.0)
        assert np.allclose(state.T, 350.0)
        print("ok test_initial_state_broadcasts_and_flashes")

    def test_initial_state_accepts_component_rows():
        model = _Model(2)
        z = np.array([[0.5, 0.6], [0.3, 0.2], [0.2, 0.2]])
        state = init_compositional_state(model, [50e5, 60e5], 350.0, [], z)
        assert np.allclose(state.z[:, 0], [0.5, 0.6])
        assert np.allclose(state.z.sum(axis=1), 1.0)
        print("ok test_initial_state_accepts_component_rows")

    def test_water_saturation_scales_hydrocarbon_void():
        model = _Model(3, water=True)
        state = init_compositional_state(model, 50e5, 350.0, 0.2, [0.5, 0.3, 0.2])
        assert state.sW is not None
        assert state.s.shape == (3, 3)
        assert np.allclose(state.s[:, 0], 0.2)
        assert np.allclose(state.s.sum(axis=1), 1.0)
        print("ok test_water_saturation_scales_hydrocarbon_void")

    test_initial_state_broadcasts_and_flashes()
    test_initial_state_accepts_component_rows()
    test_water_saturation_scales_hydrocarbon_void()
    print("\nAll tests passed!")
