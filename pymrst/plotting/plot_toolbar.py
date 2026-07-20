"""Interactive model inspector (MRST mrst-gui/plotToolbar.m).

A drop-in replacement for plot_cell_data that, instead of one field frozen
at one instant, lets you walk around a whole simulation:

  - switch between datasets (pressure, saturation, time-of-flight, ...)
  - scrub through timesteps, or play the run back
  - transform on the fly (log10, |x|) without knowing the range up front
  - threshold on the current field, to keep only the cells that matter
  - drag a clip plane through the model to see inside it
  - read a histogram of what is currently shown

    plot_toolbar(G, {"pressure": p, "sW": sw})       # several fields
    plot_toolbar(G, states)                          # a whole schedule
    plot_toolbar(G, state.pressure)                  # just one field

The data layer (`normalize_data`, `states_to_fields`, `apply_transform`,
`threshold_mask`) is deliberately free of any window, so it can be tested
headless; the widgets sit on top of it.

Uses pyvistaqt's BackgroundPlotter when it is installed, which gives a real
non-blocking window with menus, as plotToolbar has. Without it, everything
still works in a plain (blocking) PyVista window driven by keys and
sliders.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pyvista as pv

from pymrst.gridprocessing.grid import Grid
from pymrst.plotting.plot_grid import to_pyvista

try:                                        # optional: the nicer window
    from pyvistaqt import BackgroundPlotter
    HAVE_QT = True
except ImportError:                         # pragma: no cover
    BackgroundPlotter = None
    HAVE_QT = False

TRANSFORMS = ("linear", "log10", "abs", "log10(abs)")


# --------------------------------------------------------------------------
# data layer - no window anywhere in here
# --------------------------------------------------------------------------
def states_to_fields(states: Sequence, nc: int) -> Dict[str, np.ndarray]:
    """Every per-cell field of a list of states, as (n_steps, nc) arrays.

    Picks out the entries that carry one value per cell and ignores the rest
    (face fluxes, well solutions, displacement dofs). A saturation array of
    shape (nc, n_phases) is split into one field per phase, so an incomp
    `State` yields pressure, s[0], s[1] and an `OWState` yields pressure and
    sW.

    Handles both state flavours: the dataclass states of the hand-rolled
    models (read through `__dict__`) and the dict-based `State` of the
    state-function graph, whose fields are keys rather than attributes.
    """
    if not states:
        raise ValueError("no states given")
    first = states[0]

    if isinstance(first, Mapping):
        def keys_of(s):
            return list(s.keys())

        def get(s, key):
            return s[key]
    else:
        attrs = getattr(first, "__dict__", None)
        if attrs is None:
            raise TypeError(f"cannot read fields off a {type(first).__name__}")

        def keys_of(s):
            return list(vars(s).keys())

        def get(s, key):
            return getattr(s, key)

    out: Dict[str, np.ndarray] = {}
    for key in keys_of(first):
        v = get(first, key)
        if not isinstance(v, (np.ndarray, list, tuple)):
            continue
        arr = np.asarray(v)
        if arr.dtype == object:
            continue
        if arr.ndim == 1 and arr.size == nc:
            out[key] = np.stack([
                np.asarray(get(s, key), float).ravel() for s in states])
        elif arr.ndim == 2 and arr.shape[0] == nc:
            for j in range(arr.shape[1]):
                out[f"{key}[{j}]"] = np.stack([
                    np.asarray(get(s, key), float)[:, j] for s in states])
    if not out:
        raise ValueError(
            f"no per-cell fields found on {type(first).__name__} "
            f"(expected arrays of length {nc})")
    return out


def normalize_data(data, nc: int) -> Dict[str, np.ndarray]:
    """Bring every accepted input into one shape: name -> (n_steps, nc).

    Accepts a single field, a dict of fields, a list of states, or a list
    of arrays. A field may be static (nc,) or time-varying (n_steps, nc).
    """
    def as_series(name, v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 1:
            if a.size != nc:
                raise ValueError(
                    f"field '{name}': expected {nc} values (one per cell), "
                    f"got {a.size}")
            return a[None, :]
        if a.ndim == 2:
            if a.shape[1] != nc:
                raise ValueError(
                    f"field '{name}': expected (n_steps, {nc}), "
                    f"got {a.shape}")
            return a
        raise ValueError(f"field '{name}': expected 1 or 2 dimensions, "
                         f"got {a.ndim}")

    if isinstance(data, dict):
        return {k: as_series(k, v) for k, v in data.items()}

    if isinstance(data, np.ndarray):
        return {"data": as_series("data", data)}

    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("no data given")
        if isinstance(data[0], np.ndarray):
            return {"data": as_series("data", np.stack(
                [np.asarray(d, float).ravel() for d in data]))}
        return states_to_fields(data, nc)      # a list of state objects

    raise TypeError(f"cannot plot a {type(data).__name__}")


def apply_transform(v: np.ndarray, mode: str) -> np.ndarray:
    """Transform the displayed values. Non-positive values become NaN under
    log10, which drops them from the picture rather than faking a value."""
    if mode == "linear":
        return v
    if mode == "abs":
        return np.abs(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "log10":
            return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
        if mode == "log10(abs)":
            a = np.abs(v)
            return np.where(a > 0, np.log10(np.where(a > 0, a, 1.0)), np.nan)
    raise ValueError(f"unknown transform '{mode}' (have {TRANSFORMS})")


def threshold_mask(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Cells whose value lies in [lo, hi]. NaN never passes."""
    with np.errstate(invalid="ignore"):
        return (v >= lo) & (v <= hi)


@dataclass
class ToolbarState:
    """What the toolbar is currently showing."""

    field: str
    step: int
    transform: str
    lo: float
    hi: float
    clip: bool = False


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------
class PlotToolbar:
    """Interactive inspector over a grid and one or more cell datasets."""

    def __init__(
        self,
        g: Grid,
        data,
        times: Optional[Sequence[float]] = None,
        cmap: str = "viridis",
        time_unit: float = 1.0,
        time_label: str = "s",
        off_screen: bool = False,
        use_qt: bool = True,
        title: str = "PyMRST",
    ):
        self.g = g
        self.fields = normalize_data(data, g.cells.num)
        self.names = list(self.fields)
        self.n_steps = max(v.shape[0] for v in self.fields.values())
        self.times = (None if times is None
                      else np.asarray(times, float).ravel())
        if self.times is not None and self.times.size != self.n_steps:
            raise ValueError(
                f"got {self.times.size} times for {self.n_steps} steps")
        self.time_unit = time_unit
        self.time_label = time_label
        self.cmap = cmap
        self.off_screen = off_screen

        self.base = to_pyvista(g)
        self.state = ToolbarState(field=self.names[0], step=0,
                                  transform="linear", lo=-np.inf, hi=np.inf)
        self._actor = None
        self._chart = None
        self._title_actor = None

        self.qt = bool(use_qt and HAVE_QT and not off_screen)
        if self.qt:
            self.pl = BackgroundPlotter(title=title)
        else:
            self.pl = pv.Plotter(off_screen=off_screen, title=title)

        self._reset_range()
        if not off_screen:
            self._add_widgets()
        self._render()
        self.pl.add_axes()
        self.pl.camera_position = "xz"
        self.pl.camera.up = (0.0, 0.0, -1.0)      # depth increases downward
        self.pl.camera.azimuth = 210
        self.pl.camera.elevation = 25

    # -- data ------------------------------------------------------------
    def values(self, transformed: bool = True) -> np.ndarray:
        """The values currently on screen, before thresholding."""
        series = self.fields[self.state.field]
        step = min(self.state.step, series.shape[0] - 1)   # static fields
        v = series[step]
        return apply_transform(v, self.state.transform) if transformed else v

    def _finite_range(self) -> tuple:
        """The colour range for the current field, over its *whole* series.

        Deliberately not the range of the current step. Rescaling every step
        would shift the colours under you as you scrub, so two frames could
        not be compared and playback would be meaningless - and a field that
        happens to be uniform at step 0 (an initial saturation, say) would
        fix the scale on a degenerate range. This is what plotToolbar's fixed
        caxis does.
        """
        v = apply_transform(self.fields[self.state.field],
                            self.state.transform)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return (0.0, 1.0)
        lo, hi = float(finite.min()), float(finite.max())
        # a genuinely constant field still needs a non-empty range to draw
        return (lo, hi if hi > lo else lo + 1.0)

    def _reset_range(self) -> None:
        self.state.lo, self.state.hi = self._finite_range()

    def visible_cells(self) -> np.ndarray:
        return np.flatnonzero(
            threshold_mask(self.values(), self.state.lo, self.state.hi))

    # -- drawing ---------------------------------------------------------
    def _label(self) -> str:
        s = self.state
        txt = s.field
        if s.transform != "linear":
            txt = f"{s.transform}({txt})"
        if self.n_steps > 1:
            txt += f"   step {s.step + 1}/{self.n_steps}"
            if self.times is not None:
                txt += (f"   t = {self.times[s.step] / self.time_unit:.3g} "
                        f"{self.time_label}")
        shown = self.visible_cells().size
        if shown != self.g.cells.num:
            txt += f"   [{shown}/{self.g.cells.num} cells]"
        return txt

    def _render(self) -> None:
        if self._actor is not None:
            self.pl.remove_actor(self._actor)
            self._actor = None
        self.pl.clear_plane_widgets()

        mesh = self.base.copy()
        v = self.values()
        mesh.cell_data["value"] = v
        lo, hi = self.state.lo, self.state.hi
        if np.isfinite(lo) and np.isfinite(hi):
            mesh = mesh.threshold(value=(lo, hi), scalars="value")

        if mesh.n_cells:
            add = (self.pl.add_mesh_clip_plane if self.state.clip
                   else self.pl.add_mesh)
            self._actor = add(mesh, scalars="value", cmap=self.cmap,
                              clim=(lo, hi) if np.isfinite(lo) else None,
                              scalar_bar_args={"title": self.state.field})

        if self._title_actor is not None:
            self.pl.remove_actor(self._title_actor)
        self._title_actor = self.pl.add_text(self._label(), font_size=10,
                                             position="upper_left")
        if self._chart is not None:
            self._draw_histogram()
        self.pl.render()

    # -- actions ---------------------------------------------------------
    def set_field(self, name: str) -> None:
        if name not in self.fields:
            raise KeyError(f"no field '{name}' (have {self.names})")
        self.state.field = name
        self._reset_range()                    # ranges differ wildly by field
        self._render()

    def next_field(self, step: int = 1) -> None:
        i = (self.names.index(self.state.field) + step) % len(self.names)
        self.set_field(self.names[i])

    def set_step(self, step: int) -> None:
        self.state.step = int(np.clip(step, 0, self.n_steps - 1))
        self._render()

    def set_transform(self, mode: str) -> None:
        if mode not in TRANSFORMS:
            raise ValueError(f"unknown transform '{mode}'")
        self.state.transform = mode
        self._reset_range()
        self._render()

    def next_transform(self) -> None:
        i = (TRANSFORMS.index(self.state.transform) + 1) % len(TRANSFORMS)
        self.set_transform(TRANSFORMS[i])

    def set_threshold(self, lo: Optional[float] = None,
                      hi: Optional[float] = None) -> None:
        if lo is not None:
            self.state.lo = float(lo)
        if hi is not None:
            self.state.hi = float(hi)
        self._render()

    def reset_threshold(self) -> None:
        self._reset_range()
        self._render()

    def toggle_clip(self) -> None:
        self.state.clip = not self.state.clip
        self._render()

    def play(self, fps: float = 6.0) -> None:
        """Walk the schedule from the current step to the end."""
        import time as _time
        for k in range(self.state.step, self.n_steps):
            self.set_step(k)
            if self.qt:
                self.pl.app.processEvents()
            _time.sleep(1.0 / fps)

    # -- histogram -------------------------------------------------------
    def toggle_histogram(self) -> None:
        if self._chart is not None:
            self.pl.remove_chart(self._chart)
            self._chart = None
            self.pl.render()
            return
        self._chart = pv.Chart2D()
        self._chart.background_color = (1.0, 1.0, 1.0, 0.7)
        self.pl.add_chart(self._chart)
        self._draw_histogram()
        self.pl.render()

    def histogram(self, bins: int = 40):
        """Counts and bin edges of the values now on screen."""
        v = self.values()
        v = v[np.isfinite(v)]
        v = v[threshold_mask(v, self.state.lo, self.state.hi)]
        if v.size == 0:
            return np.zeros(bins), np.linspace(0.0, 1.0, bins + 1)
        return np.histogram(v, bins=bins)

    def _draw_histogram(self) -> None:
        counts, edges = self.histogram()
        self._chart.clear()
        centres = 0.5 * (edges[:-1] + edges[1:])
        self._chart.bar(centres, counts, color="tab:blue")
        self._chart.x_label = self.state.field
        self._chart.y_label = "cells"

    # -- widgets ---------------------------------------------------------
    def _add_widgets(self) -> None:
        if self.n_steps > 1:
            self.pl.add_slider_widget(
                lambda v: self.set_step(int(round(v))),
                rng=[0, self.n_steps - 1], value=0, title="timestep",
                pointa=(0.72, 0.90), pointb=(0.97, 0.90), fmt="%.0f",
                style="modern")

        self.pl.add_key_event("n", lambda: self.next_field(+1))
        self.pl.add_key_event("p", lambda: self.next_field(-1))
        self.pl.add_key_event("t", self.next_transform)
        self.pl.add_key_event("c", self.toggle_clip)
        self.pl.add_key_event("h", self.toggle_histogram)
        self.pl.add_key_event("r", self.reset_threshold)
        self.pl.add_key_event("space", self.play)
        self.pl.add_key_event("Right", lambda: self.set_step(self.state.step + 1))
        self.pl.add_key_event("Left", lambda: self.set_step(self.state.step - 1))

        keys = ("n/p field   t transform   c clip   h histogram   "
                "r reset   space play")
        self.pl.add_text(keys, font_size=8, position="lower_left")

        if self.qt:
            self._add_menus()

    def _add_menus(self) -> None:
        """Real menus, the closest thing to plotToolbar's toolbar."""
        menu = self.pl.main_menu.addMenu("Dataset")
        for name in self.names:
            menu.addAction(name, lambda n=name: self.set_field(n))

        tmenu = self.pl.main_menu.addMenu("Transform")
        for mode in TRANSFORMS:
            tmenu.addAction(mode, lambda m=mode: self.set_transform(m))

        vmenu = self.pl.main_menu.addMenu("View")
        vmenu.addAction("Clip plane", self.toggle_clip)
        vmenu.addAction("Histogram", self.toggle_histogram)
        vmenu.addAction("Reset threshold", self.reset_threshold)
        if self.n_steps > 1:
            vmenu.addAction("Play", self.play)

    # -- lifecycle -------------------------------------------------------
    def screenshot(self, filename: str) -> str:
        self.pl.screenshot(filename)
        return filename

    def show(self, block: bool = True):
        if self.off_screen:
            return self
        if self.qt:
            if block:
                self.pl.app.exec_()
            return self
        self.pl.show()
        return self

    def close(self) -> None:
        self.pl.close()


def plot_toolbar(g: Grid, data, show: bool = True, **kwargs) -> PlotToolbar:
    """Open the interactive inspector (plotToolbar)."""
    tb = PlotToolbar(g, data, **kwargs)
    if show and not tb.off_screen:
        tb.show()
    return tb


if __name__ == "__main__":
    import os
    import tempfile

    pv.OFF_SCREEN = True
    OUT = tempfile.mkdtemp(prefix="pymrst_toolbar_")

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.incomp.fluid import init_single_fluid
    from pymrst.modules.incomp.incomp_tpfa import incomp_tpfa
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.solvers.state import init_res_sol
    from pymrst.solvers.trans import compute_trans
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

    def _case(dims=(8, 8, 2), phys=(80.0, 80.0, 10.0)):
        g = compute_geometry(cart_grid(list(dims), list(phys)))
        # heterogeneous, so ranges and histograms have something to show
        x, y, _ = g.cells.centroids.T
        perm = (50.0 + 200.0 * (1 + np.sin(x / 17.0) * np.cos(y / 23.0))
                ) * MILLI * DARCY
        rock = make_rock(g, perm[:, None], np.full((g.cells.num, 1), 0.2))
        T = compute_trans(g, rock)
        fluid = init_single_fluid(mu=1e-3, rho=1000.0)
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=100.0 / DAY, compi=np.array([1.0]))
        W = add_well(W, g, rock, np.array([g.cells.num - 1]),
                     well_type="bhp", val=100 * BARSA, compi=np.array([1.0]))
        st = incomp_tpfa(init_res_sol(g, 100 * BARSA), g, T, fluid, wells=W)
        return g, rock, st

    def test_normalize_single_field():
        """One array becomes one static field."""
        g, _, st = _case()
        f = normalize_data(st.pressure, g.cells.num)
        assert list(f) == ["data"]
        assert f["data"].shape == (1, g.cells.num)
        assert np.allclose(f["data"][0], st.pressure)
        print("ok test_normalize_single_field")

    def test_normalize_dict_and_time_series():
        """A dict mixes static and time-varying fields."""
        g, _, st = _case()
        nc = g.cells.num
        series = np.stack([st.pressure * (1 + 0.1 * k) for k in range(5)])
        f = normalize_data({"p": st.pressure, "p_t": series}, nc)
        assert f["p"].shape == (1, nc) and f["p_t"].shape == (5, nc)

        for bad, msg in ((np.zeros(nc + 1), "one per cell"),
                         (np.zeros((3, nc + 2)), "n_steps")):
            try:
                normalize_data({"x": bad}, nc)
            except ValueError as e:
                assert msg in str(e), e
            else:
                raise AssertionError("bad shape should have been rejected")
        print("ok test_normalize_dict_and_time_series")

    def test_states_to_fields_picks_cell_data_only():
        """State fields are found; face fluxes and well sols are ignored."""
        g, _, st = _case()
        nc = g.cells.num
        states = [st, st, st]
        f = states_to_fields(states, nc)
        # incomp State carries pressure (nc,), s (nc, 1), flux (nf,), wellSol
        assert "pressure" in f and "s[0]" in f
        assert "flux" not in f, "face data must not be taken for cell data"
        assert "wellSol" not in f
        assert f["pressure"].shape == (3, nc)
        assert np.allclose(f["pressure"][0], st.pressure)
        print(f"ok test_states_to_fields_picks_cell_data_only "
              f"(found {sorted(f)})")

    def test_transforms():
        """log10 drops non-positive values instead of inventing one."""
        v = np.array([-10.0, 0.0, 1.0, 100.0])
        assert np.allclose(apply_transform(v, "linear"), v)
        assert np.allclose(apply_transform(v, "abs"), [10, 0, 1, 100])

        lg = apply_transform(v, "log10")
        assert np.isnan(lg[0]) and np.isnan(lg[1])
        assert np.allclose(lg[2:], [0.0, 2.0])

        la = apply_transform(v, "log10(abs)")
        assert np.allclose(la[0], 1.0) and np.isnan(la[1])
        assert np.allclose(la[2:], [0.0, 2.0])

        try:
            apply_transform(v, "sqrt")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown transform should raise")
        print("ok test_transforms")

    def test_threshold_mask():
        """NaN never passes a threshold."""
        v = np.array([0.0, 1.0, np.nan, 5.0])
        m = threshold_mask(v, 0.5, 4.0)
        assert m.tolist() == [False, True, False, False]
        print("ok test_threshold_mask")

    def test_toolbar_state_and_selection():
        """Switching field, step, transform and threshold changes what shows."""
        g, rock, st = _case()
        nc = g.cells.num
        series = np.stack([st.pressure + k * 1e5 for k in range(4)])
        tb = PlotToolbar(g, {"pressure": series,
                             "perm": rock.perm[:, 0]}, off_screen=True)

        assert tb.n_steps == 4 and tb.names == ["pressure", "perm"]
        assert np.allclose(tb.values(), series[0])

        tb.set_step(3)
        assert np.allclose(tb.values(), series[3])

        # a static field ignores the timestep rather than running off its end
        tb.set_field("perm")
        assert np.allclose(tb.values(), rock.perm[:, 0])
        assert tb.state.step == 3

        # switching field must rescale, or the new field would be off-scale
        assert np.isclose(tb.state.lo, rock.perm[:, 0].min())
        assert np.isclose(tb.state.hi, rock.perm[:, 0].max())

        tb.set_transform("log10")
        assert np.isclose(tb.state.hi,
                          np.log10(rock.perm[:, 0].max()), rtol=1e-12)

        # thresholding hides cells
        tb.set_field("pressure")
        tb.set_step(0)
        mid = 0.5 * (tb.state.lo + tb.state.hi)
        tb.set_threshold(lo=mid)
        vis = tb.visible_cells()
        assert 0 < vis.size < nc
        assert np.all(series[0][vis] >= mid)
        tb.reset_threshold()
        assert tb.visible_cells().size == nc
        tb.close()

        # a constant field has zero range; it must still be showable
        flat = PlotToolbar(g, {"c": np.full(nc, 7.0)}, off_screen=True)
        assert flat.state.hi > flat.state.lo, "constant field collapsed"
        assert flat.visible_cells().size == nc
        flat.close()
        print("ok test_toolbar_state_and_selection")

    def test_colour_range_spans_the_whole_series():
        """The scale covers all steps, not just the one on screen.

        A saturation starts uniform, so scaling on step 0 alone would fix
        the colours on a degenerate range and then never correct itself -
        the bug this test exists to prevent. Scaling per step would be no
        better: the colours would shift while scrubbing and no two frames
        could be compared.
        """
        g, _, _ = _case()
        nc = g.cells.num
        # uniform at step 0, spreading out over 0.15 .. 0.85 afterwards
        sw = np.stack([np.full(nc, 0.15)]
                      + [np.linspace(0.15, 0.15 + 0.7 * k / 3, nc)
                         for k in range(1, 4)])
        tb = PlotToolbar(g, {"sW": sw}, off_screen=True)

        assert np.isclose(tb.state.lo, 0.15)
        assert np.isclose(tb.state.hi, 0.85), tb.state.hi
        assert tb.state.hi <= 1.0 + 1e-12, "saturation scale ran past 1"

        # and the range does not move as the step changes
        fixed = (tb.state.lo, tb.state.hi)
        for k in range(tb.n_steps):
            tb.set_step(k)
            assert np.allclose((tb.state.lo, tb.state.hi), fixed), (
                "colour range must not move while scrubbing")

        tb.close()
        print("ok test_colour_range_spans_the_whole_series")

    def test_toolbar_renders_and_histogram():
        """The window draws, and the histogram counts only visible cells."""
        g, _, st = _case()
        tb = PlotToolbar(g, {"pressure": st.pressure}, off_screen=True)

        counts, edges = tb.histogram(bins=10)
        assert counts.sum() == g.cells.num

        mid = 0.5 * (tb.state.lo + tb.state.hi)
        tb.set_threshold(lo=mid)
        counts, _ = tb.histogram(bins=10)
        assert counts.sum() == tb.visible_cells().size < g.cells.num

        f = os.path.join(OUT, "toolbar.png")
        tb.screenshot(f)
        assert os.path.getsize(f) > 5000
        tb.close()
        print(f"ok test_toolbar_renders_and_histogram -> {f}")

    def test_toolbar_from_a_schedule():
        """A list of states drives the toolbar directly."""
        g, _, st = _case()
        states = []
        for k in range(3):
            s = init_res_sol(g, 100 * BARSA)
            s.pressure = st.pressure + k * 1e5
            states.append(s)

        tb = PlotToolbar(g, states, times=[0.0, 30 * DAY, 60 * DAY],
                         time_unit=DAY, time_label="days", off_screen=True)
        assert tb.n_steps == 3
        assert "pressure" in tb.names
        tb.set_step(2)
        assert np.allclose(tb.values(), states[2].pressure)
        assert "60" in tb._label() and "days" in tb._label()
        tb.close()
        print("ok test_toolbar_from_a_schedule")

    def test_toolbar_reads_graph_states():
        """The state-function graph's `State` is a dict, not a dataclass.

        `states_to_fields` originally read `__dict__`, which on a dict
        subclass returns the internals rather than the fields - so every
        model built on the graph (black-oil, poromechanics) was invisible to
        the inspector. It must pick the per-cell entries and skip the rest:
        `xd` and `u` are sized by *nodes*, `bhp` and `qWs` by *wells*.
        """
        from pymrst.modules.ad_core.statefunctions.state_function import State

        g, _, _ = _case()
        nc = g.cells.num
        states = [
            State(pressure=np.full(nc, 100e5 + k * 1e5),
                  sW=np.full(nc, 0.2 + 0.1 * k),
                  xd=np.zeros(3 * g.nodes.num - 12),   # node-sized, not cells
                  u=np.zeros(3 * g.nodes.num),
                  bhp=np.array([300e5, 100e5]),        # well-sized
                  qWs=np.array([1e-3, -1e-3]))
            for k in range(3)
        ]

        f = states_to_fields(states, nc)
        assert set(f) == {"pressure", "sW"}, f"picked up junk: {sorted(f)}"
        assert f["pressure"].shape == (3, nc)

        tb = PlotToolbar(g, states, off_screen=True)
        assert tb.n_steps == 3 and "sW" in tb.names
        tb.set_step(2)
        assert np.allclose(tb.values(), states[2]["pressure"])
        tb.close()
        print("ok test_toolbar_reads_graph_states")

    test_normalize_single_field()
    test_normalize_dict_and_time_series()
    test_states_to_fields_picks_cell_data_only()
    test_toolbar_reads_graph_states()
    test_transforms()
    test_threshold_mask()
    test_toolbar_state_and_selection()
    test_colour_range_spans_the_whole_series()
    test_toolbar_renders_and_histogram()
    test_toolbar_from_a_schedule()
    print("\nAll tests passed!")
