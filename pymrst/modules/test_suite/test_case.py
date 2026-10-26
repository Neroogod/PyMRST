"""MRST ``test-suite/TestCase.m``.

A test case: the state, model and schedule a setup function in
`test_suite/setup_functions` builds (`TestCase('spe10_wo', layers=1)` calls
`spe10_wo(layers=1)`), with its name, description and options, and how to plot
it. Options of `TestCase` itself are `plot`, `readFromDisk` (load a saved copy
of the same setup if there is one), `writeToDisk` (save it) and `computeID`
(hash the setup, so a later change can be detected), plus its properties;
everything else goes to the setup function. A saved test case lives in
`<dataDirectory>/test-suite/<setup function>/<name>/<hash>`, as in MRST; PyMRST
pickles it (with `cloudpickle` when installed), where MATLAB writes a `.mat`
file, so the file name ends in `.pkl`.

Plot options from the setup are sorted as MRST sorts them, into figure
properties (and `Size`), axis properties (with limits from the grid and a
default aspect ratio and view) and options for the plotting function
(`plotToolbar`). PyMRST draws with matplotlib; `setAxisProperties` applies the
limits, view, projection and aspect ratio to a matplotlib axis. `writeJutulInput`
needs MRST's Jutul writer, which PyMRST does not port.
"""

from __future__ import annotations

import copy
import importlib
import pickle
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import merge_options
from pymrst.utils.settings_manager.mrst_settings import mrst_settings

FIGURE_PROPERTIES = {
    "Alphamap", "BusyAction", "ButtonDownFcn", "Clipping", "CloseRequestFcn", "Color",
    "Colormap", "ContextMenu", "CreateFcn", "DeleteFcn", "DockControls", "FileName",
    "GraphicsSmoothing", "HandleVisibility", "IntegerHandle", "Interruptible",
    "InvertHardcopy", "KeyPressFcn", "KeyReleaseFcn", "MenuBar", "Name", "NextPlot",
    "NumberTitle", "PaperOrientation", "PaperPosition", "PaperPositionMode", "PaperSize",
    "PaperType", "PaperUnits", "Pointer", "Position", "Renderer", "RendererMode", "Resize",
    "SizeChangedFcn", "Tag", "ToolBar", "Units", "UserData", "Visible",
    "WindowButtonDownFcn", "WindowState", "WindowStyle"}
AXES_PROPERTIES = {
    "ALim", "ALimMode", "AmbientLightColor", "Box", "BoxStyle", "CLim", "CLimMode",
    "CameraPosition", "CameraTarget", "CameraUpVector", "CameraViewAngle", "Clipping",
    "Color", "ColorOrder", "DataAspectRatio", "DataAspectRatioMode", "FontName",
    "FontSize", "FontWeight", "GridColor", "GridLineStyle", "LineWidth", "NextPlot",
    "PlotBoxAspectRatio", "Position", "Projection", "SortMethod", "TickDir", "TickLength",
    "Title", "Units", "View", "Visible", "XAxisLocation", "XColor", "XDir", "XGrid",
    "XLabel", "XLim", "XLimMode", "XScale", "XTick", "XTickLabel", "YAxisLocation",
    "YColor", "YDir", "YGrid", "YLabel", "YLim", "YLimMode", "YScale", "YTick",
    "YTickLabel", "ZColor", "ZGrid", "ZLabel", "ZLim", "ZLimMode", "ZScale", "ZTick",
    "ZTickLabel"}
HASH_SKIP = ["figureProperties", "axisProperties", "toolbarOptions", "visualizationGrid",
             "plotFn", "id", "verbose"]
DESCRIPTION_MAX = 113


class TestCase:
    """MRST `TestCase`."""

    __test__ = False  # not a pytest test class

    def __init__(self, name: Optional[str] = None, *, plot: bool = False,
                 readFromDisk: bool = True, writeToDisk: bool = False,
                 computeID: bool = False, **varargin: Any):
        self.name = None
        self.description = None
        self.state0 = None
        self.model = None
        self.schedule = None
        self.options = None
        self.verbose = None
        self.extra = None
        self.figureProperties: Dict[str, Any] = {}
        self.axisProperties: Dict[str, Any] = {}
        self.toolbarOptions: List[Any] = []
        self.visualizationGrid = None
        self.plotFn = None
        self._baseName = None
        self._id = None
        if name is None:
            return
        self._baseName = name.lower()
        extra = self._merge_own_properties(varargin)
        if self.verbose is None:
            self.verbose = False
        timer = time.perf_counter()
        if self.verbose:
            print(f"Setting up {self._baseName} test \n")
        setup_fn = _setup_function(self._baseName)
        if readFromDisk:
            setup = setup_fn(False, **extra)
            self.options = setup.options
            self.description = setup.description
            if self.name is None:
                self.name = setup.name
            tc = self.load(False)
            if tc is not None:
                if self.verbose:
                    print(f"Test case loaded in {time.perf_counter() - timer:.2f} s\n")
                self.__dict__.update(tc.__dict__)
                return
            if self.verbose:
                print("Did not find a saved version of this test case, setting up from "
                      "scratch \n")
        setup = setup_fn(**extra)
        if self.name is None:
            self.name = setup.name
        self.description = setup.description
        self.state0 = setup.state0
        self.model = setup.model
        self.schedule = setup.schedule
        self.options = setup.options
        self.extra = setup.extra
        if self.verbose:
            print(f"Test case set up in {time.perf_counter() - timer:.2f} s\n")
        plot_options = self._merge_own_properties(_pairs_to_dict(setup.plotOptions))
        self.figureProperties, plot_options = self.process_figure_props(**plot_options)
        self.axisProperties, plot_options = self.process_axis_props(**plot_options)
        toolbar = self.default_toolbar_options()
        names = {k.lower(): k for k in toolbar}
        for key, val in plot_options.items():
            if key.lower() not in names:
                raise AttributeError(f"Unknown plot option {key}")
            toolbar[names[key.lower()]] = val
        self.toolbarOptions = [x for kv in toolbar.items() for x in kv]
        if self.plotFn is None:
            self.plotFn = self._default_plot_fn
        if computeID:
            self._id = self.get_test_case_hash()
        if writeToDisk:
            self.save()
        if plot:
            self.plot(self.model.rock, Name="rock", log10=True)
            self.plot(self.state0, Name="state0")

    def _merge_own_properties(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """MATLAB `[test, extra] = merge_options(test, ...)`: set this test
        case's public properties, return the rest."""
        public = {k.lower(): k for k in vars(self) if not k.startswith("_")}
        rest = {}
        for key, val in options.items():
            if key.lower() in public:
                setattr(self, public[key.lower()], val)
            else:
                rest[key] = val
        return rest

    @property
    def id(self) -> Optional[str]:
        return self._id

    def valid_properties(self, kind: str, args: Dict[str, Any],
                         extra: Sequence[str] = ()) -> Tuple[Dict[str, Any], Dict[str, Any], set]:
        """Split `args` into the `kind` ('Figure' or 'Axes') properties and the rest."""
        valid = set(FIGURE_PROPERTIES if kind == "Figure" else AXES_PROPERTIES) | set(extra)
        props = {k: v for k, v in args.items() if k in valid}
        other = {k: v for k, v in args.items() if k not in valid}
        return props, other, valid

    def process_figure_props(self, **varargin: Any):
        props, other, _ = self.valid_properties("Figure", varargin, ["Size", "XDisplay"])
        return props, other

    def figure(self, **varargin: Any):
        """A matplotlib figure with this test case's figure properties."""
        import matplotlib.pyplot as plt
        fig = plt.figure(num=varargin.pop("Name", None))
        for name, val in self.figureProperties.items():
            if name.lower() == "size":
                fig.set_size_inches(np.asarray(val, dtype=float)/fig.dpi)
            elif name.lower() == "name":
                fig.canvas.manager.set_window_title(str(val))
        return fig

    def process_axis_props(self, **varargin: Any):
        props, other, valid = self.valid_properties(
            "Axes", varargin, ["Legend", "Colormap", "ZDir", "PlotBoxAspectRatioMode"])
        G = self.get_visualization_grid()
        nodes = getattr(G, "nodes", None)
        if nodes is not None:
            coords = np.asarray(nodes.coords, dtype=float)
        elif getattr(G, "parent", None) is not None:
            coords = np.asarray(G.parent.nodes.coords, dtype=float)
        else:
            raise ValueError("Unable to plot grid")
        xmax, xmin = coords.max(axis=0), coords.min(axis=0)
        if self.has_driving_force("W") and G.griddim == 3:
            xmin[2] = xmin[2] - 0.15*(xmax[2] - xmin[2])

        def update_prop(name: str) -> bool:
            return name not in props and name in valid

        nm = None
        for i in range(G.griddim):
            nm = "XYZ"[i] + "Lim"
            if update_prop(nm):
                props[nm] = [xmin[i], xmax[i]]
        if update_prop("PlotBoxAspectRatio"):
            props["PlotBoxAspectRatio"] = [1, 1, 0.25]
        if update_prop("View"):
            props["View"] = [0, 90] if G.griddim == 2 else [-37.5, 30]
        # MRST tests `nm`, the last limit just set, so a default projection is
        # never applied; kept.
        if update_prop(nm):
            props["Projection"] = "orthographic" if G.griddim == 2 else "perspective"
        return props, other

    def set_axis_properties(self, ax: Any) -> Any:
        """Apply the axis properties to a matplotlib axis."""
        is3d = hasattr(ax, "set_zlim")
        for name, val in self.axisProperties.items():
            key = name.lower()
            if key in ("xlim", "ylim") or (key == "zlim" and is3d):
                getattr(ax, f"set_{key}")(val)
            elif key == "view" and is3d:
                ax.view_init(elev=val[1], azim=val[0])
            elif key == "projection" and is3d:
                ax.set_proj_type("persp" if val == "perspective" else "ortho")
            elif key == "plotboxaspectratio":
                val = np.asarray(val, dtype=float)
                if is3d:
                    ax.set_box_aspect(val)
                else:
                    ax.set_box_aspect(val[1]/val[0])
        return ax

    def default_toolbar_options(self) -> Dict[str, Any]:
        return {"log10": False, "exp": False, "abs": False, "filterzero": False,
                "logical": False, "outline": False, "pauseTime": 0.150, "lockCaxis": False,
                "plot1d": False, "surf": False, "plotmarkers": False, "skipAugmented": False,
                "field": "s:1", "step_index": 1, "startplayback": False}

    def plot(self, v: Any = None, *, Name: str = "", plotWells: bool = True,
             wellOpts: Sequence[Any] = (), camlight: bool = True, **varargin: Any):
        """Plot `v` (the rock by default) on the grid, with the wells."""
        if v is None:
            v = self.model.rock
        name = self.name if not Name else f"{self.name} {Name}"
        fig = self.figure(Name=name)
        G = self.get_visualization_grid()
        self.plotFn(G, v, **varargin)
        if plotWells:
            self.plot_wells(*wellOpts)
        import matplotlib.pyplot as plt
        self.set_axis_properties(plt.gca())
        return fig

    def _default_plot_fn(self, G: Any, v: Any, **varargin: Any) -> Any:
        from pymrst.plotting.extensions.plot_toolbar import plot_toolbar
        options = dict(zip(self.toolbarOptions[0::2], self.toolbarOptions[1::2]))
        options.update(varargin)
        return plot_toolbar(G, v, **options)

    def get_visualization_grid(self) -> Any:
        return self.model.G if self.visualizationGrid is None else self.visualizationGrid

    def plot_wells(self, **varargin: Any):
        """Mark the wells of the first control: markers with names in 2D,
        `plotWell` in 3D."""
        if not self.has_driving_force("W"):
            return None
        W = _control_field(self.schedule.controls[0], "W")
        G = self.get_visualization_grid()
        import matplotlib.pyplot as plt
        ax = plt.gca()
        if G.griddim == 3:
            from pymrst.plotting.plot_well import plot_well
            zlim = ax.get_zlim() if hasattr(ax, "get_zlim") else (0, 1)
            return plot_well(G, W, color="k", height=0.15*(zlim[1] - zlim[0]), **varargin)
        fontsize = varargin.pop("fontsize", 12)
        color = varargin.pop("color", "k")
        color2 = varargin.pop("color2", "k")
        hw, htext = [], []
        for w in W:
            c = color2 if w.sign < 0 else color
            x = np.asarray(G.cells.centroids)[np.atleast_1d(w.cells)[0]]
            hw.append(ax.plot(x[0], x[1], "o", color=c, markerfacecolor=[0.8]*3,
                              markersize=8, **varargin)[0])
            if fontsize > 0:
                htext.append(ax.text(x[0], x[1], w.name, fontsize=fontsize,
                                     verticalalignment="bottom", color=c))
        return hw, htext

    def has_driving_force(self, force: str) -> bool:
        control = self.schedule.controls[0]
        val = _control_field(control, force)
        return val is not None and np.size(val) > 0 if not isinstance(val, list) else len(val) > 0

    def get_test_case_hash(self, full_hash: bool = True):
        from pymrst.modules.ad_core.utils.obj2hash import obj2hash
        skip = list(HASH_SKIP) + ["_id"]
        if not full_hash:
            skip += ["model", "state0", "schedule", "extra"]
        return obj2hash(self, skip=skip)

    def is_modified(self) -> int:
        if self._id is None:
            return -1
        return int(self.get_test_case_hash() != self._id)

    def save(self, directory: Optional[str] = None, name: Optional[str] = None,
             prompt: bool = True) -> Optional[Path]:
        name, directory, is_saved = self.get_file_parts(name, directory)
        fn = None
        if prompt:
            rmsg = "Ok, will not save test case."
            if is_saved:
                warnings.warn("Found an existing version of this test case in the same "
                              "location. This will be overwritten")
                fn = Path(directory)/name
            modified = self.is_modified()
            if modified == 1:
                if input("Test case has been changed after construction. Do you still want "
                         "to save it? y/n [n]: ").lower() != "y":
                    print(rmsg)
                    return None
            elif modified == -1:
                if input("ID not computed -- cannot determine if test case has been changed "
                         "after construction. Do you still want to save it? y/n [n]: "
                         ).lower() != "y":
                    print(rmsg)
                    return None
            size = self.get_size()
            prefix = ["k", "M", "G"]
            sz = size/np.array([1e3, 1e6, 1e9])
            om = int(np.flatnonzero(sz > 1)[-1]) if np.any(sz > 1) else 0
            prompt_text = (f"\nSaving test case {self.name} \nLocation  : {directory}\n"
                           f"File name : {name}\nSize      : {sz[om]:.2f} {prefix[om]}B \n\n"
                           f"Continue? y/n [n]: ")
            if input(prompt_text).lower() != "y":
                print(rmsg)
                return None
        Path(directory).mkdir(parents=True, exist_ok=True)
        if fn is None:
            fn = Path(directory)/name
        with fn.open("wb") as stream:
            _dump(self, stream)
        return fn

    def get_size(self) -> int:
        buffer = __import__("io").BytesIO()
        _dump(self, buffer)
        return buffer.tell()

    def load(self, throw_error: bool = True) -> Optional["TestCase"]:
        name, directory, is_saved = self.get_file_parts()
        path = Path(directory)/name
        if is_saved:
            if self.verbose:
                print("Found a saved version of this test case. Loading ... ", end="")
            with path.open("rb") as stream:
                return pickle.load(stream)
        if throw_error:
            raise FileNotFoundError(f"Could not find test case named {self.name} with hash "
                                    f"key {name}")
        return None

    def get_file_parts(self, file_name: Optional[str] = None,
                       directory: Optional[str] = None) -> Tuple[str, str, bool]:
        if not file_name:
            file_name = self.get_test_case_hash(False)
        if not directory:
            directory = str(Path(mrst_settings("get", "dataDirectory"))/"test-suite"
                            / self._baseName/self.name)
        ext = Path(file_name).suffix
        if not ext:
            file_name = f"{file_name}.pkl"
        else:
            assert ext.lower() == ".pkl", "File extension must be .pkl"
        return file_name, directory, (Path(directory)/file_name).is_file()

    def get_packed_simulation_problem(self, *, Name: Any = None, useHash: Any = None,
                                      LinearSolver: Any = None, NonLinearSolver: Any = None,
                                      **varargin: Any) -> Any:
        from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver as NLS
        from pymrst.modules.ad_core.ad_oo.wrapper_model import WrapperModel
        from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
            pack_simulation_problem,
        )
        from pymrst.modules.ad_core.solvers.backslash_solver_ad import BackslashSolverAD
        from pymrst.modules.ad_core.utils.select_linear_solver_ad import (
            select_linear_solver_ad,
        )
        name = self.name if Name is None else Name
        if useHash is None:
            useHash = bool(mrst_settings("get", "useHash"))
        if useHash:
            name = f"{name}_" if name else ""
            hash_value = self.get_test_case_hash()
            name = name + hash_value
        has_ls = LinearSolver is not None
        has_nls = NonLinearSolver is not None
        if has_nls:
            has_ls = has_ls or not isinstance(NonLinearSolver.LinearSolver, BackslashSolverAD)
        if not has_ls:
            rmodel = self.model
            if isinstance(rmodel, WrapperModel):
                rmodel = rmodel.get_reservoir_model()
            try:
                LinearSolver = select_linear_solver_ad(rmodel)
            except Exception as ex:  # MRST catches any selection failure
                warnings.warn(f"Unable to select linear solver from model. Reason: {ex} \n"
                              f"Proceeding with BackslashSolverAD")
                LinearSolver = BackslashSolverAD()
        if not has_nls:
            NonLinearSolver = NLS(LinearSolver=LinearSolver, useRelaxation=True)
        elif not has_ls:
            NonLinearSolver.LinearSolver = LinearSolver
        desc = self.description[:DESCRIPTION_MAX]
        if len(desc) < len(self.description):
            desc = desc + " ..."
        problem = pack_simulation_problem(self.state0, self.model, self.schedule, self._baseName,
                                          Name=name, Description=desc, useHash=False,
                                          NonLinearSolver=NonLinearSolver, **varargin)
        if useHash:
            directory = problem.OutputHandlers.states.get_data_path()
            with (Path(directory)/"setup_hash.pkl").open("wb") as stream:
                pickle.dump({"hash": hash_value}, stream)
        return problem

    def write_jutul_input(self, **varargin: Any):
        raise NotImplementedError("writeJutulInput (MRST's Jutul writer) is not ported")

    validProperties = valid_properties
    processFigureProps = process_figure_props
    processAxisProps = process_axis_props
    setAxisProperties = set_axis_properties
    defaultToolbarOptions = default_toolbar_options
    getVisualizationGrid = get_visualization_grid
    plotWells = plot_wells
    hasDrivingForce = has_driving_force
    getTestCaseHash = get_test_case_hash
    isModified = is_modified
    getSize = get_size
    getFileParts = get_file_parts
    getPackedSimulationProblem = get_packed_simulation_problem
    writeJutulInput = write_jutul_input


def _setup_function(name: str):
    module = importlib.import_module(f"pymrst.modules.test_suite.setup_functions.{name}")
    return getattr(module, name)


def _pairs_to_dict(pairs: Sequence[Any]) -> Dict[str, Any]:
    pairs = list(pairs or [])
    return dict(zip(pairs[0::2], pairs[1::2]))


def _control_field(control: Any, name: str) -> Any:
    if isinstance(control, dict):
        return control.get(name)
    return getattr(control, name, None)


def _dump(obj: Any, stream: Any) -> None:
    try:
        import cloudpickle
    except ImportError:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        cloudpickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = ["TestCase"]


if __name__ == "__main__":
    import contextlib
    import io
    import tempfile
    from unittest import mock

    from pymrst.params.gravity.gravity import isolated_gravity

    def _case(**kw):
        with isolated_gravity(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TestCase("spe10_wo", layers=1, J=np.arange(1, 11), make2D=True, **kw)

    def test_builds_the_setup_and_sorts_plot_options():
        test = _case()
        assert test.name == "spe10_wo" and test.model.G.griddim == 2
        assert test.figureProperties == {"Size": [500, 800]}
        assert test.axisProperties["View"] == [0, 90]
        assert "Projection" in test.axisProperties and "XLim" in test.axisProperties
        assert test.options["extra"] == {"J": test.options["extra"]["J"], "make2D": True}
        assert test.has_driving_force("W") and not test.has_driving_force("bc")
        print("ok test_builds_the_setup_and_sorts_plot_options")

    def test_save_and_load_round_trip():
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch(f"{__name__}.mrst_settings", lambda *a: tmp):
            test = _case(computeID=True)
            assert test.is_modified() == 0
            with mock.patch("builtins.input", return_value="y"), \
                    contextlib.redirect_stdout(io.StringIO()):
                path = test.save()
            assert path.suffix == ".pkl" and path.parent.parent.name == "spe10_wo"
            again = _case()
            assert again.id == test.id and again.model is not None
        print("ok test_save_and_load_round_trip")

    def test_packed_problem_uses_the_setup_name():
        test = _case()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("pymrst.modules.ad_core.simulators.sim_runner."
                           "pack_simulation_problem.mrst_settings",
                           lambda *a: False if a[1] == "useHash" else tmp):
            problem = test.get_packed_simulation_problem(useHash=False)
            assert problem.BaseName == "spe10_wo" and problem.Name == "spe10_wo"
            assert problem.Description.startswith("SPE10 Model 2") and problem.Description.endswith(" ...")
        print("ok test_packed_problem_uses_the_setup_name")

    test_builds_the_setup_and_sorts_plot_options()
    test_save_and_load_round_trip()
    test_packed_problem_uses_the_setup_name()
    print("\nAll tests passed!")
