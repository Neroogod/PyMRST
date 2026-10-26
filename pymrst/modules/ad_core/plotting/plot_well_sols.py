"""MRST ``ad-core/plotting/plotWellSols.m``.

An interactive viewer for well solutions: one well property (BHP, a surface
or reservoir rate, a cut, ...) against time for the selected wells, from
one or more datasets. MRST builds the selection panel from `uicontrol`s;
this port builds the same controls from matplotlib widgets:
- radio buttons for the property, unit system and time scale;
- check boxes for the wells, the datasets and the Misc toggles (grid, log x,
  log y, markers, legend, cumulative sum, absolute value, zoom, stairs,
  control changes, use timesteps);
- a line-width slider;
- a save button.

The drawing, the units and the cumulative sums follow MRST line for line.
"Save to workspace" (`export2wsdlg`) has no Python counterpart; the plotted
data is kept on the returned figure as `figure.currentdata`.

`plot_well_sols(wellSols, time, **options)` returns `(figure, inject)`, where
`inject(wellSols, time)` replaces the datasets and redraws.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from pymrst.utils.units import (
    ATM, BARSA, CENTI, DAY, FT, HOUR, METER, MINUTE, PASCAL, PSIA, SECOND,
    STB, YEAR,
)

DEFAULTS: Dict[str, Any] = dict(
    lowermargin=.1, plotwidth=.6, linewidth=2, field="bhp", SelectedWells=1,
    linestyles=("-", "--", "-.", ":"),
    markerstyles=("o", "x", "d", "s", "d", "^", "v", ">", "<", "*"),
    timescale="days", grid=True, logx=False, logy=False, marker=False,
    legend=True, cumsum=False, abs=True, zoom=False, stairs=False, ctrl=False,
    figure=None, datasetnames=())
TIME_CHOICES = ("Years", "Days", "Hours", "Minutes", "Seconds")
TIME_SCALES = (YEAR, DAY, HOUR, MINUTE, SECOND)
UNIT_SYSTEMS = ("Metric", "SI", "Field", "Lab")
MISC_TOGGLES = ("grid", "logx", "logy", "marker", "legend", "cumsum", "abs",
                "zoom", "stairs", "ctrl")
MISC_LABELS = ("Grid on", "Log (x)", "Log (y)", "Markers", "Legend",
               "Cumulative sum", "Absolute value", "Zoom to data",
               "Stair-step plot", "Show control changes")
#: MATLAB's `lines` colour order.
MATLAB_LINES = np.array([[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980],
                         [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
                         [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330],
                         [0.6350, 0.0780, 0.1840]])


def plot_well_sols(wellsols, *args, **options) -> Tuple[Any, Callable]:
    """MRST `plotWellSols(wellSols, time, ...)`: the interactive viewer."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

    opt = dict(DEFAULTS)
    if _is_packed(wellsols):
        from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
            get_multiple_packed_simulator_outputs,
        )
        problems = wellsols if isinstance(wellsols, (list, tuple)) else [wellsols]
        out = get_multiple_packed_simulator_outputs(problems, readFromDisk=False,
                                                    readWellSolsFromDisk=True)
        wellsols, names, timesteps = out[0], out[3], out[4]
        has_timesteps = True
        opt["datasetnames"] = names
    elif args:
        timesteps, has_timesteps = args[0], True
    else:
        timesteps, has_timesteps = None, False
    if not _is_list_of_datasets(wellsols):
        wellsols = [wellsols]
    ndata = len(wellsols)
    timesteps = _validate_timesteps(wellsols, timesteps)
    timesteps = [np.cumsum(t) if t is not None and np.any(np.diff(t) <= 0) else t
                 for t in timesteps]
    plotvararg = {k: v for k, v in options.items() if k not in DEFAULTS}
    opt.update({k: v for k, v in options.items() if k in DEFAULTS})
    if not wellsols[0] or not wellsols[0][0]:
        print("No wellSols present in input.")
        return None, None
    fn = _get_names_from_ws(wellsols[0][0][0])
    lowered = [f.lower() for f in fn]
    if opt["field"].lower() in lowered:
        fn_index = lowered.index(opt["field"].lower())
    else:
        import warnings
        warnings.warn(f"Unknown field '{opt['field']}'.")
        fn_index = 0
    all_well_names = [[w.name for w in ws[0]] for ws in wellsols]
    wellnames = all_well_names[0]
    if not opt["datasetnames"]:
        opt["datasetnames"] = [f"data{k + 1}" for k in range(ndata)]
    if opt["figure"] is None:
        width, height = plt.rcParams["figure.figsize"]
        fh = plt.figure(figsize=(1.75*width, height))
    else:
        fh = opt["figure"]
    lm, pw = opt["lowermargin"], opt["plotwidth"]
    plotaxis = fh.add_axes([.75*lm, lm, pw, 1 - 2*lm])
    panel = [lm + pw, .25*lm, 1 - 1.25*lm - pw, 1 - 1.25*lm]
    sub = lambda x, y, w, h: fh.add_axes([panel[0] + x*panel[2], panel[1] + y*panel[3],
                                          w*panel[2], h*panel[3]])
    selected = np.atleast_1d(opt["SelectedWells"]).astype(int) - 1
    controls = dict(
        field=RadioButtons(sub(0.0, .55, .5, .4), fn, active=fn_index),
        unit=RadioButtons(sub(.5, .82, .25, .13), UNIT_SYSTEMS, active=0),
        time=RadioButtons(sub(.75, .82, .25, .13),
                          TIME_CHOICES, active=[t.lower() for t in TIME_CHOICES]
                          .index(opt["timescale"].lower())),
        wells=CheckButtons(sub(.5, .45, .5, .35), wellnames,
                           [k in selected for k in range(len(wellnames))]))
    if ndata > 1:
        controls["data"] = CheckButtons(sub(.5, .3, .5, .14), opt["datasetnames"],
                                        [True]*ndata)
    misc_labels = list(MISC_LABELS)
    misc_values = [bool(opt[k]) for k in MISC_TOGGLES]
    # MRST adds this box `if hasTimesteps || nargout > 1`, checked when
    # there are timesteps. The port always returns the injector (nargout 2),
    # so it is always there.
    misc_labels.append("Use timesteps")
    misc_values.append(bool(has_timesteps))
    controls["misc"] = CheckButtons(sub(0.0, .1, .5, .44), misc_labels, misc_values)
    width_axis = sub(.58, .17, .34, .04)
    width_axis.set_title("Plot line width", fontsize="small")
    controls["width"] = Slider(width_axis, "", 0, 10, valinit=opt["linewidth"])
    save_button = Button(sub(.55, .05, .4, .07), "Save as image")
    state = {"wellsols": wellsols, "timesteps": timesteps, "has_timesteps": has_timesteps,
             "legend": None}

    def misc(label: str) -> bool:
        return controls["misc"].get_status()[misc_labels.index(label)]

    def draw_plot(*_):
        wellsols, timesteps = state["wellsols"], state["timesteps"]
        fld = controls["field"].value_selected
        wells = [w for w, on in zip(wellnames, controls["wells"].get_status()) if on]
        nw = len(wells)
        ncolors = ndata if nw == 1 else nw
        cmap = (MATLAB_LINES[np.arange(ncolors) % len(MATLAB_LINES)] if ncolors < 8
                else plt.get_cmap("nipy_spectral")(np.linspace(0, 1, ncolors + 2))[:, :3])
        markerstyles = opt["markerstyles"] if misc("Markers") else ("",)
        plotaxis.cla()
        labels: List[str] = []
        tit = ""
        currentdata: List[Any] = [None]*ndata
        Mv, mv = -np.inf, np.inf
        data_iter = [0] if ndata == 1 else [
            k for k, on in enumerate(controls["data"].get_status()) if on]
        for i in data_iter:
            for j, wname in enumerate(wells):
                line = opt["linestyles"][i % len(opt["linestyles"])]
                m = markerstyles[i % len(markerstyles)]
                d, wpos = _get_data(wname, all_well_names[i], fld, wellsols[i])
                if state["has_timesteps"] and misc("Use timesteps"):
                    k = TIME_CHOICES.index(controls["time"].value_selected)
                    x = np.asarray(timesteps[i], dtype=float)/TIME_SCALES[k]
                    plotaxis.set_xlabel(f"Time [{TIME_CHOICES[k]}]")
                    xunit = DAY
                else:
                    x = np.arange(1, d.size + 1, dtype=float)
                    xunit = 1
                    plotaxis.set_xlabel("Step #")
                tit, d, yl, do_csum = get_well_unit(d, fld, controls["unit"].value_selected,
                                                    misc("Cumulative sum"),
                                                    state["has_timesteps"])
                plotaxis.set_ylabel(yl)
                if wellsols[i] and hasattr(wellsols[i][0][0], "status"):
                    status, _ = _get_data(wname, all_well_names[i], "status", wellsols[i])
                    active_step = status > 0
                else:
                    active_step = np.ones(d.shape, dtype=bool)
                if do_csum:
                    d = np.cumsum(xunit*np.diff(np.concatenate([[0], x]))*active_step*d)
                d = np.where(active_step, d, np.nan)
                if misc("Absolute value"):
                    d = np.abs(d)
                linew = controls["width"].val
                if linew == 0:
                    line, linew = "", 1
                c = cmap[i] if nw == 1 else cmap[j]
                style = dict(linestyle=line if line else "None", marker=m or None,
                             linewidth=linew, color=c, **plotvararg)
                if misc("Stair-step plot"):
                    plotaxis.step(x, d, where="post", **style)
                else:
                    plotaxis.plot(x, d, **style)
                Mv, mv = max(Mv, np.nanmax(d)), min(mv, np.nanmin(d))
                if misc("Show control changes") and hasattr(wellsols[i][0][0], "type"):
                    _annotate_control_changes(plotaxis, wellsols[i], wpos, x, d, line, c)
                tmp = wname
                if ndata > 1:
                    tmp = opt["datasetnames"][i] if nw == 1 else \
                        f"{tmp} ({opt['datasetnames'][i]})"
                labels.append(tmp)
                currentdata[i] = d[:, None] if currentdata[i] is None else \
                    np.column_stack([currentdata[i], d])
        plotaxis.set_title(tit)
        if misc("Legend") and labels:
            state["legend"] = plotaxis.legend(labels, loc="best")
        plotaxis.grid(misc("Grid on"))
        plotaxis.set_yscale("log" if misc("Log (y)") else "linear")
        plotaxis.set_xscale("log" if misc("Log (x)") else "linear")
        plotaxis.autoscale(tight=True)
        if not misc("Zoom to data") and np.isfinite(Mv) and np.isfinite(mv):
            sM, sm = np.sign(Mv), np.sign(mv)
            if sM == sm:
                ylm = [0, Mv] if sM > 0 else [mv, 0]
            else:
                ylm = [min(mv, 0), max(Mv, 0)]
            eps = np.finfo(float).eps
            plotaxis.set_ylim(ylm[0] - eps, ylm[1] + eps)
        fh.currentdata = currentdata
        fh.selected_wells = wells
        fh.canvas.draw_idle()

    def save_plot_as_figure(*_):
        fld = controls["field"].value_selected
        wells = getattr(fh, "selected_wells", [])
        plotname = fld + "".join(f"_{w}" for w in wells)
        extent = plotaxis.get_tightbbox(fh.canvas.get_renderer()).transformed(
            fh.dpi_scale_trans.inverted())
        fh.savefig(f"{plotname}.jpg", bbox_inches=extent)

    def inject_dataset(ws, steps=None):
        state["has_timesteps"] = steps is not None
        state["wellsols"] = ws if _is_list_of_datasets(ws) else [ws]
        state["timesteps"] = _validate_timesteps(state["wellsols"], steps)
        draw_plot()

    for key in ("field", "unit", "time"):
        controls[key].on_clicked(draw_plot)
    for key in ("wells", "misc") + (("data",) if ndata > 1 else ()):
        controls[key].on_clicked(draw_plot)
    controls["width"].on_changed(draw_plot)
    save_button.on_clicked(save_plot_as_figure)
    fh._plot_well_sols_widgets = (controls, save_button)     # keep widgets alive
    draw_plot()
    return fh, inject_dataset


def get_well_unit(d: np.ndarray, fld: str, usys: str, is_csum: bool,
                  has_timesteps: bool):
    """MRST `getWellUnit`: `(title, data in display units, y label, cumsum)`."""
    do_csum = is_csum
    yl, tit = "", fld
    usys = usys.lower()
    if has_timesteps:
        t_unt, t_str = {"metric": (DAY, "day"), "field": (DAY, "day"),
                        "si": (SECOND, "s"), "lab": (HOUR, "hour")}[usys]
    else:
        t_unt, t_str = 1, "s"
        if is_csum:
            import warnings
            warnings.warn("Timesteps not provided to plotWellSols, cumulative "
                          "sum will not be accurate")
    f = fld.lower()
    if f in ("qws", "qos", "qgs", "rate", "qts", "qwr", "qgr", "qor", "qtr"):
        ph = {"qos": "oil", "qor": "oil", "qws": "water", "qwr": "water",
              "qgs": "gas", "qgr": "gas"}.get(f, "total")
        is_res = len(fld) == 3 and f[2] == "r"
        tmp = "reservoir" if is_res else "surface"
        tit = (f"{fld}: Cumulative {tmp} production ({ph})" if is_csum
               else f"{fld}: Well {tmp} rate ({ph})")
        if usys in ("metric", "si"):
            y_str, y_unit = "m^3", METER**3
        elif usys == "field":
            y_str, y_unit = ("scf", FT**3) if (not is_res and ph == "gas") else ("stb", STB)
        else:
            y_str, y_unit = "cm^3", (CENTI*METER)**3
        if is_csum:
            d, yl = d/y_unit, y_str
        else:
            d, yl = d/(y_unit/t_unt), f"{y_str}/{t_str}"
    elif f in ("bhp", "pressure"):
        tit = f"{fld}: Bottom hole pressure"
        y_unit, yl = _get_pressure_unit(usys)
        d = d/y_unit
    elif f == "gor":
        tit = f"{fld}: Gas/oil ratio at surface conditions"
        if usys in ("metric", "si"):
            yl = "Sm^3/Sm^3"
        elif usys == "field":
            yl, d = "MScf/stb", d/(1000*FT**3/STB)
        else:
            yl = "Scm^3/Scm^3"
    elif f == "t":
        tit, yl = f"{fld}: Temperature", "kelvin"
    elif f in ("ocut", "wcut", "gcut"):
        t = {"o": "Oil", "g": "Gas", "w": "Water"}[f[0]]
        tit = f"{fld}: {t} volume fraction at reservoir conditions"
    elif f == "sign":
        tit = f"{fld}: Well sign (+1 for injector, -1 for producer)"
    elif f == "val":
        tit = f"{fld}: Well control value"
    elif f == "cdp":
        tit = f"{fld}: Pressure drop from reference depth to first perforation"
        y_unit, yl = _get_pressure_unit(usys)
        d = d/y_unit
    else:
        print("Unknown well field - no unit found")
    return tit, d, yl, do_csum


def _get_pressure_unit(usys: str):
    return {"metric": (BARSA, "barsa"), "field": (PSIA, "psia"),
            "si": (PASCAL, "Pascal"), "lab": (ATM, "atm")}[usys.lower()]


def _get_names_from_ws(ws) -> List[str]:
    """Numeric scalar fields of a wellSol row, last field first."""
    fields = [k for k, v in vars(ws).items()
              if not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, float, np.number))
              or (isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "iuf")]
    return fields[::-1]


def _get_data(wellname: str, wellnames: Sequence[str], field: str, ws) -> Tuple[np.ndarray, int]:
    ind = [k for k, n in enumerate(wellnames) if n.lower() == wellname.lower()]
    assert len(ind) == 1, "Multiple wells with same name!"
    d = np.array([float(np.ravel(getattr(step[ind[0]], field))[0]) for step in ws])
    return d, ind[0]


def _validate_timesteps(wellsols, timesteps):
    if timesteps is None or not isinstance(timesteps, (list, tuple)) \
            or (timesteps and np.isscalar(timesteps[0])):
        timesteps = [None if timesteps is None else np.asarray(timesteps, dtype=float)]
    if len(timesteps) != len(wellsols):
        assert len(timesteps) == 1
        timesteps = timesteps*len(wellsols)
    return timesteps


def _annotate_control_changes(ax, ws, wpos, x, d, line, color):
    types = [step[wpos].type for step in ws]
    _, ctrl_id = np.unique(types, return_inverse=True)
    switched = np.concatenate([[0], np.flatnonzero(np.diff(ctrl_id) != 0)])
    for k, six in enumerate(switched):
        label = types[six] if k == 0 else f"{types[six]}->{types[six + 1]}"
        ax.text(x[six], d[six], label, ha="center",
                bbox=dict(facecolor=[0.7]*3, edgecolor=color, linewidth=2,
                          linestyle=line or "-"))


def _is_packed(wellsols) -> bool:
    items = wellsols if isinstance(wellsols, (list, tuple)) else [wellsols]
    return bool(items) and hasattr(items[0], "SimulatorSetup")


def _is_list_of_datasets(wellsols) -> bool:
    """`wellSols` is one dataset (steps of well rows) or a list of them."""
    first = wellsols[0] if wellsols else None
    return isinstance(first, (list, tuple)) and bool(first) and \
        isinstance(first[0], (list, tuple))


plotWellSols = plot_well_sols

__all__ = ["get_well_unit", "plotWellSols", "plot_well_sols"]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from types import SimpleNamespace

    def _wellsols(n=4):
        return [[SimpleNamespace(name="INJ", sign=1, type="rate", status=True,
                                 bhp=(200 + k)*BARSA, qWs=1e-3, qOs=0.0, qGs=0.0),
                 SimpleNamespace(name="PROD", sign=-1, type="bhp" if k > 1 else "orat",
                                 status=True, bhp=100*BARSA, qWs=-2e-4, qOs=-8e-4, qGs=0.0)]
                for k in range(n)]

    def test_numeric_scalar_fields_are_offered_last_first():
        fields = _get_names_from_ws(_wellsols()[0][0])
        assert fields == ["qGs", "qOs", "qWs", "bhp", "sign"], fields
        print("ok test_numeric_scalar_fields_are_offered_last_first")

    def test_bhp_is_drawn_in_bar_against_days():
        fig, _ = plot_well_sols(_wellsols(), np.array([1, 2, 3, 4])*DAY)
        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_ydata(), [200, 201, 202, 203])
        np.testing.assert_allclose(line.get_xdata(), [1, 2, 3, 4])
        assert fig.axes[0].get_title() == "bhp: Bottom hole pressure"
        print("ok test_bhp_is_drawn_in_bar_against_days")

    def test_cumulative_rate_integrates_over_time():
        fig, _ = plot_well_sols(_wellsols(), np.array([1, 2, 3, 4])*DAY,
                                field="qWs", cumsum=True)
        np.testing.assert_allclose(fig.currentdata[0][:, 0], 1e-3*DAY*np.arange(1, 5))
        print("ok test_cumulative_rate_integrates_over_time")

    def test_field_units_convert_gas_rates_to_scf_per_day():
        _, d, yl, _ = get_well_unit(np.array([1.0]), "qGs", "Field", False, True)
        assert yl == "scf/day" and np.isclose(d[0], DAY/FT**3)
        print("ok test_field_units_convert_gas_rates_to_scf_per_day")

    def test_injecting_a_dataset_redraws():
        fig, inject = plot_well_sols(_wellsols(), np.arange(1, 5)*DAY)
        inject(_wellsols(6), np.arange(1, 7)*DAY)
        assert fig.axes[0].lines[0].get_xdata().size == 6
        print("ok test_injecting_a_dataset_redraws")

    test_numeric_scalar_fields_are_offered_last_first()
    test_bhp_is_drawn_in_bar_against_days()
    test_cumulative_rate_integrates_over_time()
    test_field_units_convert_gas_rates_to_scf_per_day()
    test_injecting_a_dataset_redraws()
    print("\nAll tests passed!")
