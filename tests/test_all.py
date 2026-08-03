"""Run every PyMRST module's self-tests, and every MATLAB validation.

    pytest                    # module self-tests (fast, no MATLAB needed)
    pytest -m validation      # cross-checks against the saved MATLAB data
    pytest -q                 # everything

Each module under `pymrst/` is executable and runs its own unit tests in a
`__main__` block; this collects them so a single command checks the whole
project. The validation tests replay the saved MATLAB reference data in
`examples/validation/` - they do not need MATLAB itself, only the .mat
files produced earlier by the dump_phase*.m scripts.
"""

import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "examples" / "validation"


def _all_scripts():
    skip = {"__init__.py"}
    return sorted(
        p for p in (ROOT / "pymrst").rglob("*.py")
        if p.name not in skip and "__pycache__" not in p.parts
    )


def _is_example(path) -> bool:
    """Example drivers run a whole simulation; module tests do not."""
    return "examples" in path.parts


def _module_scripts():
    return [p for p in _all_scripts() if not _is_example(p)]


def _example_scripts():
    return [p for p in _all_scripts() if _is_example(p)]


def _validators():
    return sorted(VALID.glob("validate_phase*.py"))


# Import names for the extras in pyproject's [project.optional-dependencies].
# Written out rather than parsed from pyproject because an import name is not a
# distribution name in general, and guessing one from the other is how a typo
# turns into a silent skip. Only a module named here may be skipped for being
# absent; anything else missing is a real failure, including a pymrst module.
_OPTIONAL_IMPORTS = frozenset({
    "pyamg", "pyvista", "pyvistaqt", "PyQt5", "plotly", "torch", "h5py",
    "pymetis", "streamlit", "pandas",
})

_MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def _missing_optional_import(stderr: str):
    """The optional dependency a script died for, or None.

    Matches the *last* such line: a module can fail to import because something
    it imports is missing, and the innermost cause is the one that names the
    dependency actually absent.
    """
    names = _MISSING_MODULE.findall(stderr or "")
    if not names:
        return None
    root = names[-1].split(".")[0]
    return root if root in _OPTIONAL_IMPORTS else None


@pytest.mark.parametrize("script", _module_scripts(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_module_selftests(script):
    """Every module's own unit tests must pass.

    A few modules are command-line tools rather than test suites - their
    `__main__` builds an ArgumentParser and requires an argument. Running one
    bare makes argparse print usage and exit 2, which is the tool working
    correctly, not a failing test. That exact signal is skipped rather than
    skipping every module that mentions argparse: two of the three that do have
    defaults and run their tests fine, so a blanket exclusion would lose real
    coverage.
    """
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    if r.returncode == 2 and "the following arguments are required" in r.stderr:
        pytest.skip(f"{script.name} is a command-line tool, not a self-test")
    missing = _missing_optional_import(r.stderr)
    if missing is not None:
        pytest.skip(f"{script.name} needs the optional dependency {missing!r}")
    assert r.returncode == 0, (
        f"{script.relative_to(ROOT)} failed:\n{r.stdout[-3000:]}\n"
        f"{r.stderr[-3000:]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("script", _example_scripts(),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_example_drivers(script):
    """Every example driver must still run end to end.

    These are separated from the module self-tests for one reason: they run
    real simulations and some take far longer than a unit test should. SPE9
    completes all 35 of its steps but needs well over the 900s the module sweep
    allows, so it was failing on the clock rather than on anything real.

    They are **not** deselected by default - a failure here is a failure, and
    hiding it behind an opt-in marker is how the last batch of them went
    unnoticed. The `slow` marker exists so a developer can say
    `-m "not slow"` deliberately, not so CI can skip them by accident.
    """
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode == 2 and "the following arguments are required" in r.stderr:
        pytest.skip(f"{script.name} is a command-line tool, not a self-test")
    assert r.returncode == 0, (
        f"{script.relative_to(ROOT)} failed:\n{r.stdout[-3000:]}\n"
        f"{r.stderr[-3000:]}"
    )


@pytest.mark.validation
@pytest.mark.parametrize("script", _validators(),
                         ids=lambda p: p.stem.replace("validate_", ""))
def test_matlab_validation(script):
    """Every phase must still match its saved MATLAB reference."""
    ref_missing = not any(VALID.glob("ref_*.mat"))
    if ref_missing:
        pytest.skip("no MATLAB reference data present")
    r = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    if "missing (run" in r.stdout:
        pytest.skip(f"reference data for {script.stem} not generated")
    assert r.returncode == 0, (
        f"{script.name} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    )


# --------------------------------------------------------------------------
# Guard: a self-test must never open a blocking plot window.
#
# `test_module_selftests` runs each file as a captured subprocess. A self-test
# that calls a PyVista-backed plotting routine with `show` left at its default
# of True opens an interactive window and waits for someone to close it, so the
# subprocess never exits and the sweep sits on it until the 900s timeout.
#
# This cost a long debugging session precisely because it does not reproduce
# reliably: `python the_module.py` at a terminal printed "All tests passed!"
# every time, while the same file run under this harness blocked. A static
# check is therefore worth more here than any amount of running it.

# A hardcoded list missed `plot_frac_system_nf` and let `stochastic_fractures`
# block the sweep for half an hour, so the set is derived instead: every public
# name in `pymrst.plotting`, plus anything named `plot_*` or `*_view` defined
# anywhere under `pymrst/`. Over-inclusion is cheap here - a false positive
# costs one `show=False` - while a miss costs a stalled sweep with no
# diagnostic.
_OFF_BY_DEFAULT: set[str] = set()


def _blocking_plotters():
    """Functions that can actually open a window.

    Name matching alone is wrong in both directions. A hardcoded list missed
    `plot_frac_system_nf` and let `stochastic_fractures` stall the sweep for
    half an hour; expanding it by name then caught `boundary_condition_iteration_plot`
    and `show_natural_overall2_runtime_plot`, which only build payload dicts and
    draw nothing.

    So membership is decided by what a function *reaches*: a real drawing
    primitive (`pyplot`, a `pyvista` plotter, a bare `.show()`), or another
    function already known to reach one. Seeded from `pymrst.plotting.__all__`
    and closed transitively.
    """
    import ast

    seeds = set()
    try:
        import pymrst.plotting as plotting
        seeds |= {n for n in getattr(plotting, "__all__", ())
                  if n.startswith(("plot_", "outline_", "explosion_"))}
    except Exception:                                # pragma: no cover
        pass

    # function name -> names it calls, per file
    calls: dict[str, set[str]] = {}
    draws: set[str] = set(seeds)
    for path in _all_scripts():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):    # pragma: no cover
            continue

        # Which local names actually refer to a plotting library *in this
        # file*. Assuming `pv` means pyvista is wrong here: in reservoir
        # simulation `pv` is pore volume, and `pv.sum()` appears throughout
        # ad_core. Only an import decides it.
        aliases = set()
        for inner in ast.walk(tree):
            if isinstance(inner, ast.Import):
                for a in inner.names:
                    if a.name in {"pyvista", "matplotlib.pyplot"}:
                        aliases.add(a.asname or a.name.split(".")[0])
            elif isinstance(inner, ast.ImportFrom) and inner.module:
                if inner.module.startswith(("matplotlib", "pyvista")):
                    aliases |= {a.asname or a.name for a in inner.names}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            primitive = False
            called: set[str] = set()
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                target = inner.func
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name:
                    called.add(name)
                # Only what can actually display: a `.show()`, or building a
                # `Plotter`. Any-pyvista-attribute was too wide - `to_pyvista`
                # and `streamlines_to_pyvista` construct meshes and draw
                # nothing.
                if name in {"show", "Plotter"}:
                    primitive = True
                root = target
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in aliases and name in {
                    "plot", "imshow", "bar", "barh", "scatter", "contour",
                    "contourf", "hist", "pcolormesh", "tripcolor", "triplot",
                }:
                    primitive = True
            calls.setdefault(node.name, set()).update(called)
            if primitive:
                draws.add(node.name)

            # A drawer whose drawing sits behind parameters that all default to
            # False is safe when called plainly: `run_adjoint_example` and
            # `grid_helpers_example` both draw only when asked. Without this the
            # closure flags every such function and the guard becomes noise.
            switches = [a for a in node.args.args + node.args.kwonlyargs
                        if any(w in a.arg for w in ("plot", "show"))]
            if switches:
                defaults = list(node.args.defaults) + list(node.args.kw_defaults)
                off = [d for d in defaults
                       if isinstance(d, ast.Constant) and d.value is False]
                if len(off) >= len(switches):
                    _OFF_BY_DEFAULT.add(node.name)

    # Close transitively. The real chain is three deep -
    # `setup_stochastic_fractures` -> `plot_frac_system_nf` ->
    # `plot_frac_system` -> `plot_grid` - and no intermediate touches a
    # primitive, so one level misses the case this guard exists for.
    #
    # An earlier closure over-reached and dragged in half of ad_core's
    # examples, but that was the seed set: `pv.sum()` was read as pyvista
    # before aliases were bound to real imports. On clean seeds the closure is
    # what makes the deep chain visible.
    changed = True
    while changed:
        changed = False
        for name, called in calls.items():
            if name not in draws and called & draws:
                draws.add(name)
                changed = True
    return frozenset(draws)


_BLOCKING_PLOTTERS = _blocking_plotters()


def _selftest_plot_calls_without_show():
    """(file, line, description) for every risky plotting call in a self-test.

    Three shapes have to be told apart, and getting this wrong is easy:

    * a direct call with neither `show=` nor a plotter — always unsafe;
    * a helper that forwards `**kwargs` to the plotting routine — safe only if
      the self-test's call to *that helper* passes `show=`;
    * anything supplying `show=` or a plotter — safe.

    The second shape is what blocked the sweep through `tutorial_plotting.py`,
    and a first version of this check waved it through as "the caller's
    responsibility" without ever inspecting the caller.
    """
    import ast

    def _bound(call, name):
        names = {kw.arg for kw in call.keywords}
        params = _plot_parameters(name)
        names.update(params[: len(call.args)])
        return names

    risky = []
    for path in _all_scripts():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):       # pragma: no cover
            continue

        # Classify each module-level helper by what it does with a plotter.
        unsafe_helpers, forwarding_helpers = {}, {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                if name not in _BLOCKING_PLOTTERS:
                    continue
                bound = _bound(call, name)
                if "show" in bound or "plotter" in bound:
                    continue
                if any(kw.arg is None for kw in call.keywords):
                    forwarding_helpers.setdefault(node.name, (call.lineno, name))
                else:
                    unsafe_helpers.setdefault(node.name, (call.lineno, name))

        for node in tree.body:
            if not isinstance(node, ast.If) or "__name__" not in ast.dump(node.test):
                continue
            # Skip command-line entry points, as `test_module_selftests` does:
            # argparse exits 2 on a bare run, so their drawing calls sit behind
            # a parsed flag and never execute in the sweep.
            if "argparse" in source or any(
                isinstance(c, ast.Call) and getattr(c.func, "id", None) == "main"
                for c in ast.walk(node)
            ):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
                rel = str(path.relative_to(ROOT))

                if name in _BLOCKING_PLOTTERS:
                    if any(kw.arg is None for kw in call.keywords):
                        continue                      # **kwargs from an outer scope
                    bound = _bound(call, name)
                    # `show=False` and `plot=False` are both "do not draw" in
                    # this codebase. Either settles it, but only as a literal
                    # False: `show=plotter is None` binds the keyword and
                    # evaluates to True, which is how `stochastic_fractures`
                    # blocked the sweep while looking guarded.
                    given = {kw.arg: kw.value for kw in call.keywords}
                    suppressed = any(
                        isinstance(given.get(k), ast.Constant)
                        and given[k].value is False
                        for k in ("show", "plot")
                    )
                    if suppressed or "plotter" in bound:
                        continue
                    if name in _OFF_BY_DEFAULT and not ({"show", "plot"} & bound):
                        continue        # draws only when explicitly asked
                    if {"show", "plot"} & bound:
                        risky.append((rel, call.lineno,
                                      f"{name}(show/plot=<not literally False>)"))
                    else:
                        risky.append((rel, call.lineno, f"{name}(...)"))

                elif name in unsafe_helpers:
                    line, plotted = unsafe_helpers[name]
                    risky.append((rel, call.lineno,
                                  f"{name}(...) -> {plotted}(...) at line {line}"))

                elif name in forwarding_helpers:
                    if not ({"show", "plot"} & {kw.arg for kw in call.keywords}):
                        line, plotted = forwarding_helpers[name]
                        risky.append((rel, call.lineno,
                                      f"{name}(...) forwards **kwargs to "
                                      f"{plotted}(...) at line {line}; pass show="))

            # Indirection defeats name matching: `for fn in (a, b): fn(case)`
            # never mentions `a` at a call site. So if a risky helper is merely
            # *referenced* in the block and nothing in the block passes `show=`
            # anywhere, treat it as unguarded. This is the shape that actually
            # blocked the sweep.
            passes_show = any(
                isinstance(c, ast.Call)
                and ({"show", "plot"} & {kw.arg for kw in c.keywords})
                for c in ast.walk(node)
            )
            if not passes_show:
                called_by_name = {
                    getattr(c.func, "id", None)
                    for c in ast.walk(node) if isinstance(c, ast.Call)
                }
                for ref in ast.walk(node):
                    if not isinstance(ref, ast.Name) or isinstance(ref.ctx, ast.Store):
                        continue
                    if ref.id in called_by_name:
                        continue            # already handled as a direct call
                    if ref.id in unsafe_helpers or ref.id in forwarding_helpers:
                        line, plotted = (unsafe_helpers.get(ref.id)
                                         or forwarding_helpers[ref.id])
                        risky.append((str(path.relative_to(ROOT)), ref.lineno,
                                      f"{ref.id} (referenced, reaches {plotted}"
                                      f"(...) at line {line}) and no show= in "
                                      f"this block"))
    return risky


def _plot_parameters(name):
    """Positional parameter names of a plotting routine, or () if unknown.

    Resolved from the real signature because these routines accept the plotter
    positionally as well as by keyword, and a keyword-only check misses
    `plot_well(g, w, pl)`.
    """
    import inspect

    try:
        import pymrst.plotting as plotting
        fn = getattr(plotting, name, None)
        if fn is None:
            return ()
        return tuple(
            p.name for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
    except Exception:                                # pragma: no cover
        return ()


def test_no_selftest_opens_a_blocking_plot_window():
    risky = _selftest_plot_calls_without_show()
    assert not risky, (
        "these self-tests call a plotting routine without `show=False` or an "
        "off-screen plotter, so they can block the whole sweep until it times "
        "out:\n" + "\n".join(f"  {f}:{line}  {fn}(...)" for f, line, fn in risky)
    )
