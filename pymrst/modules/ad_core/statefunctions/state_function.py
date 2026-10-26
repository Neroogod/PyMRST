"""The state-function graph, ported from MRST ad-core/statefunctions.

This is the machinery MRST's AD-OO framework is built on, and the thing
PyMRST's hand-rolled models did without. Instead of one function that
assembles a residual top to bottom, a model declares a *graph* of small
property evaluators - each naming what it depends on - and the solver
walks the graph, evaluating each property at most once per state and
caching the result.

Three pieces:

- `StateFunction`: one property (pore volume, mobility, phase flux). It
  declares dependencies with `depends_on` and computes itself in
  `evaluate_on_domain`.
- `StateFunctionGrouping`: a named collection of them, cached on the state
  under `struct_name` (MRST: `state.FlowProps`, `state.FluxDisc`, ...).
- `State`: the state itself - primary variables plus one cache container
  per grouping.

A dependency is either internal (another function in the same grouping),
external (a function in another grouping on the model - Mobility in
FlowProps depends on Viscosity in PVTProps), or a primary variable read
straight off the state (`grouping='state'`).

Why it matters: MRST's physics modules (ad-eor, solvent, compositional
flow) are written *as state functions*. Without the graph each must be
re-derived as a bespoke residual; with it they are close ports.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pymrst.utils.adi import value

STATE: str = "state"       # dependency on a primary variable, not a function
SELF: str = "self"         # dependency inside the same grouping


class Dependency:
    """A named property a function needs, and where it lives."""

    __slots__ = ("name", "grouping")

    def __init__(self, name: str, grouping: str = SELF):
        self.name = name
        self.grouping = grouping

    def __repr__(self) -> str:
        return f"Dependency({self.name!r}, {self.grouping!r})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Dependency) and self.name == other.name
                and self.grouping == other.grouping)

    def __hash__(self) -> int:
        return hash((self.name, self.grouping))


class State(dict):
    """Simulation state: primary variables plus one cache per grouping.

    A dict, so a state function reads `state['pressure']`, with the
    grouping caches held separately in `containers`. Changing a primary
    variable invalidates every property computed from it, so the caller
    must `clear_cache()` (or work on a fresh `copy()`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.containers: Dict[str, Dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, val: Any) -> None:
        if name == "containers":
            object.__setattr__(self, name, val)
        else:
            self[name] = val

    def clear_cache(self) -> None:
        """Forget every evaluated property (the state has changed)."""
        self.containers = {}

    def copy(self) -> "State":
        """A copy of the primary variables, with no cached properties."""
        return State({k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in self.items()})


class StateFunction:
    """One property of the domain. Subclasses implement evaluate_on_domain."""

    #: MRST's output contract for a property (Phase 2). A relative
    #: permeability that returns 1.4, or a mobility that returns NaN, is wrong
    #: at the moment it is produced - but nothing looks until the linear solve
    #: fails several iterations later, by which point the offending property is
    #: no longer on the state. Subclasses narrow these.
    output_range = (float("-inf"), float("inf"))
    allow_nan = False
    allow_inf = False

    def __init__(self, model: Any = None, label: Optional[str] = None,
                 regions: Optional[Any] = None):
        self.dependencies: List[Dependency] = []
        self.externals: List[Dependency] = []
        self.label = label or type(self).__name__
        self.struct_name: Optional[str] = None
        self.regions: Optional[Any] = None
        if regions is not None and np.asarray(regions).size:
            self.set_regions(regions)

    @property
    def is_single_region(self) -> bool:
        """MRST `isSingleRegion`: no map, or every cell in the same region.

        A property, not a stored flag, because callers across this port assign
        `.regions` directly and a cached flag would go stale.
        """
        return self.regions is None or np.unique(np.asarray(self.regions)).size == 1

    isSingleRegion = is_single_region

    def set_regions(self, regions: Any) -> "StateFunction":
        """Set the cell-wise region map used by regional property functions."""
        import numpy as np

        values = np.asarray(regions, dtype=np.int64).ravel()
        if values.size == 0:
            raise ValueError("regions cannot be empty")
        if np.any(values < 1):
            raise ValueError("region numbers must be positive")
        self.regions = values
        return self

    setRegions = set_regions

    def depends_on(self, names: Union[str, Sequence[str]],
                   grouping: str = SELF) -> "StateFunction":
        """Declare dependencies. Returns self so calls can be chained."""
        if isinstance(names, str):
            names = [names]
        for n in names:
            dep = Dependency(n, grouping)
            if dep not in self.dependencies:
                self.dependencies.append(dep)
            # MRST keeps cross-grouping dependencies in a second list as well,
            # which is how `get_evaluated_externals` resolves a bare name to
            # its owning grouping. `dependencies` still holds every edge, so
            # the graph walk and `validate_model` are unaffected.
            if grouping not in (SELF, STATE) and dep not in self.externals:
                self.externals.append(dep)
        return self

    def evaluate_on_domain(self, model: Any, state: State) -> Any:
        raise NotImplementedError("base StateFunction cannot be evaluated")

    def get_evaluated(self, state: State, name: str) -> Any:
        """Read an already-evaluated dependency from this grouping's cache."""
        return state.containers[self.struct_name][name]

    def get_evaluated_external(self, model: Any, state: State,
                               grouping: str, name: str) -> Any:
        """Read an already-evaluated dependency from another grouping."""
        struct = getattr(model, grouping).struct_name
        return state.containers[struct][name]

    # -- MRST StateFunction.m, ported (Phase 2) -----------------------------

    def reset_dependencies(self, internal: bool = True,
                           external: bool = True) -> "StateFunction":
        """Clear declared dependencies, so a subclass can redeclare them.

        A model that swaps one property for another has to drop the old
        property's dependencies too, or the graph keeps an edge to something
        nothing computes any more.
        """
        if internal:
            self.dependencies = []
        if external:
            self.externals = []
        return self

    resetDependencies = reset_dependencies

    def subset(self, cell_subset) -> "StateFunction":
        """Restrict this function's region map to a subset of cells.

        Only the regions need restricting: everything else is evaluated from
        the state it is handed, so it follows the subset automatically. A copy
        is returned, as MATLAB's value semantics give: subsetting one cell
        region must not restrict the function the next region subsets.
        """
        from copy import copy
        out = copy(self)
        if out.regions is not None:
            import numpy as np
            out.regions = np.asarray(out.regions)[cell_subset]
        return out

    def get_evaluated_dependencies(self, state: State, *names: str):
        """Several already-evaluated dependencies from this grouping."""
        out = [self.get_evaluated(state, n) for n in names]
        return out[0] if len(out) == 1 else out

    getEvaluatedDependencies = get_evaluated_dependencies

    def get_evaluated_externals(self, model: Any, state: State, *names: str):
        """Dependencies from other groupings, by `"Grouping.name"` or by name.

        A bare name is resolved through the declared externals, which is why
        `depends_on(..., grouping=...)` has to record them.
        """
        out = []
        for entry in names:
            if "." in entry:
                grouping, name = entry.split(".", 1)
            else:
                name = entry
                match = [d for d in self.externals if d.name == entry]
                if not match:
                    raise KeyError(
                        f"{self.label} has no declared external dependency "
                        f"{entry!r}; pass 'Grouping.{entry}' or declare it")
                grouping = match[0].grouping
            if grouping == "state":
                out.append(model.get_prop(state, name))
            else:
                out.append(self.get_evaluated_external(model, state, grouping,
                                                       name))
        return out[0] if len(out) == 1 else out

    getEvaluatedExternals = get_evaluated_externals

    def evaluate_function_cell_subset(self, fn, subset, *args):
        """Call `fn` per region when it is a list of per-region functions.

        A single callable is applied whole. A list means the property is
        tabulated per saturation or PVT region, and each cell must be
        evaluated with *its own* table - applying region one everywhere is a
        silent, plausible answer.
        """
        if not isinstance(fn, (list, tuple)):
            return fn(*args)
        if self.regions is None:
            raise ValueError(
                f"{self.label} was given a per-region function list but has no "
                f".regions; a region map is required to choose between them")
        if self.is_single_region:
            return fn[int(np.asarray(self.regions).ravel()[0]) - 1](*args)

        from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
        from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad

        # MRST decides which arguments are cell-wise against the *full* region
        # map, then subsets the map; the arguments arrive already subset.
        n_cells = np.asarray(self.regions).size
        regions = np.asarray(self.regions)
        if subset is not None:
            regions = regions[subset]
        cellwise = [_is_cellwise(a, n_cells) or _is_cellwise(a, regions.size)
                    for a in args]

        # The output must be AD whenever an argument is, or every derivative
        # through a regional table is lost - Newton then sees a constant
        # property and converges slowly or not at all.
        sample, is_ad = get_sample_ad(*args) if args else (None, False)
        out = np.zeros(regions.size)
        if is_ad:
            out = double_to_adi(out, sample)
        for index in range(len(fn)):
            active = regions == index + 1          # MRST regions are one-based
            if not np.any(active):
                continue
            local = [a[active] if is_cell else a
                     for a, is_cell in zip(args, cellwise)]
            out[active] = fn[index](*local)
        return out

    evaluateFunctionCellSubset = evaluate_function_cell_subset

    def evaluate_function_on_domain_with_arguments(self, fn, *args):
        """Evaluate over every cell, dispatching per region if `fn` is a list."""
        return self.evaluate_function_cell_subset(fn, None, *args)

    evaluateFunctionOnDomainWithArguments = evaluate_function_on_domain_with_arguments

    def evaluate_function_single_region_with_arguments(self, fn, region_index,
                                                       *args):
        """Evaluate one region's function, by one-based region index."""
        if isinstance(fn, (list, tuple)):
            if not 1 <= region_index <= len(fn):
                raise IndexError(
                    f"region {region_index} is outside the {len(fn)} regions "
                    f"of {self.label}")
            return fn[region_index - 1](*args)
        return fn(*args)

    evaluateFunctionSingleRegionWithArguments = \
        evaluate_function_single_region_with_arguments

    def evaluate_fluid(self, model: Any, name: str, *args):
        """Evaluate a named function off the model's fluid, region-aware.

        The usual way a property reaches a PVT or relperm curve: the fluid
        holds the table, this applies it with the right region per cell.
        """
        fluid = getattr(model, "raw_fluid", None) or model.fluid
        fn = getattr(fluid, name, None)
        if fn is None:
            raise AttributeError(
                f"fluid has no {name!r} for state function {self.label}")
        return self.evaluate_function_on_domain_with_arguments(fn, *args)

    evaluateFluid = evaluate_fluid

    def validate_output(self, output: Any, level: int = 1) -> None:
        """Warn when a computed property is outside its declared contract.

        Levels follow MRST: 0 off, 1+ check values, 3+ also check the Jacobian
        entries and list the first offenders. It warns rather than raises,
        deliberately - a property can be transiently out of range mid-Newton
        and still converge, so failing here would reject runs that are fine.
        The point is that the message names the property, which is what turns
        "the solve diverged" into "BlackOilCapillaryPressure produced NaN".
        """
        if level <= 0 or isinstance(output, dict):
            return
        items = output if isinstance(output, (list, tuple)) else [output]
        check_jacobian = level > 2
        low, high = self.output_range
        for column, item in enumerate(items, 1):
            if not self.allow_nan:
                self._validate_output_internal(
                    item, column, level, "NaN values", np.isnan, check_jacobian)
            if not self.allow_inf:
                self._validate_output_internal(
                    item, column, level, "Inf values", np.isinf, check_jacobian)
            if np.isfinite(low):
                self._validate_output_internal(
                    item, column, level, f"value less than {low:g}",
                    lambda x, bound=low: x < bound, False)
            if np.isfinite(high):
                self._validate_output_internal(
                    item, column, level, f"value larger than {high:g}",
                    lambda x, bound=high: x > bound, False)

    validateOutput = validate_output

    def _validate_output_internal(self, item, column, level, description,
                                  test, check_jacobian) -> None:
        import warnings

        values = np.asarray(value(item))
        if values.dtype.kind not in "fiub":
            return
        bad = np.asarray(test(values))
        count = int(bad.sum())
        if count:
            total = values.size
            warnings.warn(
                f"bad output: {description} in {count} of {total} "
                f"({100.0*count/total:.2f}%) entries for state function "
                f"'{type(self).__name__}' column {column}")
            if level > 2:
                offenders = np.flatnonzero(bad.ravel())[:level]
                for index in offenders:
                    print(f"{index}: {values.ravel()[index]}")
        if check_jacobian and hasattr(item, "jac"):
            for number, block in enumerate(item.jac, 1):
                data = getattr(block, "data", None)
                if data is None:
                    continue
                bad_jac = int(np.asarray(test(np.asarray(data))).sum())
                if bad_jac:
                    warnings.warn(
                        f"bad Jacobian: {description} in {bad_jac} entries for "
                        f"state function '{type(self).__name__}' column "
                        f"{column} Jacobian {number}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r})"


StateFunction.validateOutputInternal = StateFunction._validate_output_internal


def _is_cellwise(arg: Any, n_cells: int) -> bool:
    """Whether an argument varies per cell, and so must be subset per region."""
    array = np.asarray(value(arg))
    return array.ndim >= 1 and array.shape[0] == n_cells


def __getattr__(name: str) -> Any:
    """`StateFunctionGrouping` is MRST's own file, `state_function_grouping.py`,
    which builds on the classes above; it is loaded on first use so the two
    modules do not import each other at import time."""
    if name == "StateFunctionGrouping":
        from pymrst.modules.ad_core.statefunctions.state_function_grouping import (
            StateFunctionGrouping,
        )
        return StateFunctionGrouping
    raise AttributeError(name)


if __name__ == "__main__":
    def test_a_function_records_its_dependencies():
        fn = StateFunction(None, label="x")
        fn.depends_on(["pressure"], STATE)
        fn.depends_on("Density")
        names = [d.name for d in fn.dependencies]
        assert "pressure" in names and "Density" in names
        print("ok test_a_function_records_its_dependencies")

    test_a_function_records_its_dependencies()
    print("\nAll tests passed!")
