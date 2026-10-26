"""MRST ``ad-core/statefunctions/StateFunctionGrouping.m``: a named collection
of state functions, cached on the state under its container name (MRST's
`state.FlowProps`, `state.PVTProps`, ...). Functions are evaluated with their
dependencies - internal ones in this grouping, external ones in another
grouping or the state - at most once per state, in dependency order.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    SELF,
    STATE,
    Dependency,
    State,
    StateFunction,
)
from pymrst.utils.adi import value


class StateFunctionGrouping:
    """A named set of state functions, cached on the state under struct_name."""

    def __init__(self, struct_name: str):
        self.struct_name = struct_name
        self.functions: Dict[str, StateFunction] = {}
        # 0 is off, and is the default: validation warns on every evaluation,
        # which is right when you are debugging a property and wrong for a
        # production run. MRST makes it opt-in for the same reason.
        self.validation_level = 0

    def set_state_function(self, name: str,
                           fn: StateFunction) -> "StateFunctionGrouping":
        """Register (or replace) a state function. Returns self."""
        assert isinstance(fn, StateFunction), "not a StateFunction"
        fn.struct_name = self.struct_name
        self.functions[name] = fn
        return self

    def remove_state_function(self, name: str) -> "StateFunctionGrouping":
        del self.functions[name]
        return self

    def has_state_function(self, name: str) -> bool:
        return name in self.functions

    def get_state_function(self, name: str) -> StateFunction:
        if name not in self.functions:
            raise KeyError(
                f"'{name}' is not known to the {type(self).__name__} grouping; "
                f"it has {sorted(self.functions)}"
            )
        return self.functions[name]

    def get_names(self) -> List[str]:
        return list(self.functions)

    # -- evaluation ---------------------------------------------------------
    def _container(self, state: State) -> Dict[str, Any]:
        return state.containers.setdefault(self.struct_name, {})

    def is_evaluated(self, state: State, name: str) -> bool:
        # A slot holding `None` is "not evaluated yet": that is how
        # `init_state_function_container` lays out a fresh container, and it
        # is MRST's rule too (an empty field). Testing key membership alone
        # treated every function of a freshly initialised container as
        # already evaluated, and `get` then returned `None`.
        return state.containers.get(self.struct_name, {}).get(name) is not None

    def get(self, model: Any, state: State, name: str) -> Any:
        """The value of property `name`, evaluating it and its dependencies
        only if it is not already cached on this state."""
        if not self.is_evaluated(state, name):
            self.evaluate_with_dependencies(model, state, name)
        return self._container(state)[name]

    def evaluate_with_dependencies(self, model: Any, state: State, name: str,
                                   _stack: Tuple[str, ...] = ()) -> None:
        """Depth-first evaluation of `name` and everything beneath it."""
        key = f"{self.struct_name}.{name}"
        if key in _stack:
            raise RuntimeError("cyclic state-function dependency: "
                               + " -> ".join(_stack + (key,)))

        fn = self.get_state_function(name)
        for dep in fn.dependencies:
            if dep.grouping == STATE:
                continue                      # a primary variable; already there
            group = self if dep.grouping == SELF else getattr(
                model, dep.grouping, None)
            if group is None:
                # MRST's `isStateFunctionEvaluated` wraps `model.(grouping)` in
                # try/catch and treats a model without that grouping as
                # "already evaluated": a facility property names reservoir
                # groupings it then reaches through `ReservoirModel` itself.
                continue
            if not group.is_evaluated(state, dep.name):
                group.evaluate_with_dependencies(model, state, dep.name,
                                                 _stack + (key,))

        self._container(state)[name] = fn.evaluate_on_domain(model, state)

    # -- introspection ------------------------------------------------------
    def dependency_graph(self) -> Dict[str, List[Dependency]]:
        """Property name -> its declared dependencies. What MRST plots."""
        return {n: list(f.dependencies) for n, f in self.functions.items()}

    def evaluation_order(self) -> List[str]:
        """This grouping's names in dependency order (a topological sort).

        Externals are assumed available and are not ordered here. Raises
        on a cycle.
        """
        order: List[str] = []
        marks: Dict[str, int] = {}

        def visit(n: str, stack: Tuple[str, ...]) -> None:
            mark = marks.get(n, 0)
            if mark == 2:
                return
            if mark == 1:
                raise RuntimeError("cyclic state-function dependency: "
                                   + " -> ".join(stack + (n,)))
            marks[n] = 1
            for dep in self.functions[n].dependencies:
                if dep.grouping == SELF and dep.name in self.functions:
                    visit(dep.name, stack + (n,))
            marks[n] = 2
            order.append(n)

        for n in self.functions:
            visit(n, ())
        return order

    # -- MRST StateFunctionGrouping.m, ported (Phase 2) ---------------------

    def get_state_function_container_name(self) -> str:
        """The state field this grouping caches under."""
        return self.struct_name

    getStateFunctionContainerName = get_state_function_container_name

    def get_state_function_container(self, state: Optional[State] = None):
        """This grouping's cache - the existing one, or a fresh one.

        A fresh container is keyed by every function name with a `None` value,
        so a missing key means "not a member of this grouping" while a `None`
        means "not evaluated yet". Collapsing those two makes a typo in a
        dependency name look like an ordering problem.
        """
        if state is not None and self.struct_name in state.containers:
            return state.containers[self.struct_name]
        return {name: None for name in self.functions}

    getStateFunctionContainer = get_state_function_container

    def init_state_function_container(self, state: State) -> State:
        state.containers[self.struct_name] = self.get_state_function_container()
        return state

    initStateFunctionContainer = init_state_function_container

    def is_state_initialized(self, state: State) -> bool:
        return self.struct_name in getattr(state, "containers", {})

    isStateInitialized = is_state_initialized

    def is_state_function_evaluated(self, model: Any, state: State,
                                    dep: Any) -> bool:
        """Whether a dependency is already computed, here or elsewhere.

        A state carrying `evaluated` has had the whole graph pre-evaluated, so
        everything is ready by construction.
        """
        if state.get("evaluated"):
            return True
        if isinstance(dep, str):
            container = getattr(state, "containers", {}).get(self.struct_name)
            if container is None:
                return False
            if self.validation_level and dep not in self.functions:
                raise KeyError(
                    f"did not find {dep!r} in {self.struct_name}; it does not "
                    f"appear to belong to {type(self).__name__} - check your "
                    f"dependencies")
            return container.get(dep) is not None
        if dep.grouping == STATE:
            return True
        owner = getattr(model, dep.grouping, None)
        if owner is None:
            return True
        return owner.is_state_function_evaluated(model, state, dep.name)

    isStateFunctionEvaluated = is_state_function_evaluated

    def evaluate_state_function(self, model: Any, state: State,
                                name: str) -> Any:
        """Evaluate one function and cache it, validating the result.

        The checked path; `evaluate_state_function_unsafe` is the same without
        validation, for a walk that has already established the order.
        """
        if not self.is_state_initialized(state):
            self.init_state_function_container(state)
        prop = self.get_state_function(name)
        result = prop.evaluate_on_domain(model, state)
        if self.validation_level > 0:
            prop.validate_output(result, self.validation_level)
        state.containers[self.struct_name][name] = result
        return result

    evaluateStateFunction = evaluate_state_function

    def evaluate_state_function_unsafe(self, model: Any, state: State,
                                       name: str) -> Any:
        """Evaluate without validation or container setup.

        "Unsafe" is MRST's word and it is accurate: it assumes the container
        exists and the dependencies are evaluated, which holds only when the
        caller is walking the graph in dependency order.
        """
        result = self.get_state_function(name).evaluate_on_domain(model, state)
        state.containers[self.struct_name][name] = result
        return result

    evaluateStateFunctionUnsafe = evaluate_state_function_unsafe


    def evaluate_dependencies(self, model: Any, state: State,
                              name: str) -> State:
        """Evaluate everything `name` needs, without evaluating `name`."""
        for dep in self.get_state_function(name).dependencies:
            if dep.grouping in (SELF,):
                self.get(model, state, dep.name)
            elif dep.grouping != STATE:
                owner = getattr(model, dep.grouping, None)
                if owner is not None:
                    owner.get(model, state, dep.name)
        return state

    evaluateDependencies = evaluate_dependencies

    def evaluate_external_dependencies(self, model: Any, state: State,
                                       name: str) -> State:
        """Evaluate only the cross-grouping dependencies of `name`."""
        for dep in self.get_state_function(name).externals:
            owner = getattr(model, dep.grouping, None)
            if owner is not None:
                owner.get(model, state, dep.name)
        return state

    evaluateExternalDependencies = evaluate_external_dependencies

    def check_dependencies(self, groups: Optional[Sequence[Any]] = None, *,
                           nargout: int = 0, verbose: bool = False) -> bool:
        """MRST `checkDependencies`: internal dependencies always, external ones
        against `groups` when given (a grouping matches by class or struct
        name; the state always matches).

        As MRST without an output (`nargout=0`), the first unmet dependency
        raises `KeyError`; with one (`nargout=1`) every unmet dependency is
        reported when `verbose` (MRST's `mrstVerbose`) and `ok` is returned.
        """
        check_external = groups is not None
        cls = type(self).__name__

        def unmet(message: str) -> None:
            if nargout == 0:
                raise KeyError(message)
            if verbose:
                print(message)

        ok = True
        for name, fn in self.functions.items():
            for dep in fn.dependencies:
                if dep.grouping != SELF:
                    continue
                present = self.has_state_function(dep.name)
                if not present:
                    unmet(f"Unmet internal dependency for {name} in group {cls}: "
                          f"Did not find {dep.name} in own group.")
                ok = ok and present
            if not check_external:
                continue
            for dep in fn.dependencies:
                if dep.grouping in (SELF, STATE):
                    continue
                candidates = [g for g in groups
                              if dep.grouping in (type(g).__name__, g.struct_name)
                              or any(c.__name__ == dep.grouping for c in type(g).__mro__)]
                if not candidates:
                    ok = False
                    unmet(f"Unmet external dependency for {name} in group {cls}: Did not "
                          f"find any groups of class {dep.grouping} that may contain "
                          f"{dep.name}.")
                    continue
                found = any(g.has_state_function(dep.name) for g in candidates)
                ok = ok and found
                if not found:
                    unmet(f"Unmet external dependency for function {name} in group {cls}: "
                          f"Did not find {dep.name} in group of class {dep.grouping}.")
        if ok and verbose:
            kind = "internal and external" if check_external else "internal"
            print(f"All {kind} dependencies are met for group of class {cls}.")
        return ok

    checkDependencies = check_dependencies

    def set_validation_level(self, level: int) -> "StateFunctionGrouping":
        self.validation_level = int(level)
        return self

    setValidationLevel = set_validation_level

    def set_compact_evaluation(self, enabled: bool) -> "StateFunctionGrouping":
        """Set `use_compact_evaluation` on every function that has it."""
        for prop in self.functions.values():
            if hasattr(prop, "use_compact_evaluation"):
                prop.use_compact_evaluation = bool(enabled)
        return self

    setCompactEvaluation = set_compact_evaluation

    def set_internal_names(self, names: Sequence[str]) -> "StateFunctionGrouping":
        """Restrict the grouping to a named subset, dropping the rest."""
        keep = {str(n) for n in names}
        unknown = keep - set(self.functions)
        if unknown:
            raise KeyError(
                f"{self.struct_name} does not define {sorted(unknown)}")
        self.functions = {k: v for k, v in self.functions.items() if k in keep}
        return self

    setInternalNames = set_internal_names

    def subset(self, cell_subset) -> "StateFunctionGrouping":
        """Restrict every member function to a subset of cells (on a copy)."""
        from copy import copy
        out = copy(self)
        out.functions = {name: prop.subset(cell_subset)
                         for name, prop in self.functions.items()}
        return out

    def disp(self) -> None:
        """MRST `disp`: the grouping's class and its state functions.

        MRST lists the class's own (intrinsic) functions apart from those added
        to the instance; PyMRST's groupings register every function on the
        instance, so all are listed as added.
        """
        print(f"  {type(self).__name__} state function grouping instance.")
        names = self.get_names()
        width = max((len(n) for n in names), default=0) + 1
        print("  Class has no intrinsic state functions.")
        if not names:
            print("  No extra state functions have been added to this class instance.")
            return
        print("\n  Extra state functions (Added to class instance):")
        for name in names:
            print(f"    {name:>{width}}: {type(self.get_state_function(name)).__name__}")

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.struct_name!r}, "
                f"{len(self.functions)} functions)")


# MRST's spelling for the method this port already had under another name.
_G = StateFunctionGrouping
_G.evaluateStateFunctionWithDependencies = _G.evaluate_with_dependencies


if __name__ == "__main__":
    class _Const(StateFunction):
        """A leaf: reads a primary variable off the state."""

        def __init__(self, model=None, key="pressure"):
            super().__init__(model, label="Const")
            self.key = key
            self.depends_on(key, STATE)

        def evaluate_on_domain(self, model, state):
            model.n_eval["A"] = model.n_eval.get("A", 0) + 1
            return np.asarray(state[self.key], dtype=float)

    class _Double(StateFunction):
        def __init__(self, model=None, src="A", tag="B"):
            super().__init__(model, label=tag)
            self.src, self.tag = src, tag
            self.depends_on(src)

        def evaluate_on_domain(self, model, state):
            model.n_eval[self.tag] = model.n_eval.get(self.tag, 0) + 1
            return 2.0 * self.get_evaluated(state, self.src)

    class _Sum(StateFunction):
        def __init__(self, model=None, srcs=("B", "C")):
            super().__init__(model, label="D")
            self.srcs = srcs
            self.depends_on(list(srcs))

        def evaluate_on_domain(self, model, state):
            model.n_eval["D"] = model.n_eval.get("D", 0) + 1
            return sum(self.get_evaluated(state, s) for s in self.srcs)

    class _External(StateFunction):
        """Depends on a function living in another grouping."""

        def __init__(self, model=None):
            super().__init__(model, label="E")
            self.depends_on("D", "props")

        def evaluate_on_domain(self, model, state):
            return 10.0 * self.get_evaluated_external(model, state, "props", "D")

    class _Model:
        def __init__(self):
            self.n_eval = {}
            self.props = StateFunctionGrouping("Props")
            self.props.set_state_function("A", _Const(self, "pressure"))
            self.props.set_state_function("B", _Double(self, "A", "B"))
            self.props.set_state_function("C", _Double(self, "A", "C"))
            self.props.set_state_function("D", _Sum(self, ("B", "C")))
            self.flux = StateFunctionGrouping("Flux")
            self.flux.set_state_function("E", _External(self))

    def test_diamond_evaluates_each_node_once():
        """A dependency diamond (D <- B,C <- A) evaluates A exactly once.

        This is the whole point of the graph: without caching, a deep
        property tree re-evaluates its leaves combinatorially.
        """
        m = _Model()
        st = State(pressure=np.array([1.0, 2.0]))
        d = m.props.get(m, st, "D")
        assert np.allclose(d, [4.0, 8.0])       # 2p + 2p
        assert m.n_eval["A"] == 1, "A must be cached, not recomputed"
        assert m.n_eval["B"] == 1 and m.n_eval["C"] == 1
        print("ok test_diamond_evaluates_each_node_once")

    def test_cache_is_reused_across_gets():
        """A second get() of the same property does not re-evaluate."""
        m = _Model()
        st = State(pressure=np.array([1.0]))
        m.props.get(m, st, "D")
        before = dict(m.n_eval)
        m.props.get(m, st, "D")
        m.props.get(m, st, "B")
        assert m.n_eval == before
        print("ok test_cache_is_reused_across_gets")

    def test_external_dependency_crosses_groupings():
        """A function in one grouping pulls a value from another."""
        m = _Model()
        st = State(pressure=np.array([3.0]))
        e = m.flux.get(m, st, "E")
        assert np.allclose(e, [120.0])          # 10 * (2*3 + 2*3)
        assert "Props" in st.containers and "Flux" in st.containers
        print("ok test_external_dependency_crosses_groupings")

    def test_clearing_cache_forces_reevaluation():
        """A changed state invalidates everything computed from it."""
        m = _Model()
        st = State(pressure=np.array([1.0]))
        m.props.get(m, st, "D")
        st.pressure = np.array([5.0])
        st.clear_cache()
        assert np.allclose(m.props.get(m, st, "D"), [20.0])
        assert m.n_eval["A"] == 2
        print("ok test_clearing_cache_forces_reevaluation")

    def test_evaluation_order_is_topological():
        """evaluation_order lists a property after everything it needs."""
        order = _Model().props.evaluation_order()
        assert order.index("A") < order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")
        print("ok test_evaluation_order_is_topological")

    def test_cycle_is_detected():
        """A dependency cycle raises rather than recursing forever."""
        m = _Model()
        m.props.get_state_function("A").depends_on("D")
        try:
            m.props.get(m, State(pressure=np.array([1.0])), "D")
            raise SystemExit("cycle should have raised")
        except RuntimeError as exc:
            assert "cyclic" in str(exc)
        try:
            m.props.evaluation_order()
            raise SystemExit("cycle should have raised")
        except RuntimeError:
            pass
        print("ok test_cycle_is_detected")

    def test_unknown_function_raises():
        """Asking for a property nobody registered is an error, not a None."""
        try:
            _Model().props.get(_Model(), State(), "NoSuchThing")
            raise SystemExit("should have raised")
        except KeyError:
            pass
        print("ok test_unknown_function_raises")

    def test_regional_tables_keep_their_derivatives():
        """Two regions, two tables: each cell's value *and* derivative come
        from its own region's table. The old port kept values only."""
        from pymrst.utils.adi import init_variables_adi
        (sw,) = init_variables_adi(np.array([0.2, 0.4, 0.6]))
        prop = StateFunction(regions=[1, 2, 1])
        out = prop.evaluate_function_on_domain_with_arguments(
            [lambda s: s**2, lambda s: 3.0*s], sw)
        np.testing.assert_allclose(out.val, [0.04, 1.2, 0.36])
        diagonal = out.jac[0].toarray().diagonal()
        np.testing.assert_allclose(diagonal, [0.4, 3.0, 1.2])
        print("ok test_regional_tables_keep_their_derivatives")

    def test_a_single_region_map_uses_that_regions_table():
        """MRST `isSingleRegion`: every cell in region 2 means table 2, whole."""
        prop = StateFunction(regions=[2, 2, 2])
        assert prop.is_single_region
        out = prop.evaluate_function_on_domain_with_arguments(
            [lambda s: s, lambda s: 10.0*s], np.ones(3))
        np.testing.assert_allclose(out, 10.0)
        print("ok test_a_single_region_map_uses_that_regions_table")

    def test_a_region_list_without_a_map_is_refused():
        try:
            StateFunction().evaluate_function_on_domain_with_arguments(
                [abs, abs], np.ones(2))
            raise SystemExit("should have raised")
        except ValueError:
            pass
        print("ok test_a_region_list_without_a_map_is_refused")

    test_diamond_evaluates_each_node_once()
    test_cache_is_reused_across_gets()
    test_external_dependency_crosses_groupings()
    test_clearing_cache_forces_reevaluation()
    test_evaluation_order_is_topological()
    test_cycle_is_detected()
    test_unknown_function_raises()
    test_regional_tables_keep_their_derivatives()
    test_a_single_region_map_uses_that_regions_table()
    test_a_region_list_without_a_map_is_refused()
    print("\nAll tests passed!")
