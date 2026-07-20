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

    def __init__(self, model: Any = None, label: Optional[str] = None):
        self.dependencies: List[Dependency] = []
        self.label = label or type(self).__name__
        self.struct_name: Optional[str] = None

    def depends_on(self, names: Union[str, Sequence[str]],
                   grouping: str = SELF) -> "StateFunction":
        """Declare dependencies. Returns self so calls can be chained."""
        if isinstance(names, str):
            names = [names]
        for n in names:
            dep = Dependency(n, grouping)
            if dep not in self.dependencies:
                self.dependencies.append(dep)
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r})"


class StateFunctionGrouping:
    """A named set of state functions, cached on the state under struct_name."""

    def __init__(self, struct_name: str):
        self.struct_name = struct_name
        self.functions: Dict[str, StateFunction] = {}

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
        return name in state.containers.get(self.struct_name, {})

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
            group = self if dep.grouping == SELF else getattr(model,
                                                              dep.grouping)
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

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.struct_name!r}, "
                f"{len(self.functions)} functions)")


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

    test_diamond_evaluates_each_node_once()
    test_cache_is_reused_across_gets()
    test_external_dependency_crosses_groupings()
    test_clearing_cache_forces_reevaluation()
    test_evaluation_order_is_topological()
    test_cycle_is_detected()
    test_unknown_function_raises()
    print("\nAll tests passed!")
