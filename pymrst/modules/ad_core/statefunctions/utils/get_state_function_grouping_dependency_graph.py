"""MRST ``ad-core/statefunctions/utils/getStateFunctionGroupingDependencyGraph.m``.

The dependency graph of one or more state-function groupings. Nodes are every
state function of the groupings, followed by the external dependencies none of
them defines (such as `state.pressure`), sorted by `grouping.name`.
`C[i, j]` is true when node `j` depends on node `i`. Each node has its
implementation (class name, `'state'` or `'?'` for an external), name, label
(`Labels`, the name when the function has none), and grouping (`GroupIndex`
into `GroupNames`: the groupings' class names, then the external groupings,
sorted). `GroupTypes` is 1 inside the given groupings, -1 outside and 0 for the
state. A dependency on a base grouping class is attributed to a given grouping
that is an instance of it.

MRST orders a grouping's functions as its class-declared ones (sorted) and then
the ones added later; PyMRST's groupings register every function the same way,
so their order is the order of registration. Indices (`GroupIndex`) are
0-based.

PyMRST keeps every dependency in `dependencies`, with its grouping, where MRST
keeps the names in the same grouping in `dependencies` and the rest, the state
included, in `externals`; the graph is built from MRST's split.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np

SELF = "self"
STATE = "state"


def get_state_function_grouping_dependency_graph(*function_groupings: Any) -> SimpleNamespace:
    """MRST `getStateFunctionGroupingDependencyGraph`."""
    group_names = [type(g).__name__ for g in function_groupings]
    names, implementation, symbols, observed, category = [], [], [], [], []
    for i, group in enumerate(function_groupings):
        n, obs, impl, sym = _get_names(group)
        names += n
        implementation += impl
        symbols += sym
        observed += obs
        category += [i]*len(n)
    category = np.array(category, dtype=int)
    names_full = [f"{group_names[c]}.{n}" for c, n in zip(category, names)]
    observed = _fix_subclass_dependencies(function_groupings, observed)
    dep_full = [f"{g}.{n}" for g, n in observed]
    observed_deps = sorted(set(dep_full) - set(names_full))
    all_dependencies = [observed[dep_full.index(d)] for d in observed_deps]
    nnames = len(names)
    n = nnames + len(observed_deps)
    A = np.zeros((n, n), dtype=bool)
    for i, name in enumerate(names):
        c = category[i]
        sf = function_groupings[c].get_state_function(name)
        local = category == c
        A[:nnames, i] = np.isin(names, _local_dependencies(sf)) & local
        for d in _fix_subclass_dependencies(function_groupings, _externals(sf)):
            A[nnames:, i] |= np.array([dep == d for dep in all_dependencies], dtype=bool)
            if d[0] in group_names:
                A[:nnames, i] |= (np.isin(names, [d[1]])
                                  & (category == group_names.index(d[0])))
    n_base_group = len(group_names)
    ext_groups = sorted({g for g, _ in all_dependencies})
    group_names = group_names + sorted(set(ext_groups) - set(group_names))
    dep_cat = [group_names.index(g) for g, _ in all_dependencies]
    dep_impl = [STATE if g.lower() == STATE else "?" for g, _ in all_dependencies]
    dep_names = [name for _, name in all_dependencies]
    I = implementation + dep_impl
    N = names + dep_names
    S = symbols + dep_names
    C = np.concatenate([category, np.array(dep_cat, dtype=int)])
    group_types = 1 - 2*(C >= n_base_group).astype(int)
    group_types[np.array([group_names[c] == STATE for c in C], dtype=bool)] = 0
    assert len(I) == len(C) == len(N) == len(S)
    return SimpleNamespace(C=A, Implementation=I, FunctionNames=N, Labels=S,
                           GroupIndex=C, GroupTypes=group_types, GroupNames=group_names)


def _local_dependencies(sf: Any) -> List[str]:
    return [d.name for d in sf.dependencies if d.grouping == SELF]


def _externals(sf: Any) -> List[Tuple[str, str]]:
    return [(d.grouping, d.name) for d in sf.dependencies if d.grouping != SELF]


def _fix_subclass_dependencies(groups: Any, deps: List[Tuple[str, str]]
                               ) -> List[Tuple[str, str]]:
    """A dependency on a grouping class goes to the first grouping that is one."""
    fixed = []
    for grouping, name in deps:
        for g in groups:
            if any(cls.__name__ == grouping for cls in type(g).__mro__):
                grouping = type(g).__name__
                break
        fixed.append((grouping, name))
    return fixed


def _get_names(group: Any):
    if group is None:
        raise ValueError("Empty state function. Did you forget to set it up?")
    names = list(group.get_names())
    impl, symbols, observed = [], [], []
    for name in names:
        prop = group.get_state_function(name)
        impl.append(type(prop).__name__)
        symbols.append(prop.label or name)
        observed += _externals(prop)
    return names, observed, impl, symbols


getStateFunctionGroupingDependencyGraph = get_state_function_grouping_dependency_graph

__all__ = ["getStateFunctionGroupingDependencyGraph",
           "get_state_function_grouping_dependency_graph"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model

    def _graph():
        model = toy_model("WO", nx=3)
        groups, _, _ = model.get_state_function_groupings()
        return model, get_state_function_grouping_dependency_graph(*groups)

    def test_nodes_are_functions_then_sorted_state_dependencies():
        model, graph = _graph()
        groups, _, _ = model.get_state_function_groupings()
        nnames = sum(len(g.get_names()) for g in groups)
        tail = graph.FunctionNames[nnames:]
        assert tail == sorted(tail) and "pressure" in tail
        assert all(graph.Implementation[k] == "state" for k in range(nnames, len(tail) + nnames))
        assert graph.GroupNames[-1] == "state"
        assert set(graph.GroupTypes[nnames:]) == {0}
        assert set(graph.GroupTypes[:nnames]) == {1}
        print("ok test_nodes_are_functions_then_sorted_state_dependencies")

    def test_edges_follow_declared_dependencies():
        model, graph = _graph()
        names = graph.FunctionNames
        groups = [graph.GroupNames[c] for c in graph.GroupIndex]
        node = {(g, n): k for k, (g, n) in enumerate(zip(groups, names))}
        mob = node[("FlowPropertyFunctions", "Mobility")]
        kr = node[("FlowPropertyFunctions", "RelativePermeability")]
        mu = node[("PVTPropertyFunctions", "Viscosity")]
        assert graph.C[kr, mob] and graph.C[mu, mob] and not graph.C[mob, kr]
        print("ok test_edges_follow_declared_dependencies")

    def test_empty_grouping_is_an_error():
        try:
            get_state_function_grouping_dependency_graph(None)
        except ValueError:
            pass
        else:
            raise AssertionError("expected a ValueError")
        print("ok test_empty_grouping_is_an_error")

    test_nodes_are_functions_then_sorted_state_dependencies()
    test_edges_follow_declared_dependencies()
    test_empty_grouping_is_an_error()
    print("\nAll tests passed!")
