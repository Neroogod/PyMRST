"""State-function graph introspection (MRST ad-core/statefunctions/utils).

Ports the reasoning behind MRST's `getStateFunctionGroupingDependencyGraph`,
`buildStateFunctionDigraph`, `plotStateFunctionGroupings` and
`printStateFunctionGroupingTikz`: walk every grouping on a model, resolve each
state function's dependencies across groupings, and produce the directed graph
that says what must be evaluated before what.

A `StateFunctionGrouping` already reports its own dependency graph and
evaluation order; this module lifts that to the *whole model* - the union of
all groupings, with cross-grouping and primary-variable edges - which is what
MRST draws to document or debug a model, and what a topological order over the
full graph is built from.

Nodes are fully qualified as ``"<struct_name>.<name>"`` (e.g.
``"FluxDisc.PhaseFlux"``); primary variables appear as ``"state.<var>"``. An
edge ``a -> b`` means *b depends on a* (a is evaluated first). The rendering
helpers emit Graphviz DOT and a compact text listing - no plotting backend is
required, so the structure can be checked as data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from pymrst.modules.ad_core.statefunctions.state_function import SELF, STATE


def _node(struct: str, name: str) -> str:
    return f"{struct}.{name}"


def get_state_function_grouping_dependency_graph(model: Any) -> Dict[str, Any]:
    """The full cross-grouping dependency graph of ``model``.

    Returns a dict with:
      ``nodes``    - every ``"struct.name"`` state-function node;
      ``inputs``   - the ``"state.<var>"`` primary-variable nodes referenced;
      ``edges``    - ``(from, to)`` pairs, meaning ``to`` depends on ``from``;
      ``category`` - node -> owning struct_name (for grouping / colouring).
    """
    nodes: List[str] = []
    category: Dict[str, str] = {}
    struct_of_group: Dict[str, str] = {}

    for gname in model.groupings:
        grp = getattr(model, gname)
        struct_of_group[gname] = grp.struct_name
        for fname in grp.get_names():
            nd = _node(grp.struct_name, fname)
            nodes.append(nd)
            category[nd] = grp.struct_name

    node_set = set(nodes)
    edges: Set[Tuple[str, str]] = set()
    inputs: Set[str] = set()

    for gname in model.groupings:
        grp = getattr(model, gname)
        for fname, deps in grp.dependency_graph().items():
            to = _node(grp.struct_name, fname)
            for dep in deps:
                if dep.grouping == STATE:
                    src = f"state.{dep.name}"
                    inputs.add(src)
                elif dep.grouping == SELF:
                    src = _node(grp.struct_name, dep.name)
                else:
                    other = getattr(model, dep.grouping, None)
                    struct = (other.struct_name if other is not None
                              else dep.grouping)
                    src = _node(struct, dep.name)
                if src not in node_set and not src.startswith("state."):
                    # a dependency on a grouping the model does not host, or a
                    # name it does not define - keep it as an (external) input
                    inputs.add(src)
                edges.add((src, to))

    return {
        "nodes": nodes,
        "inputs": sorted(inputs),
        "edges": sorted(edges),
        "category": category,
    }


def build_state_function_digraph(model: Any) -> Dict[str, Any]:
    """The dependency graph plus a topological order and cycle check.

    Adds ``order`` (a global evaluation order - every node after all it
    depends on) to the graph dict. Raises ``RuntimeError`` on a cycle.
    """
    g = get_state_function_grouping_dependency_graph(model)
    all_nodes = list(g["nodes"]) + list(g["inputs"])
    succ: Dict[str, List[str]] = {n: [] for n in all_nodes}
    indeg: Dict[str, int] = {n: 0 for n in all_nodes}
    for a, b in g["edges"]:
        succ.setdefault(a, []).append(b)
        indeg.setdefault(b, 0)
        indeg[b] += 1
        indeg.setdefault(a, indeg.get(a, 0))

    # Kahn's algorithm, deterministic (sorted) for reproducible output
    order: List[str] = []
    ready = sorted(n for n, d in indeg.items() if d == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in sorted(succ.get(n, [])):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort()
    if len(order) != len(indeg):
        raise RuntimeError("cyclic state-function dependency graph")
    g["order"] = order
    return g


def ancestors_of(graph: Dict[str, Any], target: str) -> Set[str]:
    """Every node ``target`` transitively depends on (its evaluation subtree)."""
    preds: Dict[str, List[str]] = {}
    for a, b in graph["edges"]:
        preds.setdefault(b, []).append(a)
    seen: Set[str] = set()
    stack = [target]
    while stack:
        n = stack.pop()
        for p in preds.get(n, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def to_dot(graph: Dict[str, Any]) -> str:
    """Graphviz DOT of the dependency graph (grouped by struct, one cluster
    each). Renderable by any Graphviz tool; needs no Python backend."""
    lines = ["digraph StateFunctions {", "  rankdir=LR;"]
    by_cat: Dict[str, List[str]] = {}
    for n, cat in graph["category"].items():
        by_cat.setdefault(cat, []).append(n)
    for cat, ns in sorted(by_cat.items()):
        lines.append(f'  subgraph "cluster_{cat}" {{ label="{cat}";')
        for n in sorted(ns):
            lines.append(f'    "{n}";')
        lines.append("  }")
    for src in graph.get("inputs", []):
        lines.append(f'  "{src}" [shape=box, style=dashed];')
    for a, b in graph["edges"]:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)


def format_grouping_text(model: Any) -> str:
    """A compact text listing: each node and what it depends on."""
    g = get_state_function_grouping_dependency_graph(model)
    preds: Dict[str, List[str]] = {}
    for a, b in g["edges"]:
        preds.setdefault(b, []).append(a)
    lines = []
    for n in g["nodes"]:
        deps = sorted(preds.get(n, []))
        lines.append(f"{n} <- {', '.join(deps) if deps else '(leaf)'}")
    return "\n".join(lines)


__all__ = [
    "ancestors_of",
    "build_state_function_digraph",
    "format_grouping_text",
    "get_state_function_grouping_dependency_graph",
    "to_dot",
]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([3, 3, 1], [30.0, 30.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        return GenericBlackOilModel(g, rock, fluid, w)

    def test_graph_covers_all_groupings():
        """Every state function of every grouping is a node."""
        model = _model()
        g = get_state_function_grouping_dependency_graph(model)
        for gname in model.groupings:
            grp = getattr(model, gname)
            for fname in grp.get_names():
                assert f"{grp.struct_name}.{fname}" in g["nodes"]
        # cross-grouping edge present: FluxDisc.PressureGradient depends on
        # nothing external here, but FaceMobility depends on FlowProps.Mobility
        assert any(a.startswith("FlowProps.Mobility") and
                   b.startswith("FluxDisc.FaceMobility") for a, b in g["edges"])
        print("ok test_graph_covers_all_groupings")

    def test_topological_order_respects_dependencies():
        """In the global order every node follows all it depends on."""
        model = _model()
        g = build_state_function_digraph(model)
        pos = {n: i for i, n in enumerate(g["order"])}
        for a, b in g["edges"]:
            assert pos[a] < pos[b], f"{a} must precede {b}"
        # a primary variable is an input, ordered before its consumers
        assert "state.pressure" in g["order"]
        print("ok test_topological_order_respects_dependencies")

    def test_ancestors_are_the_evaluation_subtree():
        """ComponentTotalFlux depends (transitively) on pressure and mobility,
        but not on, say, PoreVolume-only nodes it never reaches."""
        model = _model()
        g = get_state_function_grouping_dependency_graph(model)
        anc = ancestors_of(g, "FluxDisc.ComponentTotalFlux")
        assert "FluxDisc.PhaseFlux" in anc
        assert "FlowProps.Mobility" in anc
        assert "state.pressure" in anc
        # every ancestor edge target is reachable; sanity on closure
        assert "FluxDisc.ComponentTotalFlux" not in anc   # not its own ancestor
        print("ok test_ancestors_are_the_evaluation_subtree")

    def test_dot_and_text_render():
        """The DOT and text exports are well formed and mention real nodes."""
        model = _model()
        g = get_state_function_grouping_dependency_graph(model)
        dot = to_dot(g)
        assert dot.startswith("digraph StateFunctions {") and dot.endswith("}")
        assert '"FluxDisc.PhaseFlux"' in dot
        txt = format_grouping_text(model)
        assert "FluxDisc.PhaseFlux <-" in txt
        print("ok test_dot_and_text_render")

    test_graph_covers_all_groupings()
    test_topological_order_respects_dependencies()
    test_ancestors_are_the_evaluation_subtree()
    test_dot_and_text_render()
    print("\nAll tests passed!")
