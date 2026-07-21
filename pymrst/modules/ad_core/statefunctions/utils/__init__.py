"""MRST ad-core state-function graph utilities (`.../statefunctions/utils/`).

Introspection over the whole-model state-function graph: build the
cross-grouping dependency digraph, order it, find a node's evaluation subtree,
and export it (Graphviz DOT / text). Ports the reasoning of MRST's
`getStateFunctionGroupingDependencyGraph`, `buildStateFunctionDigraph`,
`plotStateFunctionGroupings` and `printStateFunctionGroupingTikz`.
"""

from pymrst.modules.ad_core.statefunctions.utils.dependency_graph import (
    ancestors_of,
    build_state_function_digraph,
    format_grouping_text,
    get_state_function_grouping_dependency_graph,
    to_dot,
)

__all__ = [
    "ancestors_of",
    "build_state_function_digraph",
    "format_grouping_text",
    "get_state_function_grouping_dependency_graph",
    "to_dot",
]
