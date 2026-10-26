"""MRST ``ad-core/statefunctions/utils/buildStateFunctionDigraph.m``.

The directed graph of a dependency matrix `C` (an edge `i -> j` where
`C[i, j]` is nonzero) with node `names`, reduced to the nodes that matter for
`Start` (those reachable from the named nodes), `Stop` (those the named nodes
are reachable from) and `Center` (either), each a name or a list of names.
A name is looked up among `names` and then, ignoring case, among
`FilterNames` (default `names`), taking the first match with a warning when
there are several. Returns `(G, names, category, keep)`.

MATLAB returns a `digraph`; PyMRST returns its parts: `Nodes`, `Edges` (a
`(nedge, 2)` array of 0-based node indices, sorted as MATLAB's edge table is)
and the `Adjacency` matrix of the kept nodes.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


def build_state_function_digraph(C: Any, names: Sequence[str], category: Sequence[Any], *,
                                 Start: Any = (), Stop: Any = (), Center: Any = (),
                                 FilterNames: Any = None
                                 ) -> Tuple[SimpleNamespace, list, list, np.ndarray]:
    """MRST `buildStateFunctionDigraph`."""
    names = list(names)
    filter_names = names if FilterNames is None else list(FilterNames)
    A = sp.csr_array(np.asarray(C, dtype=float) if not sp.issparse(C) else C)
    n = len(names)
    keep = np.ones(n, dtype=bool)
    has_start, has_stop, has_center = (_given(Start), _given(Stop), _given(Center))
    if has_stop or has_center:
        D = shortest_path(A, directed=True, unweighted=True)
    if has_start or has_center:
        Dr = shortest_path(A.T, directed=True, unweighted=True)
    if has_start:
        keep &= _filter(Dr, names, filter_names, Start)
    if has_stop:
        keep &= _filter(D, names, filter_names, Stop)
    if has_center:
        keep &= (_filter(D, names, filter_names, Center)
                 | _filter(Dr, names, filter_names, Center))
    keep_ix = np.flatnonzero(keep)
    sub = A[keep_ix][:, keep_ix].tocoo()
    order = np.lexsort((sub.col, sub.row))
    edges = np.column_stack([sub.row[order], sub.col[order]])
    G = SimpleNamespace(Nodes=[names[k] for k in keep_ix], Edges=edges,
                        Adjacency=A[keep_ix][:, keep_ix])
    category = [category[k] for k in keep_ix]
    return G, G.Nodes, category, keep


def _given(targets: Any) -> bool:
    return isinstance(targets, str) or len(targets) > 0


def _filter(D: np.ndarray, names: Sequence[str], shortnames: Sequence[str],
            targets: Any) -> np.ndarray:
    keep = np.zeros(len(names), dtype=bool)
    if isinstance(targets, str):
        targets = [targets]
    lower = [x.lower() for x in names]
    short = [x.lower() for x in shortnames]
    for name in targets:
        pos = np.array([x == name.lower() for x in lower])
        if not pos.any():
            pos = np.array([x == name.lower() for x in short])
            if pos.sum() > 1:
                first = int(np.flatnonzero(pos)[0])
                pos = np.zeros_like(pos)
                pos[first] = True
                warnings.warn(f"Multiple entries found for {name}. Arbitrarily taking "
                              f"{names[first]}. Does it exist in multiple StateGrouping "
                              f"instances?")
        keep |= np.isfinite(D[:, pos]).any(axis=1)
    return keep


buildStateFunctionDigraph = build_state_function_digraph

__all__ = ["buildStateFunctionDigraph", "build_state_function_digraph"]


if __name__ == "__main__":
    # a -> b -> c, d -> c, e alone
    C = np.zeros((5, 5))
    C[0, 1] = C[1, 2] = C[3, 2] = 1
    names = ["g.a", "g.b", "g.c", "g.d", "g.e"]
    short = ["a", "b", "c", "d", "e"]

    def test_start_keeps_what_the_start_reaches():
        _, nm, cat, keep = build_state_function_digraph(C, names, [1]*5, Start="g.b")
        assert nm == ["g.b", "g.c"] and cat == [1, 1] and keep.sum() == 2
        print("ok test_start_keeps_what_the_start_reaches")

    def test_stop_keeps_what_reaches_the_stop():
        G, nm, _, _ = build_state_function_digraph(C, names, [1]*5, Stop=["c"],
                                                   FilterNames=short)
        assert nm == ["g.a", "g.b", "g.c", "g.d"]
        assert G.Edges.tolist() == [[0, 1], [1, 2], [3, 2]]
        print("ok test_stop_keeps_what_reaches_the_stop")

    def test_center_keeps_both_directions_and_no_filter_keeps_all():
        _, nm, _, _ = build_state_function_digraph(C, names, [1]*5, Center="g.b")
        assert nm == ["g.a", "g.b", "g.c"]
        _, nm, _, keep = build_state_function_digraph(C, names, [1]*5)
        assert nm == names and keep.all()
        print("ok test_center_keeps_both_directions_and_no_filter_keeps_all")

    def test_ambiguous_short_name_warns_and_takes_the_first():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, nm, _, _ = build_state_function_digraph(C, names, [1]*5, Start="x",
                                                       FilterNames=["x", "b", "c", "x", "e"])
        assert nm == ["g.a", "g.b", "g.c"] and "Multiple entries" in str(caught[0].message)
        print("ok test_ambiguous_short_name_warns_and_takes_the_first")

    test_start_keeps_what_the_start_reaches()
    test_stop_keeps_what_reaches_the_stop()
    test_center_keeps_both_directions_and_no_filter_keeps_all()
    test_ambiguous_short_name_warns_and_takes_the_first()
    print("\nAll tests passed!")
