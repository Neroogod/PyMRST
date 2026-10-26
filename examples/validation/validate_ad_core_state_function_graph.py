"""Validate MRST getStateFunctionGroupingDependencyGraph and
buildStateFunctionDigraph, the ports in `ad_core/statefunctions/utils/`.

The reference is MRST 2024a (`dump_state_function_graph_reference.m`): a
three-phase `GenericBlackOilModel` on a 3x3 grid, its four groupings
(flow properties, PVT, flow discretization, facility flow discretization).
- The dependency graph: the same nodes, with the same implementation, label,
  grouping and group type, and the same edges. MRST orders a grouping's
  functions as its class properties (sorted) and then the added ones; PyMRST
  groupings have no class properties, so the nodes are matched by grouping
  and name and the matrices compared under that matching.
- The digraph kept by the `Start`, `Stop` and `Center` filters (with names
  qualified by grouping, and bare names resolved through `FilterNames`, as
  `plotStateFunctionGroupings` passes them): the same nodes and edges.
MRST's `setupStateFunctionGraph` needs `matlab_bgl`'s MEX, which does not run
on this machine, so its evaluation order is not compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.statefunctions.utils.build_state_function_digraph import (
    build_state_function_digraph,
)
from pymrst.modules.ad_core.statefunctions.utils.get_state_function_grouping_dependency_graph import (  # noqa: E501
    get_state_function_grouping_dependency_graph,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock

HERE = Path(__file__).resolve().parent
FILTERS = ({"Start": "PhasePressures"}, {"Stop": "FaceMobility"}, {"Center": "Mobility"},
           {"Start": "PVTPropertyFunctions.Density"})


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def main() -> int:
    ref = loadmat(HERE/"ref_state_function_graph.mat", squeeze_me=True)
    G = compute_geometry(cart_grid([3, 3]))
    fluid = init_simple_adi_fluid(phases="WOG", n=[2, 2, 2], rho=[1000, 800, 100],
                                  mu=[1, 5, 0.1])
    model = GenericBlackOilModel(G, make_rock(G, 1, 1), fluid).validate_model()
    groups, _, _ = model.get_state_function_groupings()
    graph = get_state_function_grouping_dependency_graph(*groups)
    ok = report(graph.GroupNames == list(ref["GroupNames"]),
                f"group names {graph.GroupNames}")
    full = [f"{graph.GroupNames[c]}.{n}" for c, n in zip(graph.GroupIndex, graph.FunctionNames)]
    ref_full = [f"{ref['GroupNames'][c - 1]}.{n}"
                for c, n in zip(ref["GroupIndex"], ref["FunctionNames"])]
    ok &= report(sorted(full) == sorted(ref_full) and len(set(full)) == len(full),
                 f"the same {len(full)} nodes")
    if not ok:
        print("     only in PyMRST:", sorted(set(full) - set(ref_full)))
        print("     only in MRST:", sorted(set(ref_full) - set(full)))
        return 1
    perm = np.array([full.index(name) for name in ref_full])
    for field in ("Implementation", "Labels"):
        mine = [getattr(graph, field)[k] for k in perm]
        bad = [(ref_full[k], a, b) for k, (a, b) in enumerate(zip(mine, ref[field])) if a != b]
        ok &= report(not bad, f"{field} identical" + (f"; differ: {bad[:4]}" if bad else ""))
    ok &= report(np.array_equal(graph.GroupTypes[perm], ref["GroupTypes"]),
                 "group types identical")
    C = graph.C[np.ix_(perm, perm)]
    Cref = np.asarray(ref["C"], dtype=bool)
    diff = [(ref_full[i], ref_full[j], bool(C[i, j])) for i, j in zip(*np.nonzero(C != Cref))]
    ok &= report(not diff, f"dependency matrix identical ({int(Cref.sum())} edges)"
                 + (f"; differ (from, to, PyMRST): {diff[:6]}" if diff else ""))
    for k, opts in enumerate(FILTERS, start=1):
        D, names, _, keep = build_state_function_digraph(graph.C, full, graph.GroupIndex,
                                                         FilterNames=graph.FunctionNames,
                                                         **opts)
        ref_names = list(np.atleast_1d(ref[f"names{k}"]))
        edges = sorted((names[a], names[b]) for a, b in D.Edges)
        ref_edges = sorted((ref_names[a - 1], ref_names[b - 1])
                           for a, b in np.atleast_2d(ref[f"edges{k}"]))
        ok &= report(sorted(names) == sorted(ref_names) and edges == ref_edges,
                     f"digraph {opts}: {len(names)} nodes and {len(edges)} edges identical")
    print("\nThe state-function graph utilities match MRST" if ok
          else "\nThe state-function graph utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
