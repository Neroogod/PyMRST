"""MRST ``ad-core/utils/convert2MSWell.m``.

Give a standard well the nodes and segments of a multisegment well. Nodes
gather perforations through `cell2node` (nodes x perforations, by default
one node per perforation; a vector names each perforation's node), and sit
at `nodeDepth` (by default the reference depth plus the mean `dZ` of their
perforations). Segments join pairs of nodes, `topo` (by default a single
branch), with lengths from `segLength` (by default the distance between the
mean perforated-cell centroids of the nodes, which needs `G`) and diameters
from `segDiam` (by default the sum of the mean well radii of the two nodes).
The well is flagged `isMS`.

Node and segment numbers are 0-based, as cells are in PyMRST: MRST's
`topo = [1 2; 2 3]` is `[[0, 1], [1, 2]]` here. Options keep MRST's names.

As in MRST, `segRoughness` and `segFlowModel` are read from fields that do
not exist (`opt.roughness`, `opt.flowModel`), and a valve segment (`segType`
3) looks up an undefined `s`; each raises, as MATLAB does.
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

VALVE_SEGMENT: int = 3
TUBING_SEGMENT: int = 1


def convert2_ms_well(w: Any, *, cell2node: Any = None, connDZ: Any = None,
                     nodeDepth: Any = None, topo: Any = None, segLength: Any = None,
                     segRoughness: Any = None, segFlowModel: Any = None,
                     segType: Any = None, segDiam: Any = None, G: Any = None,
                     vol: Any = None) -> Any:
    """MRST `w = convert2MSWell(w, ...)`."""
    opt = SimpleNamespace(cell2node=cell2node, connDZ=connDZ, nodeDepth=nodeDepth,
                          topo=topo, segLength=segLength, segRoughness=segRoughness,
                          segFlowModel=segFlowModel, segType=segType, segDiam=segDiam,
                          G=G, vol=vol)
    w = copy(w)
    w.cell2node = _create_cell2node(np.asarray(w.cells).size, opt)
    cell_count = np.maximum(np.asarray(w.cell2node.sum(axis=1)).ravel(), 1)
    nn, nc = w.cell2node.shape
    nodes = SimpleNamespace(depth=np.full(nn, np.nan), vol=np.ones(nn),
                            dist=np.full(nn, np.nan))
    nodes.depth = _node_depth(w, cell_count, opt)
    if opt.vol is not None:
        nodes.vol = np.asarray(opt.vol, dtype=float).ravel()
    w.connDZ = opt.connDZ
    if w.connDZ is None:
        w.connDZ = np.zeros(np.shape(w.dZ))
    s = _create_segments(_node_topology(nn, opt))
    s.length = _segment_length(w, s.topo, cell_count, opt)
    s.diam = _segment_diameter(w, s.topo, nc, cell_count, opt)
    s.deltaDistance = _segment_delta_distance(s.length, s.topo, opt)
    if opt.segRoughness is not None:
        raise AttributeError("Reference to non-existent field 'roughness'.")
    if opt.segFlowModel is not None:
        raise AttributeError("Reference to non-existent field 'flowModel'.")
    w.nodes = nodes
    w.segments = s
    w.isMS = True
    return w


convert2MSWell = convert2_ms_well


def _create_cell2node(nperf: int, opt: SimpleNamespace) -> sp.csr_matrix:
    c2n = opt.cell2node
    if c2n is None:
        return sp.identity(nperf, format="csr")
    if not sp.issparse(c2n) and (np.ndim(c2n) == 1 or np.shape(c2n)[1] == 1):
        nodes = np.asarray(c2n, dtype=np.int64).ravel()
        return sp.csr_matrix((np.ones(nperf), (nodes, np.arange(nperf))))
    return sp.csr_matrix(c2n, dtype=float)


def _node_depth(w: Any, cell_count: np.ndarray, opt: SimpleNamespace) -> np.ndarray:
    if opt.nodeDepth is None:
        dz = np.asarray(w.dZ, dtype=float).ravel()
        return w.refDepth + (w.cell2node @ dz)/cell_count
    return np.asarray(opt.nodeDepth, dtype=float).ravel()


def _node_topology(nn: int, opt: SimpleNamespace) -> np.ndarray:
    if opt.topo is None:
        return np.column_stack([np.arange(nn - 1), np.arange(1, nn)])
    return np.asarray(opt.topo, dtype=np.int64).reshape(-1, 2)


def _create_segments(topo: np.ndarray) -> SimpleNamespace:
    ns = topo.shape[0]
    return SimpleNamespace(length=np.zeros(ns), roughness=np.zeros(ns),
                           diam=np.zeros(ns), flowModel=np.zeros(ns), topo=topo)


def _segment_length(w: Any, topo: np.ndarray, cell_count: np.ndarray,
                    opt: SimpleNamespace) -> np.ndarray:
    if opt.segLength is not None:
        return np.asarray(opt.segLength, dtype=float).ravel()
    assert opt.G is not None
    cc = np.asarray(opt.G.cells.centroids, dtype=float)[np.asarray(w.cells).ravel(), :]
    cn = (w.cell2node @ cc)/cell_count[:, None]
    ds = cn[topo[:, 1], :] - cn[topo[:, 0], :]
    return np.sqrt(np.sum(ds**2, axis=1))


def _segment_diameter(w: Any, topo: np.ndarray, nc: int, cell_count: np.ndarray,
                      opt: SimpleNamespace) -> np.ndarray:
    if opt.segDiam is not None:
        return np.asarray(opt.segDiam, dtype=float).ravel()
    rc = np.asarray(w.r, dtype=float).ravel()
    if rc.size == 1:
        rc = np.repeat(rc, nc)
    rn = (w.cell2node @ rc)/cell_count
    return rn[topo[:, 1]] + rn[topo[:, 0]]


def _segment_delta_distance(length: np.ndarray, topo: np.ndarray,
                            opt: SimpleNamespace) -> np.ndarray:
    dist = np.zeros(length.shape)
    if opt.segType is not None:
        out_from_tube = np.flatnonzero(np.asarray(opt.segType).ravel() == VALVE_SEGMENT)
        if out_from_tube.size:
            # MRST's subfunction reads `s.topo` with no `s` in scope.
            raise NameError("Undefined function or variable 's'.")
    return dist


__all__ = ["convert2_ms_well", "convert2MSWell"]


if __name__ == "__main__":
    def _well(ncells: int = 3) -> SimpleNamespace:
        return SimpleNamespace(cells=np.arange(ncells), dZ=np.arange(ncells)*2.0,
                               refDepth=100.0, r=0.1, name="w")

    def _grid(ncells: int = 3) -> SimpleNamespace:
        centroids = np.column_stack([np.arange(ncells)*10.0, np.zeros(ncells),
                                     np.zeros(ncells)])
        return SimpleNamespace(cells=SimpleNamespace(centroids=centroids))

    def test_defaults_give_one_node_per_perforation_on_a_branch():
        w = convert2_ms_well(_well(), G=_grid())
        assert w.isMS and w.cell2node.shape == (3, 3)
        np.testing.assert_array_equal(w.segments.topo, [[0, 1], [1, 2]])
        np.testing.assert_allclose(w.nodes.depth, [100.0, 102.0, 104.0])
        np.testing.assert_allclose(w.segments.length, [10.0, 10.0])
        np.testing.assert_allclose(w.segments.diam, [0.2, 0.2])
        np.testing.assert_array_equal(w.connDZ, np.zeros(3))
        print("ok test_defaults_give_one_node_per_perforation_on_a_branch")

    def test_a_node_vector_groups_perforations():
        w = convert2_ms_well(_well(), cell2node=np.array([0, 0, 1]), G=_grid())
        assert w.cell2node.shape == (2, 3)
        np.testing.assert_allclose(w.nodes.depth, [101.0, 104.0])
        np.testing.assert_allclose(w.segments.length, [15.0])
        print("ok test_a_node_vector_groups_perforations")

    def test_given_geometry_is_kept_and_the_input_is_not_changed():
        original = _well()
        topo = np.array([[0, 1], [0, 2]])
        w = convert2_ms_well(original, topo=topo, nodeDepth=[1.0, 2.0, 3.0],
                             segLength=[5.0, 6.0], segDiam=[0.3, 0.4], vol=[1, 2, 3])
        np.testing.assert_allclose(w.segments.length, [5.0, 6.0])
        np.testing.assert_allclose(w.nodes.vol, [1.0, 2.0, 3.0])
        assert not hasattr(original, "isMS")
        print("ok test_given_geometry_is_kept_and_the_input_is_not_changed")

    def test_mrst_option_bugs_raise_as_in_matlab():
        for kwargs in ({"segRoughness": [1e-4, 1e-4]},
                       {"segType": [1, 3]}):
            try:
                convert2_ms_well(_well(), G=_grid(), **kwargs)
            except (AttributeError, NameError):
                continue
            raise AssertionError(f"{kwargs} should raise as MRST does")
        print("ok test_mrst_option_bugs_raise_as_in_matlab")

    test_defaults_give_one_node_per_perforation_on_a_branch()
    test_a_node_vector_groups_perforations()
    test_given_geometry_is_kept_and_the_input_is_not_changed()
    test_mrst_option_bugs_raise_as_in_matlab()
    print("\nAll tests passed!")
