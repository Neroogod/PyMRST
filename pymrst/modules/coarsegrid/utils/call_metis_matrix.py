"""Partition a connection matrix with METIS (MRST ``utils/callMetisMatrix.m``).

MRST writes the graph to a temporary file, shells out to the `gpmetis` binary
and reads the partition back. This calls METIS in process through `pymetis`,
which is the same library without the file round trip.

The graph is the matrix's sparsity pattern with the diagonal removed: vertices
are cells, edges are connections, and an edge's weight is its connection
strength. That is what makes the partition follow the flow rather than the
geometry, so a high-permeability channel is not cut in half.

**Weights are integers, and the scaling is MRST's.** METIS takes integer edge
weights, so MRST maps the magnitudes onto 1 to 10000 with
``ceil(1e4 * |v| / max|v|)``. Two details matter and both are reproduced here:
the absolute value, since a TPFA off-diagonal is negative and a negative weight
is meaningless to METIS, and the ceiling, which keeps the weakest connection at
1 rather than 0. A connection rounded to 0 is a connection METIS is free to cut
for nothing.

**`pymetis` is optional** (`pip install pymrst[metis]`). Without it, the graph
is still built and can be inspected; only the partitioning call raises, with a
message saying what to install. The rest of the port needs nothing but numpy and
scipy, and so does everything here except that one call.

A substitute partitioner was considered and rejected. `scipy.sparse.csgraph` and
`networkx` can both partition a graph, but not the way METIS does, so the
partitions would differ and parity would be lost rather than preserved. An
optional native dependency matches how `pyamg` and `torch` are already handled.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:                                     # optional: the graph partitioner
    import pymetis
    HAVE_METIS = True
except ImportError:                      # pragma: no cover
    pymetis = None
    HAVE_METIS = False

WEIGHT_SCALE = 10_000


def metis_graph(a):
    """Return ``(xadj, adjncy, eweights)``, the CSR graph METIS wants.

    Built from the off-diagonal pattern of `a`, with weights scaled to
    integers. Needs no METIS installation, so the construction stays testable.
    """
    a = sp.coo_matrix(a)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"the connection matrix must be square, got {a.shape}")

    off = a.row != a.col
    rows, cols, values = a.row[off], a.col[off], np.abs(a.data[off])
    if values.size == 0:
        raise ValueError("the connection matrix has no off-diagonal entries")

    largest = values.max()
    weights = np.ceil(WEIGHT_SCALE * values / largest).astype(np.int64)

    n = a.shape[0]
    order = np.lexsort((cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    xadj = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])
    return xadj.astype(np.int64), cols.astype(np.int64), weights


def call_metis_matrix(a, n: int, ufactor: float = 1.5, ncuts: int = 1,
                      seed: int = 0, niter: int = 10, minconn: bool = True,
                      contig: bool = True, no2hop: bool = False) -> np.ndarray:
    """Partition the graph of `a` into `n` blocks. Returns a 0-based partition.

    The keyword defaults are MRST's, taken from `partitionMETIS`.
    """
    if n <= 1:
        raise ValueError(f"the number of blocks must be greater than one, got {n}")
    xadj, adjncy, weights = metis_graph(a)
    if not HAVE_METIS:
        raise ImportError(
            "partitioning with METIS needs the optional 'pymetis' package: "
            "pip install pymrst[metis]. The graph itself is available from "
            "metis_graph() without it.")

    options = pymetis.Options()
    # MRST passes -ufactor as an integer permille of allowed imbalance, so its
    # 1.5 means 500, meaning blocks may vary by 50% in size.
    options.ufactor = int(np.floor(1000 * (ufactor - 1.0)))
    options.ncuts = int(ncuts)
    options.seed = int(seed)
    options.niter = int(niter)
    options.minconn = bool(minconn)
    options.contig = bool(contig)
    options.no2hop = bool(no2hop)

    _, membership = pymetis.part_graph(
        int(n), xadj=xadj, adjncy=adjncy, eweights=weights, options=options)
    return np.asarray(membership, dtype=np.int64)


__all__ = ["HAVE_METIS", "WEIGHT_SCALE", "call_metis_matrix", "metis_graph"]


if __name__ == "__main__":
    def _matrix(n=6, strength=None):
        """A 1D chain's TPFA matrix, with given connection strengths."""
        if strength is None:
            strength = np.ones(n - 1)
        a = sp.lil_matrix((n, n))
        for i, t in enumerate(strength):
            a[i, i + 1] = -t
            a[i + 1, i] = -t
            a[i, i] += t
            a[i + 1, i + 1] += t
        return a.tocsr()

    def test_the_graph_is_the_off_diagonal_pattern():
        """The diagonal is a cell's own accumulation, not a connection.

        Leaving it in gives every vertex a self-loop, which METIS accepts and
        which changes the balance it computes.
        """
        a = _matrix(4)
        xadj, adjncy, _ = metis_graph(a)
        assert xadj.tolist() == [0, 1, 3, 5, 6], xadj.tolist()
        assert adjncy.tolist() == [1, 0, 2, 1, 3, 2], adjncy.tolist()
        for v in range(4):
            assert v not in adjncy[xadj[v]:xadj[v + 1]].tolist(), v
        print("ok test_the_graph_is_the_off_diagonal_pattern")

    def test_weights_are_positive_integers():
        """A TPFA off-diagonal is negative; METIS needs a positive integer.

        Passing the signed value through would either be rejected or read as a
        tiny weight, and the strongest connections would be the first cut.
        """
        a = _matrix(5, np.array([1.0, 10.0, 100.0, 1000.0]))
        _, _, w = metis_graph(a)
        assert w.dtype.kind == "i", w.dtype
        assert np.all(w >= 1), w
        assert w.max() == WEIGHT_SCALE, w.max()
        print(f"ok test_weights_are_positive_integers ({w.tolist()})")

    def test_the_weakest_connection_never_rounds_to_zero():
        """MRST ceils rather than rounds, and the difference matters.

        A connection weighted 0 costs nothing to cut, so METIS will cut it
        first. With a 1e6 contrast, rounding would zero it.
        """
        a = _matrix(3, np.array([1.0, 1e7]))
        _, _, w = metis_graph(a)
        assert w.min() >= 1, w
        assert np.round(WEIGHT_SCALE * 1.0 / 1e7) == 0, "the case is too weak"
        print(f"ok test_the_weakest_connection_never_rounds_to_zero ({w.min()})")

    def test_weights_are_ordered_by_connection_strength():
        a = _matrix(5, np.array([1.0, 4.0, 2.0, 3.0]))
        _, adjncy, w = metis_graph(a)
        # Edge (0,1) is weakest, (1,2) strongest.
        xadj, _, _ = metis_graph(a)
        first = w[xadj[0]:xadj[1]][0]
        assert first == int(np.ceil(WEIGHT_SCALE * 1.0 / 4.0)), first
        assert w.max() == WEIGHT_SCALE
        print("ok test_weights_are_ordered_by_connection_strength")

    def test_the_graph_is_symmetric():
        """METIS wants an undirected graph; a one-sided edge is rejected."""
        a = _matrix(6)
        xadj, adjncy, w = metis_graph(a)
        edges = set()
        for v in range(6):
            for k in range(xadj[v], xadj[v + 1]):
                edges.add((v, int(adjncy[k])))
        for u, v in edges:
            assert (v, u) in edges, (u, v)
        print(f"ok test_the_graph_is_symmetric ({len(edges)} directed edges)")

    def test_it_works_on_a_real_tpfa_matrix():
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.coarsegrid.partition_metis import connection_matrix
        from pymrst.params.rock.make_rock import make_rock
        from pymrst.solvers.compute_trans import compute_trans
        from pymrst.utils.units import DARCY

        g = compute_geometry(cart_grid([4, 4, 1], [4.0, 4.0, 1.0]))
        rock = make_rock(g, perm=DARCY, poro=0.3)
        a = connection_matrix(g, compute_trans(g, rock))
        xadj, adjncy, w = metis_graph(a)
        assert xadj.size == g.cells.num + 1
        assert adjncy.size == 2 * int(np.count_nonzero(
            np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)))
        assert np.all(w >= 1)
        print(f"ok test_it_works_on_a_real_tpfa_matrix "
              f"({adjncy.size} directed edges)")

    def test_a_matrix_with_no_connections_is_rejected():
        a = sp.eye(4, format="csr")
        try:
            metis_graph(a)
        except ValueError as exc:
            assert "off-diagonal" in str(exc), exc
        else:
            raise AssertionError("expected a diagonal matrix to be rejected")
        print("ok test_a_matrix_with_no_connections_is_rejected")

    def test_one_block_is_rejected():
        try:
            call_metis_matrix(_matrix(4), 1)
        except ValueError as exc:
            assert "greater than one" in str(exc), exc
        else:
            raise AssertionError("expected n=1 to be rejected")
        print("ok test_one_block_is_rejected")

    def test_the_missing_package_says_what_to_install():
        """A bare ImportError from a transitive call is unhelpful."""
        if HAVE_METIS:
            print("ok test_the_missing_package_says_what_to_install (pymetis present)")
            return
        try:
            call_metis_matrix(_matrix(6), 2)
        except ImportError as exc:
            assert "pip install pymrst[metis]" in str(exc), exc
        else:
            raise AssertionError("expected the absent package to be reported")
        print("ok test_the_missing_package_says_what_to_install (pymetis absent)")

    def test_partitioning_when_the_package_is_present():
        """Skipped without pymetis, and loudly, so it is not mistaken for a pass."""
        if not HAVE_METIS:
            print("SKIP test_partitioning_when_the_package_is_present "
                  "(pymetis is not installed; this call path is unexercised)")
            return
        p = call_metis_matrix(_matrix(12), 3)
        assert p.size == 12
        assert len(set(p.tolist())) == 3, p.tolist()
        print(f"ok test_partitioning_when_the_package_is_present ({p.tolist()})")

    test_the_graph_is_the_off_diagonal_pattern()
    test_weights_are_positive_integers()
    test_the_weakest_connection_never_rounds_to_zero()
    test_weights_are_ordered_by_connection_strength()
    test_the_graph_is_symmetric()
    test_it_works_on_a_real_tpfa_matrix()
    test_a_matrix_with_no_connections_is_rejected()
    test_one_block_is_rejected()
    test_the_missing_package_says_what_to_install()
    test_partitioning_when_the_package_is_present()
    print("\nAll tests passed!")
