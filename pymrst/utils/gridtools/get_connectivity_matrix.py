"""Adjacency matrix from a neighbourship list (MRST ``getConnectivityMatrix.m``).

Turns the `(n, 2)` list `get_neighbourship` returns into a sparse symmetric
matrix: `A[i, j]` is 1 when cells `i` and `j` are connected. Used to find
connected components, to build coarse partitions, and as the sparsity pattern
of a cell-centred discretisation.

`include_diagonal` adds the self-connections. Whether they belong depends on
what the matrix is for: a *sparsity pattern* needs them, because a cell's
equation involves its own unknown; a pure *adjacency* does not, and including
them makes every cell its own neighbour.

`n_rows` sizes the matrix when the connection list does not reach the last
cell - an isolated cell appears in no pair, so inferring the size from the list
would silently drop it.
"""

import numpy as np
import scipy.sparse as sp


def get_connectivity_matrix(neighbours, include_diagonal: bool = False,
                            n_rows: int = None) -> sp.csr_array:
    """Sparse symmetric adjacency of the connections in ``neighbours``."""
    neighbours = np.atleast_2d(np.asarray(neighbours, dtype=np.int64))
    if neighbours.shape[1] != 2:
        raise ValueError(f"neighbourship must be n-by-2, "
                         f"got {neighbours.shape}")
    if neighbours.size and neighbours.min() < 0:
        raise ValueError(
            "the neighbourship contains boundary rows; drop them before "
            "building a connectivity matrix")

    rows = np.concatenate([neighbours[:, 0], neighbours[:, 1]])
    columns = np.concatenate([neighbours[:, 1], neighbours[:, 0]])

    size = int(n_rows if n_rows is not None
               else (rows.max() + 1 if rows.size else 0))
    if include_diagonal:
        diagonal = np.arange(size)
        rows = np.concatenate([rows, diagonal])
        columns = np.concatenate([columns, diagonal])

    values = np.ones(rows.size, dtype=float)
    matrix = sp.coo_array((values, (rows, columns)), shape=(size, size))
    return matrix.tocsr()


getConnectivityMatrix = get_connectivity_matrix


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.utils.gridtools.get_neighbourship import get_neighbourship

    G = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    N = get_neighbourship(G)

    def test_it_is_symmetric() -> None:
        A = get_connectivity_matrix(N, n_rows=G.cells.num)
        assert (A != A.T).nnz == 0
        print("ok test_it_is_symmetric")

    def test_connected_cells_are_marked() -> None:
        A = get_connectivity_matrix(N, n_rows=G.cells.num).toarray()
        for i, j in N:
            assert A[i, j] > 0 and A[j, i] > 0, (i, j)
        print("ok test_connected_cells_are_marked")

    def test_the_diagonal_is_empty_by_default() -> None:
        A = get_connectivity_matrix(N, n_rows=G.cells.num)
        assert np.all(A.diagonal() == 0)
        print("ok test_the_diagonal_is_empty_by_default")

    def test_the_diagonal_can_be_included() -> None:
        A = get_connectivity_matrix(N, include_diagonal=True,
                                    n_rows=G.cells.num)
        assert np.all(A.diagonal() > 0)
        print("ok test_the_diagonal_can_be_included")

    def test_the_row_count_sizes_the_matrix() -> None:
        A = get_connectivity_matrix(N, n_rows=G.cells.num)
        assert A.shape == (G.cells.num, G.cells.num), A.shape
        print("ok test_the_row_count_sizes_the_matrix")

    def test_an_isolated_cell_survives_when_the_size_is_given() -> None:
        """Inferring the size from the list would drop it."""
        pairs = np.array([[0, 1]])
        inferred = get_connectivity_matrix(pairs)
        stated = get_connectivity_matrix(pairs, n_rows=5)
        assert inferred.shape == (2, 2), inferred.shape
        assert stated.shape == (5, 5), stated.shape
        print("ok test_an_isolated_cell_survives_when_the_size_is_given")

    def test_the_degree_is_the_number_of_neighbours() -> None:
        A = get_connectivity_matrix(N, n_rows=G.cells.num)
        degree = np.asarray(A.sum(axis=1)).ravel()
        # a 3x2 grid: corner cells have 2 neighbours, middle ones 3
        assert sorted(degree.tolist()) == [2, 2, 2, 2, 3, 3], degree
        print("ok test_the_degree_is_the_number_of_neighbours")

    def test_a_boundary_row_is_rejected() -> None:
        """-1 marks outside and is not a matrix index."""
        try:
            get_connectivity_matrix(get_neighbourship(G, include_boundary=True))
        except ValueError as error:
            assert "boundary rows" in str(error)
            print("ok test_a_boundary_row_is_rejected")
            return
        raise AssertionError("boundary rows must be rejected")

    def test_a_malformed_list_is_rejected() -> None:
        try:
            get_connectivity_matrix(np.zeros((3, 3), dtype=int))
        except ValueError as error:
            assert "n-by-2" in str(error)
            print("ok test_a_malformed_list_is_rejected")
            return
        raise AssertionError("a 3-column list must be rejected")

    test_it_is_symmetric()
    test_connected_cells_are_marked()
    test_the_diagonal_is_empty_by_default()
    test_the_diagonal_can_be_included()
    test_the_row_count_sizes_the_matrix()
    test_an_isolated_cell_survives_when_the_size_is_given()
    test_the_degree_is_the_number_of_neighbours()
    test_a_boundary_row_is_rejected()
    test_a_malformed_list_is_rejected()
    print("\nAll tests passed!")
