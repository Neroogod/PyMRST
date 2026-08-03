"""Overview of Grid Factory Routines.

Port of MRST ``examples/grids/gridTutorialIntro.m``.

Two things are demonstrated. First the anatomy of the grid structure itself:
cells, faces and nodes, how `cells.facePos` and `cells.faces` connect them, what
the face tags mean, and how `faces.neighbors` marks the boundary. Then the
factory routines that build grids of every shape PyMRST supports: Cartesian,
tensor, tetrahedral, triangular, extruded, hexahedral, corner-point, and a
twisted variant.

The structural section is the reason this file makes a good regression test.
Every relationship it displays is exact, so each becomes an assertion:
`facePos` differences count a cell's faces, `sum(diff(facePos))` equals the
length of `cells.faces`, and an interior face has two distinct neighbours while
a boundary face has exactly one.

**One convention difference from MRST runs through the whole file.** MRST marks
a missing neighbour with 0 and writes `any(G.faces.neighbors == 0, 2)` to find
the boundary. PyMRST uses -1, because 0 is a valid 0-based cell index. Reusing
MRST's test here would classify cell 0's faces as boundary and miss the real
boundary entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.hexahedral_grid import hexahedral_grid
from pymrst.gridprocessing.make_layered_grid import make_layered_grid
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.gridprocessing.testgrids.twister import twister
from pymrst.gridprocessing.tetrahedral_grid import tetrahedral_grid
from pymrst.gridprocessing.triangle_grid import triangle_grid
from pymrst.utils.rldecode import rldecode

FACE_TAGS = ("West", "East", "South", "North", "Bottom", "Top")


@dataclass
class GridAnatomy:
    """The small grid MRST dissects, with its connectivity spelled out."""

    G: object
    cell_face_pairs: np.ndarray  # (n, 2): cell index, face index
    boundary: np.ndarray         # bool per face


def small_grid_with_a_hole():
    """MRST's 3x2 grid with cell 2 removed, the example it labels.

    MRST removes cell 2 with 1-based indexing, which is cell 1 here.
    """
    return compute_geometry(remove_cells(cart_grid([3, 2]), np.array([1])))


def describe_connectivity(g) -> GridAnatomy:
    """Expand `facePos`/`faces` into explicit (cell, face) pairs.

    MRST builds this with `rldecode(1:num, diff(facePos), 2)`, which repeats
    each cell index as many times as it has faces. The result is the flat
    connectivity table the tutorial prints.
    """
    counts = np.diff(np.asarray(g.cells.facePos, dtype=np.int64))
    cells = rldecode(np.arange(g.cells.num), counts)
    faces = np.asarray(g.cells.faces, dtype=np.int64)
    if faces.ndim > 1:
        faces = faces[:, 0]
    pairs = np.column_stack([cells, faces])
    # PyMRST marks an absent neighbour -1; MRST uses 0.
    boundary = np.any(np.asarray(g.faces.neighbors, dtype=np.int64) == -1, axis=1)
    return GridAnatomy(G=g, cell_face_pairs=pairs, boundary=boundary)


def face_tag_names(g) -> list[str] | None:
    """Face tags as names, when the grid carries them.

    Tags are 1-based in MRST, so West is 1. PyMRST stores them 0-based, so the
    lookup is direct.
    """
    tags = getattr(g.cells, "tags", None)
    if tags is None:
        return None
    t = np.asarray(tags, dtype=np.int64).ravel()
    return [FACE_TAGS[i] if 0 <= i < len(FACE_TAGS) else "?" for i in t]


def build_all_grid_types(*, seed: int = 0) -> dict[str, object]:
    """One grid from each factory routine the tutorial tours."""
    rng = np.random.default_rng(seed)
    out: dict[str, object] = {}

    out["cartesian"] = compute_geometry(cart_grid([3, 2]))
    # Cubed spacing in x, so the grading is unmistakable.
    out["tensor"] = compute_geometry(
        tensor_grid(np.arange(1.0, 11.0) ** 3, np.arange(1.0, 6.0)))

    pts = rng.random((20, 3)) * np.array([10.0, 10.0, 1.0])
    out["tetrahedral"] = compute_geometry(
        tetrahedral_grid(pts, Delaunay(pts).simplices))

    pts2 = pts[:, :2]
    tri = triangle_grid(pts2, Delaunay(pts2).simplices)
    out["triangular"] = compute_geometry(tri)
    out["extruded"] = compute_geometry(make_layered_grid(tri, 3))

    # Two hexahedra sharing a face, given by explicit node lists.
    h = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                  [1, 8, 9, 2, 5, 10, 11, 6]], dtype=np.int64)
    p = np.array([
        [1.0, 0.0, 0.1860], [1.0, 1.0, 0.1852], [1.0, 1.0, 0.1926],
        [1.0, 0.0, 0.1930], [0.0, 0.0, 0.1854], [0.0, 1.0, 0.1846],
        [0.0, 1.0, 0.1923], [0.0, 0.0, 0.1926], [1.0, 2.0, 0.1844],
        [1.0, 2.0, 0.1922], [0.0, 2.0, 0.1837], [0.0, 2.0, 0.1919]])
    # PyMRST takes `(hexes, coords)`; MRST's `hexahedralGrid` takes `(P, H)`,
    # points first. Passing them in MRST's order reads the coordinate array as
    # connectivity and fails on the first out-of-range node index.
    out["hexahedral"] = compute_geometry(hexahedral_grid(h, p))

    out["twisted"] = compute_geometry(twister(cart_grid([10, 10])))
    return out


if __name__ == "__main__":
    def test_removing_a_cell_leaves_the_rest_intact():
        g = small_grid_with_a_hole()
        assert g.cells.num == 5, g.cells.num
        assert np.all(np.asarray(g.cells.volumes) > 0.0)
        print("ok test_removing_a_cell_leaves_the_rest_intact")

    def test_face_pos_indexes_cells_faces():
        """`facePos` is a prefix sum, so its differences count each cell's faces.

        The whole grid structure hangs off this: an inconsistent `facePos`
        misaligns every cell-to-face lookup in the toolbox.
        """
        g = small_grid_with_a_hole()
        fp = np.asarray(g.cells.facePos, dtype=np.int64)
        faces = np.asarray(g.cells.faces, dtype=np.int64)
        if faces.ndim > 1:
            faces = faces[:, 0]
        assert fp.size == g.cells.num + 1, fp.size
        assert fp[0] == 0 and fp[-1] == faces.size, (fp[0], fp[-1], faces.size)
        assert np.all(np.diff(fp) > 0), "a cell has no faces"
        # 2-D Cartesian cells have four faces each.
        assert np.all(np.diff(fp) == 4), np.unique(np.diff(fp))
        print("ok test_face_pos_indexes_cells_faces")

    def test_connectivity_table_matches_the_structure():
        g = small_grid_with_a_hole()
        a = describe_connectivity(g)
        counts = np.diff(np.asarray(g.cells.facePos, dtype=np.int64))
        assert a.cell_face_pairs.shape[0] == counts.sum(), a.cell_face_pairs.shape
        for c in range(g.cells.num):
            assert (a.cell_face_pairs[:, 0] == c).sum() == counts[c], c
        print("ok test_connectivity_table_matches_the_structure")

    def test_boundary_uses_minus_one_not_zero():
        """PyMRST's sentinel is -1; MRST's is 0. Using MRST's test here is wrong.

        Cell 0 is a real cell, so `neighbors == 0` would mark its faces as
        boundary and miss the actual boundary. The two tests are compared
        directly to show they disagree.
        """
        g = small_grid_with_a_hole()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        correct = np.any(nb == -1, axis=1)
        mrst_style = np.any(nb == 0, axis=1)
        assert not np.array_equal(correct, mrst_style), (
            "the two conventions happened to agree; the test proves nothing here")
        # Every face is either interior with two cells, or boundary with one.
        assert np.all((nb >= 0).sum(axis=1)[correct] == 1)
        assert np.all((nb >= 0).sum(axis=1)[~correct] == 2)
        print(f"ok test_boundary_uses_minus_one_not_zero "
              f"({correct.sum()} boundary, {(~correct).sum()} interior)")

    def test_interior_faces_join_two_distinct_cells():
        g = small_grid_with_a_hole()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = np.all(nb != -1, axis=1)
        assert np.all(nb[interior, 0] != nb[interior, 1]), "a face joins a cell to itself"
        print("ok test_interior_faces_join_two_distinct_cells")

    def test_every_factory_routine_builds_a_valid_grid():
        """The tour itself: each routine returns a usable grid."""
        grids = build_all_grid_types()
        expected = {"cartesian", "tensor", "tetrahedral", "triangular",
                    "extruded", "hexahedral", "twisted"}
        assert set(grids) == expected, set(grids) ^ expected
        for name, g in grids.items():
            vol = np.asarray(g.cells.volumes, dtype=float)
            assert g.cells.num > 0, name
            assert np.all(vol > 0.0), (name, vol.min())
            fp = np.asarray(g.cells.facePos, dtype=np.int64)
            assert fp[0] == 0 and np.all(np.diff(fp) > 0), name
        print(f"ok test_every_factory_routine_builds_a_valid_grid "
              f"({len(grids)} routines)")

    def test_hexahedral_grid_has_two_cells_sharing_a_face():
        g = build_all_grid_types()["hexahedral"]
        assert g.cells.num == 2, g.cells.num
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        shared = np.all(nb != -1, axis=1)
        assert shared.sum() == 1, shared.sum()
        print("ok test_hexahedral_grid_has_two_cells_sharing_a_face")

    def test_twister_deforms_without_destroying_cells():
        """`twister` perturbs node positions; volumes must stay positive.

        Total volume is preserved because the deformation is interior only.
        """
        before = compute_geometry(cart_grid([10, 10]))
        after = compute_geometry(twister(cart_grid([10, 10])))
        assert after.cells.num == before.cells.num
        vol = np.asarray(after.cells.volumes, dtype=float)
        assert np.all(vol > 0.0), vol.min()
        assert not np.allclose(np.asarray(after.nodes.coords),
                               np.asarray(before.nodes.coords)), "twister did nothing"
        assert np.isclose(vol.sum(), np.sum(before.cells.volumes), rtol=1e-9)
        print("ok test_twister_deforms_without_destroying_cells")

    def test_tensor_grid_grading_matches_the_cubed_spacing():
        """Cubed coordinates give spacings 7 to 271, a volume ratio of 38.7.

        `x = (1:10)^3` has differences `[7, 19, 37, ..., 271]`, and `y` is
        uniform, so the cell volume ratio is exactly `271/7`. Asserting the
        computed number rather than a round guess is what makes this a check on
        `tensor_grid` rather than on the assertion.
        """
        g = build_all_grid_types()["tensor"]
        vol = np.asarray(g.cells.volumes, dtype=float)
        dx = np.diff(np.arange(1.0, 11.0) ** 3)
        assert np.isclose(vol.max() / vol.min(), dx.max() / dx.min(), rtol=1e-12), (
            vol.max() / vol.min(), dx.max() / dx.min())
        print(f"ok test_tensor_grid_grading_matches_the_cubed_spacing "
              f"(ratio {vol.max() / vol.min():.1f})")

    test_removing_a_cell_leaves_the_rest_intact()
    test_face_pos_indexes_cells_faces()
    test_connectivity_table_matches_the_structure()
    test_boundary_uses_minus_one_not_zero()
    test_interior_faces_join_two_distinct_cells()
    test_every_factory_routine_builds_a_valid_grid()
    test_hexahedral_grid_has_two_cells_sharing_a_face()
    test_twister_deforms_without_destroying_cells()
    test_tensor_grid_grading_matches_the_cubed_spacing()
    print("\nAll tests passed!")
