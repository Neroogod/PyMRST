"""Basic Grid Tutorial: Rectilinear and Curvilinear Grids.

Port of MRST ``examples/grids/gridTutorialStruct.m``.

A graded tensor grid, refined towards the centre in `x` and towards the bottom
in `y`, is built and then progressively worked on:

1. ``tensor_grid`` from two coordinate vectors, giving a rectilinear grid whose
   spacing is deliberately non-uniform;
2. ``compute_geometry``, which adds the volumes and centroids the raw structure
   does not carry;
3. selecting cells by radius and cutting them out with ``extract_subgrid``,
   which keeps an ``indexMap`` back to the parent;
4. lognormal permeability from ``log_norm_layers``, sampled through that map;
5. perturbing interior nodes to turn the rectilinear grid curvilinear, leaving
   the boundary untouched so the domain keeps its shape.

Step 5 is the one worth reading closely. MRST finds the boundary faces, marks
their nodes, and moves only the nodes that are left. Perturbing a boundary node
would change the domain rather than the grid inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.grid_structure import extract_subgrid
from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.params.rock.log_norm_layers import log_norm_layers
from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.utils.mcolon import mcolon
from pymrst.utils.units import DARCY, MILLI


@dataclass
class StructuredGridTutorial:
    """The parent grid, the extracted subgrid and its permeability."""

    G: object
    G_sub: object
    perm: np.ndarray
    radius: np.ndarray
    selected: np.ndarray


def graded_coordinates():
    """MRST's two coordinate vectors, graded rather than uniform.

    `x` is refined towards the middle through a cosine-weighted cumulative
    sum; `y` is refined towards zero by raising a uniform vector to the power
    0.75. Uniform spacing would make several later checks vacuous, since any
    indexing error would still give plausible volumes.
    """
    dx = 1.0 - 0.5 * np.cos(np.arange(-1.0, 1.0 + 1e-12, 0.1) * np.pi)
    x = -1.15 + 0.1 * np.cumsum(dx)
    y = np.arange(0.0, 1.0 + 1e-12, 0.05) ** 0.75
    return x, y


def build_tensor_grid():
    """Steps 1 and 2: the graded tensor grid, with geometry computed."""
    x, y = graded_coordinates()
    return compute_geometry(tensor_grid(x, y))


def select_by_radius(g, threshold: float = 0.6):
    """Cells whose centroid lies outside a circle of the given radius."""
    r = np.sqrt(np.sum(np.asarray(g.cells.centroids, dtype=float) ** 2, axis=1))
    return r, r > threshold


def perturb_interior_nodes(g, *, scale: float = 0.007, seed: int = 0):
    """Move interior nodes only, leaving the boundary in place.

    Returns a new grid; the input is untouched. Boundary nodes are found
    through the boundary faces, exactly as MRST does, because moving one would
    deform the domain rather than the grid within it.
    """
    rng = np.random.default_rng(seed)
    out = compute_geometry(_copy_grid(g))

    interior = np.ones(out.nodes.num, dtype=bool)
    bf = np.asarray(boundary_faces(out), dtype=np.int64)
    node_pos = np.asarray(out.faces.nodePos, dtype=np.int64)
    # MRST writes `mcolon(nodePos(fI), nodePos(fI) + 1)`, whose `mcolon` is
    # inclusive, so `+1` picks up both endpoints of a 2-D face. PyMRST's
    # `mcolon` is half-open, `[lo, hi)`, and translating the `+1` literally
    # marks only the first node of each boundary face - leaving half the
    # boundary free to move, which shifts the domain outline while still
    # looking like a perturbed grid. Using `nodePos[bf + 1]` is both correct
    # here and independent of how many nodes a face has, so it works in 3-D
    # where MRST's `+1` would not.
    idx = mcolon(node_pos[bf], node_pos[bf + 1])
    interior[np.asarray(out.faces.nodes, dtype=np.int64)[idx]] = False

    coords = np.asarray(out.nodes.coords, dtype=float).copy()
    coords[interior] += scale * rng.standard_normal((int(interior.sum()),
                                                     coords.shape[1]))
    out.nodes.coords = coords
    return compute_geometry(out), interior


def _copy_grid(g):
    import copy
    return copy.deepcopy(g)


def run_grid_tutorial_struct(*, threshold: float = 0.6,
                             seed: int = 0) -> StructuredGridTutorial:
    """Build the grid, extract the subgrid, and sample permeability onto it."""
    g = build_tensor_grid()
    r, selected = select_by_radius(g, threshold)
    g_sub = extract_subgrid(g, np.flatnonzero(selected))

    dims = list(np.asarray(g.cartDims, dtype=int)) + [1]
    k = log_norm_layers(dims, [1.0])
    perm = np.asarray(k, dtype=float).ravel(order="F")[
        np.asarray(g_sub.cells.indexMap, dtype=np.int64)] * MILLI * DARCY
    return StructuredGridTutorial(G=g, G_sub=g_sub, perm=perm,
                                  radius=r, selected=selected)


if __name__ == "__main__":
    def test_tensor_grid_dimensions():
        x, y = graded_coordinates()
        g = build_tensor_grid()
        assert g.cartDims is not None
        assert tuple(g.cartDims) == (x.size - 1, y.size - 1), g.cartDims
        assert g.cells.num == (x.size - 1) * (y.size - 1), g.cells.num
        assert g.griddim == 2, g.griddim
        print("ok test_tensor_grid_dimensions")

    def test_grading_produces_unequal_cells():
        """The point of the graded coordinates: spacing genuinely varies.

        If this ever became uniform the later checks would still pass while
        testing much less, so the premise is asserted rather than assumed.
        """
        g = build_tensor_grid()
        vol = np.asarray(g.cells.volumes, dtype=float)
        assert vol.max() / vol.min() > 5.0, vol.max() / vol.min()
        assert np.all(vol > 0.0)
        print(f"ok test_grading_produces_unequal_cells "
              f"(volume ratio {vol.max() / vol.min():.1f})")

    def test_compute_geometry_adds_volumes_and_centroids():
        x, y = graded_coordinates()
        raw = tensor_grid(x, y)
        assert getattr(raw.cells, "volumes", None) is None
        g = compute_geometry(raw)
        assert np.asarray(g.cells.volumes).shape == (g.cells.num,)
        assert np.asarray(g.cells.centroids).shape == (g.cells.num, 2)
        # Total volume is the bounding rectangle, since the grid is a tensor product.
        assert np.isclose(np.sum(g.cells.volumes),
                          (x[-1] - x[0]) * (y[-1] - y[0]), rtol=1e-12)
        print("ok test_compute_geometry_adds_volumes_and_centroids")

    def test_subgrid_keeps_a_map_back_to_the_parent():
        """`extract_subgrid` is only useful if `indexMap` is right.

        Every later lookup, permeability included, goes through it, so a wrong
        map silently pairs each subgrid cell with another cell's properties.
        """
        out = run_grid_tutorial_struct()
        keep = np.flatnonzero(out.selected)
        assert out.G_sub.cells.num == keep.size, out.G_sub.cells.num
        assert np.array_equal(np.asarray(out.G_sub.cells.indexMap), keep)
        assert np.allclose(np.asarray(out.G_sub.cells.volumes),
                           np.asarray(out.G.cells.volumes)[keep])
        print("ok test_subgrid_keeps_a_map_back_to_the_parent")

    def test_radius_selection_is_what_it_claims():
        out = run_grid_tutorial_struct(threshold=0.6)
        c = np.asarray(out.G.cells.centroids, dtype=float)
        r = np.sqrt(np.sum(c ** 2, axis=1))
        assert np.array_equal(out.selected, r > 0.6)
        assert 0 < out.selected.sum() < out.G.cells.num, out.selected.sum()
        print(f"ok test_radius_selection_is_what_it_claims "
              f"({out.selected.sum()} of {out.G.cells.num} cells)")

    def test_permeability_is_positive_and_sized_to_the_subgrid():
        out = run_grid_tutorial_struct()
        assert out.perm.shape == (out.G_sub.cells.num,), out.perm.shape
        assert np.all(out.perm > 0.0)
        print("ok test_permeability_is_positive_and_sized_to_the_subgrid")

    def test_perturbation_leaves_the_boundary_alone():
        """The check the picture cannot make: the domain outline is unchanged.

        Interior nodes move, boundary nodes do not, so the bounding box of the
        perturbed grid must equal the original exactly.
        """
        out = run_grid_tutorial_struct()
        moved, interior = perturb_interior_nodes(out.G_sub, seed=3)
        a = np.asarray(out.G_sub.nodes.coords, dtype=float)
        b = np.asarray(moved.nodes.coords, dtype=float)
        assert np.allclose(a[~interior], b[~interior]), "a boundary node moved"
        assert not np.allclose(a[interior], b[interior]), "no interior node moved"
        assert np.allclose(a.min(axis=0), b.min(axis=0)), "bounding box changed"
        assert np.allclose(a.max(axis=0), b.max(axis=0)), "bounding box changed"
        print(f"ok test_perturbation_leaves_the_boundary_alone "
              f"({interior.sum()} of {interior.size} nodes moved)")

    def test_perturbation_keeps_cells_valid():
        """A perturbed grid is only useful if every cell still has positive volume."""
        out = run_grid_tutorial_struct()
        moved, _ = perturb_interior_nodes(out.G_sub, seed=3)
        vol = np.asarray(moved.cells.volumes, dtype=float)
        assert np.all(vol > 0.0), vol.min()
        assert np.isclose(vol.sum(), np.sum(out.G_sub.cells.volumes), rtol=5e-2)
        print("ok test_perturbation_keeps_cells_valid")

    test_tensor_grid_dimensions()
    test_grading_produces_unequal_cells()
    test_compute_geometry_adds_volumes_and_centroids()
    test_subgrid_keeps_a_map_back_to_the_parent()
    test_radius_selection_is_what_it_claims()
    test_permeability_is_positive_and_sized_to_the_subgrid()
    test_perturbation_leaves_the_boundary_alone()
    test_perturbation_keeps_cells_valid()
    print("\nAll tests passed!")
