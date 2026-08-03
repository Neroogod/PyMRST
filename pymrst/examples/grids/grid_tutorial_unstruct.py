"""Basic Grid Operations and Manipulations: Unstructured Grids.

Port of MRST ``examples/grids/gridTutorialUnstruct.m``.

Four constructions, each building on the previous:

1. a triangulated grid from a perturbed point cloud, and its Voronoi dual
   through ``pebi``;
2. a Delaunay triangulation of scattered data;
3. extrusion of a 2-D triangulation into 3-D prisms with ``make_layered_grid``;
4. radially graded point sets, refined around one or two well locations, meshed
   as PEBI grids.

The triangulation and its dual are the pair worth understanding. `triangle_grid`
gives cells that are triangles with the input points as vertices; `pebi` returns
the Voronoi diagram of the same points, so the input points become cell centres
rather than corners. Both are unstructured, so neither carries `cartDims` and
neither has a logical index space.

**One deviation from MRST.** Its second section loads `seamount`, sample data
shipped with MATLAB rather than with MRST, which has no Python equivalent.
``scattered_points`` generates a comparable irregular cloud instead. The
construction being demonstrated, a Delaunay triangulation of scattered data, is
unchanged; only the coordinates differ, so the tests here check structural
properties rather than specific values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.make_layered_grid import make_layered_grid
from pymrst.gridprocessing.pebi import pebi
from pymrst.gridprocessing.triangle_grid import triangle_grid


@dataclass
class UnstructuredGrids:
    """The grids MRST's tutorial builds, in the order it builds them."""

    triangulated: object
    voronoi: object
    scattered: object
    layered: object
    radial: object


def perturbed_lattice(nx: int = 11, ny: int = 9, *, scale: float = 0.25,
                      seed: int = 0) -> np.ndarray:
    """A lattice with its interior points jittered, MRST's starting cloud."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    x[1:-1, 1:-1] += scale * rng.standard_normal(x[1:-1, 1:-1].shape)
    y[1:-1, 1:-1] += scale * rng.standard_normal(y[1:-1, 1:-1].shape)
    return np.column_stack([x.ravel(order="F"), y.ravel(order="F")])


def scattered_points(n: int = 200, *, seed: int = 1) -> np.ndarray:
    """Irregular scattered data, standing in for MATLAB's `seamount`.

    `seamount` ships with MATLAB, not with MRST, so there is nothing to port.
    What the section demonstrates is triangulating scattered data, which any
    irregular cloud shows equally well.
    """
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n))
    theta = 2.0 * np.pi * rng.random(n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def radially_graded_points(centres=((2.0, 2.0), (12.0, 6.0)),
                           *, background=(0.0, 14.0, 0.0, 8.0),
                           spacing: float = 0.5, n_theta: int = 28) -> np.ndarray:
    """Points refined in rings around each centre, over a coarse background.

    This is how MRST grades a mesh towards a well: rings of radius
    `exp(-3.5 .. 0)` crowd together near the centre, so cells shrink where the
    pressure gradient is steepest.
    """
    pts = []
    for cx, cy in centres:
        for r in np.exp(np.arange(-3.5, 0.0 + 1e-12, 0.2)):
            t = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
            pts.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]))
        pts.append(np.array([[cx, cy]]))
    x0, x1, y0, y1 = background
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-12, spacing),
                         np.arange(y0, y1 + 1e-12, spacing))
    pts.append(np.column_stack([gx.ravel(), gy.ravel()]))
    return np.unique(np.vstack(pts), axis=0)


def run_grid_tutorial_unstruct(*, seed: int = 0) -> UnstructuredGrids:
    """Build every grid the tutorial demonstrates."""
    p = perturbed_lattice(seed=seed)
    tri = compute_geometry(triangle_grid(p))
    vor = compute_geometry(pebi(triangle_grid(p)))

    sp = scattered_points()
    scattered = compute_geometry(triangle_grid(sp, Delaunay(sp).simplices))

    layered = compute_geometry(make_layered_grid(triangle_grid(sp, Delaunay(sp).simplices), 5))

    rp = radially_graded_points()
    radial = compute_geometry(pebi(triangle_grid(rp)))
    return UnstructuredGrids(triangulated=tri, voronoi=vor, scattered=scattered,
                             layered=layered, radial=radial)


if __name__ == "__main__":
    def test_triangulation_has_triangular_cells():
        p = perturbed_lattice()
        g = compute_geometry(triangle_grid(p))
        counts = np.diff(np.asarray(g.cells.facePos, dtype=np.int64))
        assert np.all(counts == 3), np.unique(counts)
        assert g.griddim == 2, g.griddim
        assert np.all(np.asarray(g.cells.volumes) > 0.0)
        print(f"ok test_triangulation_has_triangular_cells ({g.cells.num} triangles)")

    def test_pebi_is_the_dual_of_the_triangulation():
        """The Voronoi dual turns input points into cell centres, not corners.

        So the PEBI grid has one cell per input point, where the triangulation
        has one *node* per input point. Getting that backwards is the usual
        confusion between a triangulation and its dual.
        """
        p = perturbed_lattice()
        tri = triangle_grid(p)
        vor = pebi(triangle_grid(p))
        assert tri.nodes.num == p.shape[0], (tri.nodes.num, p.shape[0])
        assert vor.cells.num == p.shape[0], (vor.cells.num, p.shape[0])
        print(f"ok test_pebi_is_the_dual_of_the_triangulation "
              f"({tri.cells.num} triangles -> {vor.cells.num} Voronoi cells)")

    def test_unstructured_grids_carry_no_logical_dimensions():
        """`cartDims` belongs to grids with a Cartesian index space.

        An unstructured grid has none, which is why the MATLAB file bridge had
        to stop requiring the field.
        """
        out = run_grid_tutorial_unstruct()
        for name in ("triangulated", "voronoi", "scattered", "radial"):
            g = getattr(out, name)
            assert getattr(g, "cartDims", None) is None, (name, g.cartDims)
        print("ok test_unstructured_grids_carry_no_logical_dimensions")

    def test_delaunay_covers_the_point_set():
        sp = scattered_points()
        g = compute_geometry(triangle_grid(sp, Delaunay(sp).simplices))
        assert g.nodes.num == sp.shape[0], (g.nodes.num, sp.shape[0])
        # A Delaunay triangulation of a convex hull has no gaps, so the areas
        # sum to the hull area.
        from scipy.spatial import ConvexHull
        assert np.isclose(np.sum(g.cells.volumes), ConvexHull(sp).volume, rtol=1e-9)
        print("ok test_delaunay_covers_the_point_set")

    def test_extrusion_multiplies_cells_by_the_layer_count():
        """`make_layered_grid` turns a 2-D mesh into 3-D prisms."""
        sp = scattered_points()
        base = triangle_grid(sp, Delaunay(sp).simplices)
        n_layers = 5
        g = compute_geometry(make_layered_grid(base, n_layers))
        assert g.griddim == 3, g.griddim
        assert g.cells.num == base.cells.num * n_layers, (g.cells.num, base.cells.num)
        assert g.nodes.num == base.nodes.num * (n_layers + 1), g.nodes.num
        assert np.all(np.asarray(g.cells.volumes) > 0.0)
        print(f"ok test_extrusion_multiplies_cells_by_the_layer_count "
              f"({base.cells.num} x {n_layers} = {g.cells.num})")

    def test_radial_grading_refines_towards_the_centres():
        """Cells near a well centre must be smaller than background cells.

        That is the entire purpose of grading, and it is the one property a
        picture of the mesh conveys at a glance but never proves.
        """
        rp = radially_graded_points()
        g = compute_geometry(pebi(triangle_grid(rp)))
        c = np.asarray(g.cells.centroids, dtype=float)
        vol = np.asarray(g.cells.volumes, dtype=float)
        d = np.sqrt(((c - np.array([2.0, 2.0])) ** 2).sum(axis=1))
        near = vol[d < 0.4]
        far = vol[(d > 3.0) & (d < 6.0)]
        assert near.size and far.size, (near.size, far.size)
        assert near.mean() < far.mean(), (near.mean(), far.mean())
        print(f"ok test_radial_grading_refines_towards_the_centres "
              f"(near {near.mean():.4f} vs far {far.mean():.4f})")

    def test_all_grids_build_and_have_positive_volume():
        out = run_grid_tutorial_unstruct()
        for name in ("triangulated", "voronoi", "scattered", "layered", "radial"):
            g = getattr(out, name)
            vol = np.asarray(g.cells.volumes, dtype=float)
            assert g.cells.num > 0 and np.all(vol > 0.0), (name, vol.min())
        print("ok test_all_grids_build_and_have_positive_volume")

    test_triangulation_has_triangular_cells()
    test_pebi_is_the_dual_of_the_triangulation()
    test_unstructured_grids_carry_no_logical_dimensions()
    test_delaunay_covers_the_point_set()
    test_extrusion_multiplies_cells_by_the_layer_count()
    test_radial_grading_refines_towards_the_centres()
    test_all_grids_build_and_have_positive_volume()
    print("\nAll tests passed!")
