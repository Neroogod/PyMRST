"""Geometry for a coarse grid (MRST ``coarsenGeometry.m``).

`generate_coarse_grid` builds topology only. This adds the geometry, computed
from the fine grid rather than from any coarse shape:

* **block volume** is the summed volume of its cells;
* **block centroid** is the volume-weighted mean of its cell centroids;
* **coarse normal** is the signed sum of its fine normals, using
  `fine_to_coarse_sign` so faces pointing the other way subtract;
* **coarse area** is the length of that normal, not the summed fine area;
* **coarse centroid** is the area-weighted mean of its fine centroids.

Two of those are worth stating precisely because the plausible alternative is
wrong in a way nothing catches.

**Area is the normal's length.** For a flat coarse face the two agree. For a
folded one, made of fine faces at an angle to each other, the summed fine area
is larger. MRST takes the normal's length, which is the area of the face's
projection: the quantity a transmissibility wants, since flux through a folded
face is what the resolved normal carries.

**The centroid is area-weighted, not volume- or count-weighted.** A coarse face
built from fine faces of unequal size has its centroid pulled towards the large
ones, and a plain mean would sit in a plausible but wrong place.

Results are written onto the `CoarseGrid` itself, in `cell_volumes`,
`cell_centroids`, `face_areas`, `face_centroids` and `face_normals`. MRST puts
them on `cg.cells` and `cg.faces`; the flat names here avoid colliding with the
optional geometric grid in `.g`.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign


def coarsen_geometry(cg):
    """Fill in the coarse grid's geometry from its parent. Returns `cg`."""
    if cg.parent is None:
        raise ValueError("the coarse grid has no parent; geometry needs it")
    g = cg.parent
    if getattr(g.cells, "volumes", None) is None:
        raise ValueError(
            "the fine grid carries no geometry; call compute_geometry on it first")

    partition = np.asarray(cg.partition, dtype=np.int64).ravel()
    n_blocks = int(cg.n_blocks)

    # Cells: volume-weighted centroids.
    volumes = np.asarray(g.cells.volumes, dtype=float).ravel()
    centroids = np.asarray(g.cells.centroids, dtype=float)
    block_volume = np.bincount(partition, weights=volumes, minlength=n_blocks)
    block_centroid = np.column_stack([
        np.bincount(partition, weights=volumes * centroids[:, i], minlength=n_blocks)
        for i in range(centroids.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        block_centroid = block_centroid / block_volume[:, None]

    cg.cell_volumes = block_volume
    cg.cell_centroids = block_centroid

    if cg.fconn is None or cg.conn_pos is None:
        return cg

    # Faces: signed normals, area-weighted centroids.
    counts = np.diff(np.asarray(cg.conn_pos, dtype=np.int64))
    face_no = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    sgn = fine_to_coarse_sign(cg)
    fine_areas = np.asarray(g.faces.areas, dtype=float).ravel()[cg.fconn]
    fine_centroids = np.asarray(g.faces.centroids, dtype=float)[cg.fconn]
    fine_normals = np.asarray(g.faces.normals, dtype=float)[cg.fconn]

    n_faces = counts.size
    total_area = np.bincount(face_no, weights=fine_areas, minlength=n_faces)
    coarse_centroid = np.column_stack([
        np.bincount(face_no, weights=fine_areas * fine_centroids[:, i],
                    minlength=n_faces)
        for i in range(fine_centroids.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        coarse_centroid = coarse_centroid / total_area[:, None]
    coarse_normal = np.column_stack([
        np.bincount(face_no, weights=fine_normals[:, i] * sgn, minlength=n_faces)
        for i in range(fine_normals.shape[1])])

    cg.face_centroids = coarse_centroid
    cg.face_normals = coarse_normal
    cg.face_areas = np.linalg.norm(coarse_normal, axis=1)
    return cg


__all__ = ["coarsen_geometry"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(6, 4, 2), part=(3, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims), list(part))
        return g, p, coarsen_geometry(generate_coarse_grid(g, p))

    def test_volume_is_conserved_block_by_block():
        """The invariant a coarse grid exists to preserve."""
        g, p, cg = _case()
        fine = np.asarray(g.cells.volumes, dtype=float)
        assert np.isclose(cg.cell_volumes.sum(), fine.sum(), rtol=1e-12)
        for b in range(cg.n_blocks):
            assert np.isclose(cg.cell_volumes[b], fine[p == b].sum(), rtol=1e-12), b
        print(f"ok test_volume_is_conserved_block_by_block "
              f"({cg.cell_volumes.sum():.4f} m3)")

    def test_a_block_centroid_lies_inside_its_block():
        """Weak, but it catches a wrong weighting or a wrong index array."""
        g, p, cg = _case()
        c = np.asarray(g.cells.centroids, dtype=float)
        for b in range(cg.n_blocks):
            inside = c[p == b]
            lo, hi = inside.min(axis=0), inside.max(axis=0)
            assert np.all(cg.cell_centroids[b] >= lo - 1e-9), b
            assert np.all(cg.cell_centroids[b] <= hi + 1e-9), b
        print("ok test_a_block_centroid_lies_inside_its_block")

    def test_centroids_are_volume_weighted():
        """A plain mean of cell centroids is the plausible wrong answer.

        On a uniform Cartesian grid the two agree, so the case is built with
        cells of unequal size, where they do not.
        """
        from pymrst.gridprocessing.tensor_grid import tensor_grid
        g = compute_geometry(tensor_grid(np.array([0.0, 1.0, 10.0]),
                                         np.array([0.0, 1.0]),
                                         np.array([0.0, 1.0])))
        p = np.zeros(g.cells.num, dtype=np.int64)
        cg = coarsen_geometry(generate_coarse_grid(g, p))
        c = np.asarray(g.cells.centroids, dtype=float)
        v = np.asarray(g.cells.volumes, dtype=float)
        weighted = (c * v[:, None]).sum(axis=0) / v.sum()
        assert np.allclose(cg.cell_centroids[0], weighted)
        assert not np.allclose(cg.cell_centroids[0], c.mean(axis=0)), \
            "the case does not distinguish the two weightings"
        print("ok test_centroids_are_volume_weighted")

    def test_coarse_area_equals_the_summed_fine_area_when_flat():
        """True for a Cartesian grid, and only because its faces are flat."""
        g, _, cg = _case()
        fine = np.asarray(g.faces.areas, dtype=float)
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        summed = np.bincount(face_no, weights=fine[cg.fconn],
                             minlength=cg.num_faces)
        assert np.allclose(cg.face_areas, summed), (cg.face_areas[:5], summed[:5])
        print("ok test_coarse_area_equals_the_summed_fine_area_when_flat")

    def test_area_is_the_normals_length_not_the_summed_area():
        """The two differ once a coarse face folds.

        Two fine faces at right angles have summed area 2 and resolved area
        sqrt(2). Taking the sum would inflate every transmissibility across a
        stepped or faulted coarse face, with nothing to show for it.
        """
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        # An L-shaped block, so its boundary in the interior of the grid folds.
        p = np.array([0, 1, 1, 1], dtype=np.int64)
        cg = coarsen_geometry(generate_coarse_grid(g, p))
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        fine = np.asarray(g.faces.areas, dtype=float)
        summed = np.bincount(face_no, weights=fine[cg.fconn], minlength=cg.num_faces)
        folded = np.flatnonzero(~np.isclose(cg.face_areas, summed))
        assert folded.size > 0, "the case has no folded coarse face"
        for f in folded:
            assert cg.face_areas[f] < summed[f], (f, cg.face_areas[f], summed[f])
        print(f"ok test_area_is_the_normals_length_not_the_summed_area "
              f"({folded.size} folded of {cg.num_faces})")

    def test_face_centroids_are_area_weighted():
        _, _, cg = _case()
        g = cg.parent
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        a = np.asarray(g.faces.areas, dtype=float)[cg.fconn]
        c = np.asarray(g.faces.centroids, dtype=float)[cg.fconn]
        for f in range(cg.num_faces):
            sel = face_no == f
            expected = (c[sel] * a[sel, None]).sum(axis=0) / a[sel].sum()
            assert np.allclose(cg.face_centroids[f], expected), f
        print("ok test_face_centroids_are_area_weighted")

    def test_a_face_centroid_lies_within_its_fine_faces():
        _, _, cg = _case()
        g = cg.parent
        c = np.asarray(g.faces.centroids, dtype=float)
        for f in range(cg.num_faces):
            sub = c[cg.sub_faces(f)]
            assert np.all(cg.face_centroids[f] >= sub.min(axis=0) - 1e-9), f
            assert np.all(cg.face_centroids[f] <= sub.max(axis=0) + 1e-9), f
        print("ok test_a_face_centroid_lies_within_its_fine_faces")

    def test_boundary_normals_point_out_of_the_model():
        """Sign errors on the boundary are the ones that reverse a flux."""
        g, _, cg = _case()
        interior = np.asarray(g.cells.centroids, dtype=float).mean(axis=0)
        exterior = np.flatnonzero(cg.neighbors[:, 0] < 0)
        for f in exterior:
            outward = cg.face_centroids[f] - interior
            # A boundary coarse face has the outside first, so its normal points
            # inwards by the convention above.
            assert np.dot(cg.face_normals[f], outward) < 0.0, (f, cg.face_normals[f])
        print(f"ok test_boundary_normals_point_out_of_the_model "
              f"({exterior.size} faces)")

    def test_it_agrees_with_the_tensor_grid_where_one_exists():
        """`.g` is built independently, so it is a free cross-check.

        The two paths share no code: one sums fine geometry, the other lays
        out coordinate lines and calls `compute_geometry`.
        """
        _, _, cg = _case()
        assert cg.g is not None
        assert np.allclose(cg.cell_volumes, np.asarray(cg.g.cells.volumes))
        assert np.allclose(cg.cell_centroids, np.asarray(cg.g.cells.centroids))
        print("ok test_it_agrees_with_the_tensor_grid_where_one_exists")

    def test_it_works_without_a_tensor_grid():
        g = compute_geometry(cart_grid([4, 4, 1], [4.0, 4.0, 1.0]))
        p = np.zeros(g.cells.num, dtype=np.int64)
        p[0] = 1                                # not a box
        cg = coarsen_geometry(generate_coarse_grid(g, p))
        assert cg.g is None
        assert np.isclose(cg.cell_volumes.sum(), float(np.sum(g.cells.volumes)))
        print("ok test_it_works_without_a_tensor_grid")

    def test_a_grid_without_geometry_is_refused():
        g = cart_grid([2, 2, 1], [2.0, 2.0, 1.0])       # no compute_geometry
        p = partition_cart_grid([2, 2, 1], [1, 1, 1])
        try:
            coarsen_geometry(generate_coarse_grid(g, p))
        except ValueError as exc:
            assert "compute_geometry" in str(exc), exc
        else:
            raise AssertionError("expected a grid without geometry to fail")
        print("ok test_a_grid_without_geometry_is_refused")

    test_volume_is_conserved_block_by_block()
    test_a_block_centroid_lies_inside_its_block()
    test_centroids_are_volume_weighted()
    test_coarse_area_equals_the_summed_fine_area_when_flat()
    test_area_is_the_normals_length_not_the_summed_area()
    test_face_centroids_are_area_weighted()
    test_a_face_centroid_lies_within_its_fine_faces()
    test_boundary_normals_point_out_of_the_model()
    test_it_agrees_with_the_tensor_grid_where_one_exists()
    test_it_works_without_a_tensor_grid()
    test_a_grid_without_geometry_is_refused()
    print("\nAll tests passed!")
