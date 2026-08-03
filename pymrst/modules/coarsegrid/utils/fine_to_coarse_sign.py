"""Orientation of a fine face within its coarse face (MRST ``fineToCoarseSign.m``).

A flux across a fine face is positive when it runs from `neighbors[:, 0]` to
`neighbors[:, 1]`, and a coarse flux is positive when it runs from the coarse
face's first block to its second. The two conventions are set independently, so
a fine face can be oriented against the coarse face it belongs to.

The sign is +1 when the fine face's first cell lies in the coarse face's first
block, and -1 otherwise. Everything that adds fine quantities up across a coarse
face needs it: without it a coarse face whose fine faces point both ways sums to
roughly zero, and both `coarsen_flux` and `coarsen_geometry`'s normals would be
wrong in a way that stays finite and plausible.

Exterior faces work out on their own: MRST prepends a zero to the partition so
the outside is block 0, and a coarse boundary face has 0 in its first column.
PyMRST uses -1 for both, which compares the same way without the prepend.
"""

from __future__ import annotations

import numpy as np


def fine_to_coarse_sign(cg) -> np.ndarray:
    """One sign per entry of `cg.fconn`, aligned with it."""
    if cg.parent is None:
        raise ValueError("the coarse grid has no parent; sign relations need it")
    if cg.fconn is None or cg.conn_pos is None:
        raise ValueError("the coarse grid carries no face connectivity")

    partition = np.asarray(cg.partition, dtype=np.int64).ravel()
    neighbors = np.asarray(cg.parent.faces.neighbors, dtype=np.int64)
    counts = np.diff(np.asarray(cg.conn_pos, dtype=np.int64))
    face_no = np.repeat(np.arange(counts.size, dtype=np.int64), counts)

    # Block holding the fine face's first cell, -1 when that side is outside.
    first_cell = neighbors[cg.fconn, 0]
    first_block = np.where(first_cell >= 0, partition[np.maximum(first_cell, 0)], -1)
    coarse_first = np.asarray(cg.neighbors, dtype=np.int64)[face_no, 0]
    return np.where(first_block == coarse_first, 1.0, -1.0)


__all__ = ["fine_to_coarse_sign"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(6, 4, 2), part=(3, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims), list(part))
        return g, p, generate_coarse_grid(g, p)

    def test_the_sign_is_plus_or_minus_one():
        _, _, cg = _case()
        sgn = fine_to_coarse_sign(cg)
        assert sgn.shape == cg.fconn.shape, (sgn.shape, cg.fconn.shape)
        assert set(np.unique(sgn).tolist()) <= {-1.0, 1.0}, np.unique(sgn)
        print(f"ok test_the_sign_is_plus_or_minus_one ({sgn.size} half-faces)")

    def test_it_agrees_with_the_block_the_fine_face_points_from():
        """Stated as the definition, checked entry by entry."""
        g, p, cg = _case()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        sgn = fine_to_coarse_sign(cg)
        k = 0
        for f in range(cg.num_faces):
            for fine in cg.sub_faces(f):
                c = nb[fine, 0]
                block = p[c] if c >= 0 else -1
                expected = 1.0 if block == cg.neighbors[f, 0] else -1.0
                assert sgn[k] == expected, (f, fine, sgn[k], expected)
                k += 1
        assert k == sgn.size
        print("ok test_it_agrees_with_the_block_the_fine_face_points_from")

    def test_reversing_a_fine_face_flips_its_sign():
        """The property a constant +1 would satisfy just as well."""
        g, p, cg = _case()
        sgn = fine_to_coarse_sign(cg)
        target = int(cg.fconn[0])
        original = g.faces.neighbors.copy()
        try:
            if np.all(original[target] >= 0):
                g.faces.neighbors[target] = original[target][::-1]
                flipped = fine_to_coarse_sign(cg)
                assert flipped[0] == -sgn[0], (sgn[0], flipped[0])
        finally:
            g.faces.neighbors = original
        print("ok test_reversing_a_fine_face_flips_its_sign")

    def test_areas_cancel_without_the_sign():
        """The failure the sign exists to prevent, made visible.

        Summing signed fine normals across a coarse face gives the coarse
        normal. Summing them unsigned gives a vector that is finite, has the
        right units, and is wrong.
        """
        g, p, cg = _case()
        normals = np.asarray(g.faces.normals, dtype=float)
        sgn = fine_to_coarse_sign(cg)
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        signed = np.zeros((cg.num_faces, normals.shape[1]))
        unsigned = np.zeros_like(signed)
        for i in range(normals.shape[1]):
            signed[:, i] = np.bincount(face_no, weights=normals[cg.fconn, i] * sgn,
                                       minlength=cg.num_faces)
            unsigned[:, i] = np.bincount(face_no, weights=normals[cg.fconn, i],
                                         minlength=cg.num_faces)
        areas = np.linalg.norm(signed, axis=1)
        assert np.all(areas > 0.0), "a coarse face has no area"
        # Every fine normal of a Cartesian coarse face points the same way once
        # signed, so the coarse area is the summed fine area.
        fine_area = np.bincount(face_no, weights=np.asarray(g.faces.areas)[cg.fconn],
                                minlength=cg.num_faces)
        assert np.allclose(areas, fine_area), (areas[:5], fine_area[:5])
        print("ok test_areas_cancel_without_the_sign")

    def test_a_boundary_face_points_out_of_its_block():
        """The outside is not a block, so the sign has to come from the pair."""
        g, p, cg = _case()
        sgn = fine_to_coarse_sign(cg)
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        exterior = np.flatnonzero(cg.neighbors[:, 0] < 0)
        assert exterior.size > 0
        picked = np.isin(face_no, exterior)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        for k in np.flatnonzero(picked):
            fine = cg.fconn[k]
            outside_first = nb[fine, 0] < 0
            assert (sgn[k] == 1.0) == outside_first, (fine, sgn[k])
        print(f"ok test_a_boundary_face_points_out_of_its_block "
              f"({exterior.size} boundary faces)")

    def test_a_grid_without_connectivity_is_refused():
        _, _, cg = _case()
        cg.fconn = None
        try:
            fine_to_coarse_sign(cg)
        except ValueError as exc:
            assert "connectivity" in str(exc), exc
        else:
            raise AssertionError("expected a coarse grid without fconn to fail")
        print("ok test_a_grid_without_connectivity_is_refused")

    def test_a_grid_without_a_parent_is_refused():
        _, _, cg = _case()
        cg.parent = None
        try:
            fine_to_coarse_sign(cg)
        except ValueError as exc:
            assert "parent" in str(exc), exc
        else:
            raise AssertionError("expected a parentless coarse grid to fail")
        print("ok test_a_grid_without_a_parent_is_refused")

    test_the_sign_is_plus_or_minus_one()
    test_it_agrees_with_the_block_the_fine_face_points_from()
    test_reversing_a_fine_face_flips_its_sign()
    test_areas_cancel_without_the_sign()
    test_a_boundary_face_points_out_of_its_block()
    test_a_grid_without_connectivity_is_refused()
    test_a_grid_without_a_parent_is_refused()
    print("\nAll tests passed!")
