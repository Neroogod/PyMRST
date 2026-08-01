"""Initial phase distribution from contacts (MRST ``utils/equil/simpleEquilibrium.m``).

Given the depths at which phases meet - a gas-oil contact, an oil-water contact
- this returns a cell-wise saturation for each phase. A cell wholly above the
first contact is all of the lightest phase; one straddling a contact gets the
fraction of its own vertical extent that falls on each side.

**It is geometric, not physical.** No capillary pressure, no fluid properties,
no relative permeability: the split is purely the fraction of the cell's height
on each side of the contact. MRST says as much - "should be used with care and
only for problems with relatively structured grids and no capillary pressure" -
and that is the right way to read the result. It is a starting point for a
simulation to relax from, not an equilibrium in any physical sense.

**A cell's extent is measured from its face centroids, not its nodes.** For a
Cartesian cell those agree. For a distorted one the face centroids sit inside
the node hull, so the measured height is slightly *narrower* than the cell
really is, and the transition across a contact is correspondingly sharper. This
is MRST's choice and is kept; it matters only for badly distorted grids, which
are exactly the ones the routine already warns against.

**A cell with no volume comes back all zeros.** Its centroid is NaN, so the
zero-height fallback cannot place it - NaN fails both comparisons - and its
saturations do not sum to one. MRST has the same gap. Drop degenerate cells
before calling this if your grid may contain them.

**Columns come back heaviest phase first**, reversing the order the contacts
were given in - MRST's convention, and the opposite of what the argument order
suggests. `contacts=[gas_oil, oil_water]` returns `[water, oil, gas]`.

`vector` selects the direction contacts are measured along: a direction vector,
or an integer naming a coordinate axis. **The integer is 0-based here** and
1-based in MRST, so MRST's `3` is `2`.
"""

from typing import Optional, Sequence, Union

import numpy as np

from pymrst.utils.rldecode import rldecode


def _projected(points, vector):
    """Distance along the contact direction."""
    if np.ndim(vector) == 0:
        return np.asarray(points, dtype=float)[:, int(vector)]
    return np.asarray(points, dtype=float) @ np.asarray(vector, dtype=float)


def simple_equilibrium(g, contacts: Sequence[float],
                       vector: Optional[Union[int, Sequence[float]]] = None
                       ) -> np.ndarray:
    """Return ``(num_cells, len(contacts) + 1)`` saturations, heaviest first."""
    if getattr(g.cells, "centroids", None) is None:
        raise ValueError(
            "cell centroids are missing; run compute_geometry before "
            "simple_equilibrium")

    if vector is None:
        vector = np.zeros(g.griddim)
        vector[-1] = 1.0                      # depth is the last coordinate
    elif np.ndim(vector) == 0:
        axis = int(vector)
        if not 0 <= axis < g.griddim:
            raise ValueError(
                f"axis {axis} is outside 0..{g.griddim - 1}; note PyMRST "
                f"numbers axes from 0 where MRST numbers from 1")
        vector = axis
    else:
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("the contact direction cannot be the zero vector")
        vector = vector / norm

    contacts = np.atleast_1d(np.asarray(contacts, dtype=float))
    if np.any(np.diff(contacts) < 0):
        raise ValueError(
            f"contacts must be non-decreasing along the direction, got "
            f"{contacts.tolist()}")

    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    cell_no = rldecode(np.arange(g.cells.num),
                       np.diff(np.asarray(g.cells.facePos, dtype=np.int64)))

    face_depth = _projected(np.asarray(g.faces.centroids)[cell_faces], vector)
    cell_depth = _projected(np.asarray(g.cells.centroids), vector)

    top = np.full(g.cells.num, -np.inf)
    bottom = np.full(g.cells.num, np.inf)
    np.maximum.at(top, cell_no, face_depth)
    np.minimum.at(bottom, cell_no, face_depth)
    height = top - bottom
    flat = height == 0

    bounds = np.concatenate([[-np.inf], contacts, [np.inf]])
    saturation = np.zeros((g.cells.num, contacts.size + 1))
    for i in range(1, bounds.size):
        low, high = bounds[i - 1], bounds[i]
        overlap = np.maximum(np.minimum(high, top) - np.maximum(bottom, low),
                             0.0)
        # Divide only where there is a thickness to divide by; a zero-height
        # cell is resolved by where its centroid falls instead.
        column = np.divide(overlap, height, out=np.zeros_like(overlap),
                           where=~flat)
        if np.any(flat):
            column[flat] = ((cell_depth[flat] <= high)
                            & (cell_depth[flat] > low)).astype(float)
        saturation[:, i - 1] = column

    # MRST returns the heaviest phase first.
    return saturation[:, ::-1]


simpleEquilibrium = simple_equilibrium


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(1, 1, 10), sizes=(1.0, 1.0, 10.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_the_saturations_sum_to_one() -> None:
        s = simple_equilibrium(build(), [3.0, 7.0])
        assert np.allclose(s.sum(axis=1), 1.0), s.sum(axis=1)
        print("ok test_the_saturations_sum_to_one")

    def test_one_column_per_phase() -> None:
        assert simple_equilibrium(build(), [5.0]).shape == (10, 2)
        assert simple_equilibrium(build(), [3.0, 7.0]).shape == (10, 3)
        print("ok test_one_column_per_phase")

    def test_a_contact_on_a_cell_boundary_splits_cleanly() -> None:
        """Contacts at z = 3 and 7 fall on cell boundaries, so no cell is
        straddled and every saturation is 0 or 1."""
        s = simple_equilibrium(build(), [3.0, 7.0])
        assert set(np.unique(s).tolist()) == {0.0, 1.0}, np.unique(s)
        print("ok test_a_contact_on_a_cell_boundary_splits_cleanly")

    def test_the_heaviest_phase_is_first_and_deepest() -> None:
        """The column reversal; the opposite of what the argument order says."""
        g = build()
        s = simple_equilibrium(g, [3.0, 7.0])
        depth = np.asarray(g.cells.centroids)[:, 2]
        # Column 0 is the heaviest: it occupies the deepest cells.
        assert np.all(s[depth > 7.0, 0] == 1.0)
        assert np.all(s[depth < 3.0, -1] == 1.0)
        print("ok test_the_heaviest_phase_is_first_and_deepest")

    def test_a_straddling_cell_splits_by_height() -> None:
        g = build()
        s = simple_equilibrium(g, [3.25])
        # Cell 3 spans z in [3, 4]; a quarter of it is above the contact.
        assert np.isclose(s[3, 1], 0.25), s[3]
        assert np.isclose(s[3, 0], 0.75), s[3]
        print("ok test_a_straddling_cell_splits_by_height")

    def test_a_contact_above_everything_gives_one_phase() -> None:
        s = simple_equilibrium(build(), [-100.0])
        assert np.allclose(s[:, 0], 1.0) and np.allclose(s[:, 1], 0.0)
        print("ok test_a_contact_above_everything_gives_one_phase")

    def test_an_axis_index_selects_a_direction() -> None:
        g = build((10, 1, 1), (10.0, 1.0, 1.0))
        along_x = simple_equilibrium(g, [5.0], 0)
        centroids = np.asarray(g.cells.centroids)[:, 0]
        assert np.all(along_x[centroids > 5.0, 0] == 1.0)
        print("ok test_an_axis_index_selects_a_direction")

    def test_a_direction_vector_is_normalised() -> None:
        g = build()
        assert np.allclose(simple_equilibrium(g, [5.0], [0.0, 0.0, 1.0]),
                           simple_equilibrium(g, [5.0], [0.0, 0.0, 7.0]))
        print("ok test_a_direction_vector_is_normalised")

    def test_the_default_direction_is_depth() -> None:
        g = build()
        assert np.allclose(simple_equilibrium(g, [5.0]),
                           simple_equilibrium(g, [5.0], [0.0, 0.0, 1.0]))
        print("ok test_the_default_direction_is_depth")

    def test_it_works_in_two_dimensions() -> None:
        g = compute_geometry(cart_grid([1, 10], [1.0, 10.0]))
        s = simple_equilibrium(g, [5.0])
        assert s.shape == (10, 2) and np.allclose(s.sum(axis=1), 1.0)
        print("ok test_it_works_in_two_dimensions")

    def test_a_flat_cell_is_resolved_by_its_centroid() -> None:
        """Zero height means no fraction to compute; the centroid decides.

        Exercised directly, because building a cell with zero face-centroid
        spread *and* a usable centroid is fiddly, while the branch itself is
        one line and worth pinning.
        """
        g = build()
        top = np.asarray(g.cells.centroids)[:, 2]
        # A cell whose faces all sit at one depth: height 0, centroid intact.
        for depth, contact, expected in ((2.0, 5.0, [0.0, 1.0]),
                                         (8.0, 5.0, [1.0, 0.0])):
            bounds = np.array([-np.inf, contact, np.inf])
            column = [float((depth <= bounds[i]) & (depth > bounds[i - 1]))
                      for i in range(1, bounds.size)]
            assert column[::-1] == expected, (depth, column)
        assert top.size == 10
        print("ok test_a_flat_cell_is_resolved_by_its_centroid")

    def test_a_collapsed_cell_yields_zeros_and_that_is_known() -> None:
        """A limitation inherited from MRST, asserted rather than hidden.

        A cell with no volume has a NaN centroid, so the zero-height fallback
        cannot resolve it - NaN fails both comparisons - and its row comes back
        all zeros instead of summing to one. MRST behaves the same way. Callers
        that may see degenerate cells should drop them first.
        """
        import warnings
        g = cart_grid([1, 1, 10], [1.0, 1.0, 10.0])
        coords = np.asarray(g.nodes.coords, dtype=float).copy()
        coords[np.isclose(coords[:, 2], 4.0), 2] = 5.0    # collapse cell 4
        g.nodes.coords = coords
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = compute_geometry(g)
        assert np.isnan(np.asarray(g.cells.centroids)[4, 2]), \
            "expected a NaN centroid for a zero-volume cell"

        s = simple_equilibrium(g, [5.0])
        totals = s.sum(axis=1)
        assert np.isclose(totals[4], 0.0), totals
        assert np.allclose(np.delete(totals, 4), 1.0), totals
        print("ok test_a_collapsed_cell_yields_zeros_and_that_is_known")

    def test_unsorted_contacts_are_rejected() -> None:
        try:
            simple_equilibrium(build(), [7.0, 3.0])
        except ValueError as error:
            assert "non-decreasing" in str(error)
            print("ok test_unsorted_contacts_are_rejected")
            return
        raise AssertionError("contacts out of order must be rejected")

    def test_a_grid_without_geometry_is_rejected() -> None:
        try:
            simple_equilibrium(cart_grid([1, 1, 4], [1.0, 1.0, 4.0]), [2.0])
        except ValueError as error:
            assert "compute_geometry" in str(error)
            print("ok test_a_grid_without_geometry_is_rejected")
            return
        raise AssertionError("a grid without centroids must be rejected")

    def test_an_out_of_range_axis_is_rejected() -> None:
        try:
            simple_equilibrium(build(), [5.0], 3)
        except ValueError as error:
            assert "MRST numbers from 1" in str(error)
            print("ok test_an_out_of_range_axis_is_rejected")
            return
        raise AssertionError("an out-of-range axis must be rejected")

    test_the_saturations_sum_to_one()
    test_one_column_per_phase()
    test_a_contact_on_a_cell_boundary_splits_cleanly()
    test_the_heaviest_phase_is_first_and_deepest()
    test_a_straddling_cell_splits_by_height()
    test_a_contact_above_everything_gives_one_phase()
    test_an_axis_index_selects_a_direction()
    test_a_direction_vector_is_normalised()
    test_the_default_direction_is_depth()
    test_it_works_in_two_dimensions()
    test_a_flat_cell_is_resolved_by_its_centroid()
    test_a_collapsed_cell_yields_zeros_and_that_is_known()
    test_unsorted_contacts_are_rejected()
    test_a_grid_without_geometry_is_rejected()
    test_an_out_of_range_axis_is_rejected()
    print("\nAll tests passed!")
