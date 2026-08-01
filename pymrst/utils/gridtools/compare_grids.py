"""Are two grids the same? (MRST ``utils/gridtools/compareGrids.m``).

Compares structure and geometry field by field and reports what differs.
Useful when a grid has been round-tripped through a file, rebuilt by a
different route, or perturbed by a processing step that was meant to be
lossless.

MRST prints its findings and returns nothing. This returns
`(same, differences)`, so a test can assert on the result instead of capturing
stdout - the same change made to `check_grid`, for the same reason.

**Numbering is compared unless you ask otherwise.** Two grids can describe
identical geometry with their cells in a different order, and comparing those
field by field reports differences everywhere. Pass `sort=True` to put both
into canonical order first, with `sort_grid`, which answers "are these the same
grid?" rather than "are these the same arrays?". It is off by default because
sorting is much the more expensive of the two, and because a round-tripped grid
whose numbering *should* have been preserved is exactly the case where you want
to be told that it was not.

Sorting drops derived geometry, so `sort=True` recomputes it and the geometric
fields are still compared.
"""

from typing import List, Tuple

import numpy as np


def compare_grids(g1, g2, tol: float = 1e-10,
                  sort: bool = False) -> Tuple[bool, List[str]]:
    """Return ``(same, differences)`` for two grids.

    With ``sort=False`` (the default) the grids are compared in their given
    order; with ``sort=True`` both are put into canonical order first, so a
    difference in numbering alone is not reported. See the module docstring.
    """
    differences: List[str] = []

    if sort:
        # Imported here: sort_grid's tests compare grids, so a module-level
        # import would close a cycle.
        from pymrst.gridprocessing import compute_geometry
        from pymrst.utils.gridtools.sort_grid import sort_grid
        g1, g2 = (compute_geometry(sort_grid(g)) for g in (g1, g2))

    for what, a, b in (("cells", g1.cells.num, g2.cells.num),
                       ("faces", g1.faces.num, g2.faces.num),
                       ("nodes", g1.nodes.num, g2.nodes.num)):
        if a != b:
            differences.append(f"the number of {what} differs: {a} vs {b}")
    if differences:
        return False, differences        # nothing else can line up

    if g1.griddim != g2.griddim:
        differences.append(
            f"the dimension differs: {g1.griddim} vs {g2.griddim}")
        return False, differences

    def compare(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            differences.append(f"{name}: shape {a.shape} vs {b.shape}")
            return
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            scale = max(float(np.max(np.abs(a))), 1.0) if a.size else 1.0
            if not np.allclose(a, b, rtol=0.0, atol=tol * scale):
                worst = float(np.max(np.abs(a - b))) if a.size else 0.0
                differences.append(f"{name}: max difference {worst:.3e}")
        elif not np.array_equal(a, b):
            count = int(np.count_nonzero(a != b))
            differences.append(f"{name}: {count} entries differ")

    compare("nodes.coords", g1.nodes.coords, g2.nodes.coords)
    compare("faces.nodes", g1.faces.nodes, g2.faces.nodes)
    compare("faces.nodePos", g1.faces.nodePos, g2.faces.nodePos)
    compare("faces.neighbors", g1.faces.neighbors, g2.faces.neighbors)
    compare("cells.faces", g1.cells.faces, g2.cells.faces)
    compare("cells.facePos", g1.cells.facePos, g2.cells.facePos)

    for name in ("volumes", "centroids"):
        a, b = getattr(g1.cells, name, None), getattr(g2.cells, name, None)
        if a is not None and b is not None:
            compare(f"cells.{name}", a, b)
    for name in ("areas", "normals", "centroids"):
        a, b = getattr(g1.faces, name, None), getattr(g2.faces, name, None)
        if a is not None and b is not None:
            compare(f"faces.{name}", a, b)

    return not differences, differences


compareGrids = compare_grids


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def build(dims=(3, 3), sizes=(3.0, 3.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def test_a_grid_equals_itself() -> None:
        g = build()
        same, differences = compare_grids(g, g)
        assert same, differences
        print("ok test_a_grid_equals_itself")

    def test_two_identical_builds_agree() -> None:
        same, differences = compare_grids(build(), build())
        assert same, differences
        print("ok test_two_identical_builds_agree")

    def test_a_different_size_is_reported() -> None:
        same, differences = compare_grids(build(), build(sizes=(4.0, 3.0)))
        assert not same and any("coords" in d for d in differences), differences
        print("ok test_a_different_size_is_reported")

    def test_a_different_cell_count_is_reported_first() -> None:
        """And stops the comparison, since nothing else can line up."""
        same, differences = compare_grids(build(), build(dims=(4, 3)))
        assert not same and "number of cells" in differences[0], differences
        print("ok test_a_different_cell_count_is_reported_first")

    def test_a_moved_node_is_caught() -> None:
        a, b = build(), build()
        coords = np.asarray(b.nodes.coords, dtype=float).copy()
        coords[0, 0] += 0.5
        b.nodes.coords = coords
        same, differences = compare_grids(a, b)
        assert not same and any("coords" in d for d in differences), differences
        print("ok test_a_moved_node_is_caught")

    def test_a_tiny_perturbation_is_within_tolerance() -> None:
        a, b = build(), build()
        coords = np.asarray(b.nodes.coords, dtype=float).copy()
        coords[0, 0] += 1e-14
        b.nodes.coords = coords
        assert compare_grids(a, b)[0]
        print("ok test_a_tiny_perturbation_is_within_tolerance")

    def test_a_changed_topology_is_caught() -> None:
        a, b = build(), build()
        neighbours = np.asarray(b.faces.neighbors).copy()
        neighbours[[0, 1]] = neighbours[[1, 0]]
        b.faces.neighbors = neighbours
        same, differences = compare_grids(a, b)
        assert not same and any("neighbors" in d for d in differences)
        print("ok test_a_changed_topology_is_caught")

    def test_dimension_mismatch_is_reported() -> None:
        a = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        b = compute_geometry(cart_grid([8, 1], [8.0, 1.0]))
        same, differences = compare_grids(a, b)
        assert not same, differences
        print("ok test_dimension_mismatch_is_reported")

    def test_every_difference_is_named() -> None:
        a, b = build(), build()
        coords = np.asarray(b.nodes.coords, dtype=float).copy()
        coords[0, 0] += 1.0
        b.nodes.coords = coords
        volumes = np.asarray(b.cells.volumes, dtype=float).copy()
        volumes[0] += 1.0
        b.cells.volumes = volumes
        same, differences = compare_grids(a, b)
        assert len(differences) >= 2, differences
        print("ok test_every_difference_is_named")

    test_a_grid_equals_itself()
    test_two_identical_builds_agree()
    test_a_different_size_is_reported()
    test_a_different_cell_count_is_reported_first()
    test_a_moved_node_is_caught()
    test_a_tiny_perturbation_is_within_tolerance()
    test_a_changed_topology_is_caught()
    test_dimension_mismatch_is_reported()
    def _permuted_nodes(g, seed=0):
        """The same grid with its nodes numbered differently."""
        import copy
        h = copy.deepcopy(g)
        order = np.random.default_rng(seed).permutation(g.nodes.num)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        h.nodes.coords = np.asarray(g.nodes.coords)[order]
        h.faces.nodes = rank[np.asarray(g.faces.nodes, dtype=np.int64)]
        return h

    def test_sorting_sees_past_a_renumbering() -> None:
        """What `sort=True` is for."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        a = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        b = compute_geometry(_permuted_nodes(a))
        same, differences = compare_grids(a, b, sort=True)
        assert same, differences
        print("ok test_sorting_sees_past_a_renumbering")

    def test_without_sorting_the_renumbering_is_reported() -> None:
        """The default still answers the other question."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        a = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        b = compute_geometry(_permuted_nodes(a))
        assert not compare_grids(a, b)[0]
        print("ok test_without_sorting_the_renumbering_is_reported")

    def test_sorting_still_catches_a_real_difference() -> None:
        """Sorting must not turn into 'always equal'."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        a = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        b = _permuted_nodes(a)
        coords = np.asarray(b.nodes.coords, dtype=float).copy()
        coords[0, 0] += 0.5
        b.nodes.coords = coords
        same, differences = compare_grids(a, compute_geometry(b), sort=True)
        assert not same and differences
        print("ok test_sorting_still_catches_a_real_difference")

    def test_sorting_compares_geometry_not_just_topology() -> None:
        """Sorting drops derived geometry; it has to be recomputed, or a grid
        of the wrong size would compare equal to one of the right size."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        a = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        b = compute_geometry(cart_grid([3, 3], [6.0, 6.0]))
        same, differences = compare_grids(a, b, sort=True)
        assert not same, "grids of different size must not compare equal"
        assert any("centroid" in d or "volume" in d or "coords" in d
                   for d in differences), differences
        print("ok test_sorting_compares_geometry_not_just_topology")

    test_every_difference_is_named()
    test_sorting_sees_past_a_renumbering()
    test_without_sorting_the_renumbering_is_reported()
    test_sorting_still_catches_a_real_difference()
    test_sorting_compares_geometry_not_just_topology()
    print("\nAll tests passed!")
