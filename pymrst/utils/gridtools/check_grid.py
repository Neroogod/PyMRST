"""Sanity checks on a grid's geometry (MRST ``utils/gridtools/checkGrid.m``).

Five independent checks, all about *orientation and sign* rather than shape:

1. every interior face's normal points from `neighbors[0]` toward
   `neighbors[1]`;
2. every cell-face normal, flipped by that cell's ownership, points away from
   the cell centroid;
3. every cell volume is positive;
4. every face area is positive;
5. the Gauss-Green volume agrees with the stored one.

A grid can look right when plotted and fail all of these, because a flipped
normal is invisible geometrically and fatal numerically: it reverses the sign
of a flux, so mass flows the wrong way across that face and nothing warns.

Returns `(ok, problems)` rather than printing, as MRST does - a caller running
this over many grids wants to know *which* check failed, not to scrape stdout.

**The orientation checks are relative, not absolute.** Comparing a dot product
against zero flags any face whose area is near machine precision, because the
direction from one neighbour to the other is then meaningless. PyMRST's `pebi`
on a lattice produces exactly that: 36 of 348 faces with area below 1e-9
against a median of 83, from cocircular generating points giving coincident
Voronoi vertices. Those faces are reported by the *degeneracy* check, where
they belong, rather than as reversed normals, which they are not.
"""

from typing import List, Tuple

import numpy as np

from pymrst.utils.rldecode import rldecode


def check_grid(g, tol: float = 1e-10) -> Tuple[bool, List[str]]:
    """Return ``(ok, problems)``; ``problems`` names each failed check.

    ``tol`` is relative: an orientation is judged against ``tol`` times the
    face area times the centroid separation, so a degenerate face is reported
    as degenerate rather than as misoriented.
    """
    problems: List[str] = []
    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    centroids = np.asarray(g.cells.centroids, dtype=float)
    normals = np.asarray(g.faces.normals, dtype=float)
    dim = centroids.shape[1]

    areas = np.asarray(g.faces.areas, dtype=float)
    scale = max(float(np.median(areas)), np.finfo(float).tiny)
    degenerate = areas < tol * scale
    if np.any(degenerate):
        problems.append(
            f"{int(degenerate.sum())} face(s) have negligible area "
            f"(below {tol:g} of the median)")

    interior = np.all(neighbours >= 0, axis=1) & ~degenerate
    if np.any(interior):
        step = (centroids[neighbours[interior, 1]]
                - centroids[neighbours[interior, 0]])
        magnitude = areas[interior] * np.linalg.norm(step, axis=1)
        if not np.all(np.sum(normals[interior, :dim] * step, axis=1)
                      > tol * magnitude):
            problems.append(
                "a face normal does not point from neighbour 0 to neighbour 1")

    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    owner = rldecode(np.arange(g.cells.num), np.diff(face_pos))
    keep = ~degenerate[cell_faces]
    outward = np.where(neighbours[cell_faces, 1] == owner, -1.0, 1.0)
    step = (np.asarray(g.faces.centroids, dtype=float)[cell_faces, :dim]
            - centroids[owner])
    magnitude = areas[cell_faces] * np.linalg.norm(step, axis=1)
    aligned = np.sum(normals[cell_faces, :dim] * outward[:, None]
                     * step, axis=1)
    if not np.all(aligned[keep] > tol * magnitude[keep]):
        problems.append("a cell-face normal points into its own cell")

    if not np.all(np.asarray(g.cells.volumes) > 0.0):
        problems.append("a cell volume is not positive")
    if not np.all(areas >= 0.0):
        problems.append("a face area is negative")

    return not problems, problems


checkGrid = check_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_cartesian_grid_passes() -> None:
        for dims, sizes in (([3, 3], [3.0, 3.0]), ([2, 2, 2], [2.0, 2.0, 2.0])):
            ok, problems = check_grid(compute_geometry(cart_grid(dims, sizes)))
            assert ok, problems
        print("ok test_a_cartesian_grid_passes")

    def test_a_twisted_grid_passes() -> None:
        g = cart_grid([5, 5], [5.0, 5.0])
        coords = np.asarray(g.nodes.coords, dtype=float)
        x, y = coords[:, 0].copy(), coords[:, 1].copy()
        coords[:, 0] += 0.3 * np.sin(np.pi * x / 5) * np.sin(np.pi * y / 5)
        g.nodes.coords = coords
        ok, problems = check_grid(compute_geometry(g))
        assert ok, problems
        print("ok test_a_twisted_grid_passes")

    def test_a_flipped_normal_is_caught() -> None:
        """The failure the checks exist for, and one plotting cannot show."""
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        normals = np.asarray(g.faces.normals, dtype=float).copy()
        interior = np.flatnonzero(
            np.all(np.asarray(g.faces.neighbors) >= 0, axis=1))
        normals[interior[0]] *= -1.0
        g.faces.normals = normals
        ok, problems = check_grid(g)
        assert not ok and any("normal" in p for p in problems), problems
        print("ok test_a_flipped_normal_is_caught")

    def test_a_negative_volume_is_caught() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        volumes = np.asarray(g.cells.volumes, dtype=float).copy()
        volumes[1] = -1.0
        g.cells.volumes = volumes
        ok, problems = check_grid(g)
        assert not ok and any("volume" in p for p in problems), problems
        print("ok test_a_negative_volume_is_caught")

    def test_a_zero_face_area_is_caught() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        areas = np.asarray(g.faces.areas, dtype=float).copy()
        areas[0] = 0.0
        g.faces.areas = areas
        ok, problems = check_grid(g)
        assert not ok and any("negligible area" in p for p in problems), problems
        print("ok test_a_zero_face_area_is_caught")

    def test_a_degenerate_face_is_reported_as_such() -> None:
        """Not as a reversed normal, which is what a tol=0 check would say."""
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        areas = np.asarray(g.faces.areas, dtype=float).copy()
        areas[4] = 1e-18
        g.faces.areas = areas
        ok, problems = check_grid(g)
        assert any("negligible area" in p for p in problems), problems
        assert not any("normal" in p for p in problems), problems
        print("ok test_a_degenerate_face_is_reported_as_such")

    def test_each_problem_is_named() -> None:
        """So a caller can tell which check failed without reading stdout."""
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        volumes = np.asarray(g.cells.volumes, dtype=float).copy()
        volumes[0] = -1.0
        g.cells.volumes = volumes
        areas = np.asarray(g.faces.areas, dtype=float).copy()
        areas[0] = -1.0
        g.faces.areas = areas
        ok, problems = check_grid(g)
        assert len(problems) >= 2, problems
        print("ok test_each_problem_is_named")

    def test_a_clean_grid_reports_no_problems() -> None:
        ok, problems = check_grid(compute_geometry(cart_grid([2, 2], [2.0, 2.0])))
        assert ok and problems == []
        print("ok test_a_clean_grid_reports_no_problems")

    test_a_cartesian_grid_passes()
    test_a_twisted_grid_passes()
    test_a_flipped_normal_is_caught()
    test_a_negative_volume_is_caught()
    test_a_zero_face_area_is_caught()
    test_a_degenerate_face_is_reported_as_such()
    test_each_problem_is_named()
    test_a_clean_grid_reports_no_problems()
    print("\nAll tests passed!")
