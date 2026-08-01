"""Cross-module invariants in `pymrst/utils` that no single file can check.

Each module's own tests cover it in isolation. These cover the seams: that the
exports resolve to what callers expect, and that tools which compute the same
quantity by different routes agree across every kind of grid PyMRST can build -
not just the Cartesian one each module happens to use.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry, triangle_grid


def grids():
    """One of each kind PyMRST can build, named for readable test ids."""
    made = {
        "cart2d": compute_geometry(cart_grid([4, 3], [4.0, 3.0])),
        "cart3d": compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0])),
    }

    twisted = cart_grid([4, 4, 2], [4.0, 4.0, 2.0])
    coords = np.asarray(twisted.nodes.coords, dtype=float).copy()
    rng = np.random.default_rng(11)
    inside = np.all((coords > 1e-9)
                    & (coords < np.array([4.0, 4.0, 2.0]) - 1e-9), axis=1)
    coords[inside] += rng.uniform(-0.15, 0.15, coords[inside].shape)
    twisted.nodes.coords = coords
    made["twisted3d"] = compute_geometry(twisted)

    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                       [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    made["triangle2d"] = compute_geometry(triangle_grid(points, triangles))
    return made


GRIDS = grids()
NAMES = sorted(GRIDS)


# --------------------------------------------------------------- the exports

def test_every_exported_name_resolves_to_a_callable():
    """Guards the shadowing trap this codebase has already been bitten by.

    A module and a function sharing a name in one package - `mcolon.py`
    defining `mcolon` - means the package attribute is whichever was bound
    last. Importing the module rebinds it, and a caller doing
    `from pymrst.utils import mcolon` then gets the *module* and fails on call.
    Nearly every file here has that shape, so this checks each exported name is
    the function.
    """
    import pymrst.utils as utils
    import pymrst.utils.gridtools as gridtools

    from types import ModuleType
    shadowed = []
    for package in (utils, gridtools):
        for name in package.__all__:
            value = getattr(package, name, None)
            if value is None:
                shadowed.append(f"{package.__name__}.{name} is missing")
            elif isinstance(value, ModuleType):
                shadowed.append(f"{package.__name__}.{name} is a module")
    assert not shadowed, shadowed


def test_the_export_lists_have_no_duplicates():
    import pymrst.utils as utils
    import pymrst.utils.gridtools as gridtools
    for package in (utils, gridtools):
        names = package.__all__
        assert len(names) == len(set(names)), \
            f"{package.__name__}.__all__ repeats: " \
            f"{[n for n in set(names) if names.count(n) > 1]}"


# ------------------------------------------------------- agreement on grids

@pytest.mark.parametrize("name", NAMES)
def test_gauss_greens_agrees_with_compute_geometry(name):
    """Two unrelated routes to a cell volume. They part company only where the
    faces are non-planar, which `volume_by_gauss_greens` documents - so the
    twisted grid gets the looser bound, and only on the per-cell figure."""
    from pymrst.utils.gridtools.volume_by_gauss_greens import (
        volume_by_gauss_greens,
    )
    g = GRIDS[name]
    got = volume_by_gauss_greens(g)[:, 0]
    volumes = np.asarray(g.cells.volumes, dtype=float)

    # The total holds everywhere: interior face errors cancel between cells.
    assert np.isclose(np.sum(got), np.sum(volumes), rtol=1e-9)
    if name != "twisted3d":
        assert np.allclose(got, volumes, rtol=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_sorting_a_grid_preserves_it(name):
    """`sort_grid` renumbers; it must not change the geometry."""
    from pymrst.utils.gridtools.sort_grid import sort_grid
    g = GRIDS[name]
    sorted_grid = compute_geometry(sort_grid(g))
    assert sorted_grid.cells.num == g.cells.num
    assert np.isclose(np.sum(sorted_grid.cells.volumes),
                      np.sum(g.cells.volumes), rtol=1e-10)
    assert np.allclose(np.sort(np.asarray(sorted_grid.cells.volumes)),
                       np.sort(np.asarray(g.cells.volumes)), rtol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_a_renumbered_grid_compares_equal_after_sorting(name):
    """What `compare_grids(sort=True)` is for, on every grid type."""
    import copy
    from pymrst.utils.gridtools.compare_grids import compare_grids
    g = GRIDS[name]
    shuffled = copy.deepcopy(g)
    order = np.random.default_rng(5).permutation(g.nodes.num)
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    shuffled.nodes.coords = np.asarray(g.nodes.coords)[order]
    shuffled.faces.nodes = rank[np.asarray(g.faces.nodes, dtype=np.int64)]

    same, differences = compare_grids(g, compute_geometry(shuffled), sort=True)
    assert same, differences


@pytest.mark.parametrize("name", NAMES)
def test_the_augmented_maps_are_mutually_consistent(name):
    """cell-to-node and node-to-cell must describe the same incidences."""
    from pymrst.utils.gridtools.create_augmented_grid import (
        create_augmented_grid,
    )
    g = create_augmented_grid(GRIDS[name])
    cell_pos = np.asarray(g.cells.nodePos, dtype=np.int64)
    forward = {(c, int(n)) for c in range(g.cells.num)
               for n in np.asarray(g.cells.nodes)[cell_pos[c]:cell_pos[c + 1]]}
    node_pos = np.asarray(g.nodes.cellPos, dtype=np.int64)
    backward = {(int(c), n) for n in range(g.nodes.num)
                for c in np.asarray(g.nodes.cells)[node_pos[n]:node_pos[n + 1]]}
    assert forward == backward


@pytest.mark.parametrize("name", NAMES)
def test_the_incidence_table_matches_the_grid(name):
    """`create_grid_mappings` underpins the augmented grid; its rows must be
    real incidences on every grid type, not only Cartesian ones."""
    from pymrst.utils.gridtools.create_grid_mappings import create_grid_mappings
    g = GRIDS[name]
    table = create_grid_mappings(g)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nodes = np.asarray(g.faces.nodes, dtype=np.int64)

    for cell, node, half, face in table[::7]:      # a sample; the table is big
        assert node in nodes[node_pos[face]:node_pos[face + 1]]
        assert cell_faces[half] == face
        assert face_pos[cell] <= half < face_pos[cell + 1]


@pytest.mark.parametrize("name", ["cart2d", "cart3d", "triangle2d"])
def test_the_segment_filter_never_misses_a_crossed_face(name):
    """`get_faces_close_to_segment` promises a superset. Checked against
    `find_enclosing_cell` walking the segment, on more than one grid type."""
    from pymrst.utils.gridtools.add_bounding_box_fields import (
        add_bounding_box_fields,
    )
    from pymrst.utils.gridtools.find_enclosing_cell import find_enclosing_cell
    from pymrst.utils.gridtools.get_faces_close_to_segment import (
        get_faces_close_to_segment,
    )
    g = add_bounding_box_fields(GRIDS[name])
    coords = np.asarray(g.nodes.coords, dtype=float)
    low, high = coords.min(axis=0), coords.max(axis=0)
    seg = np.vstack([low + 0.2 * (high - low), low + 0.8 * (high - low)])

    returned = set(get_faces_close_to_segment(g, seg).tolist())

    points = seg[0] + np.linspace(0, 1, 2000)[:, None] * (seg[1] - seg[0])
    cells = np.asarray(find_enclosing_cell(g, points), dtype=np.int64)
    cells = cells[cells >= 0]
    if cells.size == 0:
        pytest.skip("the probe segment missed the grid")
    path = cells[np.concatenate([[True], np.diff(cells) != 0])]

    neighbours = np.asarray(g.faces.neighbors, dtype=np.int64)
    for a, b in zip(path[:-1], path[1:]):
        shared = np.flatnonzero(
            ((neighbours[:, 0] == a) & (neighbours[:, 1] == b))
            | ((neighbours[:, 0] == b) & (neighbours[:, 1] == a)))
        assert set(shared.tolist()) <= returned, (a, b)
