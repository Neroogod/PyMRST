"""Voronoi (PEBI) dual of a triangular grid, ported from MRST pebi.m.

The dual grid's nodes are the triangles' circumcenters plus, on the
boundary, the midpoints of boundary edges and the boundary nodes
themselves; its cells are the original grid's nodes; its faces are the
perpendicular bisectors of the original edges. Circumcenters that fall
outside a boundary triangle are handled as in MATLAB, by reflecting the
opposite vertex to create auxiliary triangles so the domain boundary
survives in the dual.

If the input triangulation is not Delaunay, circumcenters may fall outside
their triangle and the result can be warped, exactly as MRST warns.
"""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.gridprocessing.grid_editing import (
    add_faces,
    empty_grid,
    remove_cells,
    remove_faces,
)
from pymrst.gridprocessing.triangle_grid import triangle_grid

_SQRT_EPS = np.sqrt(np.finfo(np.float64).eps)


def pebi(g: Grid) -> Grid:
    """Compute the Voronoi dual of the triangular grid g."""
    p = _circumcenters(g)
    g, p, art = _add_auxiliary_triangles(g, p)

    b = np.any(g.faces.neighbors == -1, axis=1)
    fe = g.faces.nodes.reshape(-1, 2)

    # Midpoints of boundary edges become dual nodes.
    new_coords = 0.5 * (g.nodes.coords[fe[b, 0]] + g.nodes.coords[fe[b, 1]])
    nb = int(b.sum())
    new_nodes = g.cells.num + np.arange(nb, dtype=np.int64)
    new_nodes_art = art[fe[b, 0]] | art[fe[b, 1]]

    # Boundary nodes of g get copies among the dual nodes.
    bd_nodes = np.unique(fe[b])
    g_coords = g.nodes.coords[bd_nodes]
    g_coords_art = art[bd_nodes]
    counts = np.bincount(fe[b].reshape(-1), minlength=g.nodes.num)
    node_rank = np.cumsum(counts > 0) - 1
    num = p.shape[0] + new_coords.shape[0]

    # Dual edges: g's neighbor pairs are dual node pairs (circumcenters);
    # boundary edges connect the inner circumcenter to the edge midpoint.
    face_nodes = g.faces.neighbors.copy()
    inner = face_nodes[b, 0] + face_nodes[b, 1] + 1  # the non-boundary entry
    face_nodes[b, 0] = inner
    face_nodes[b, 1] = new_nodes

    neighbors = np.vstack([
        fe,  # g-nodes are dual cells
        np.column_stack([np.full(nb, -1, dtype=np.int64), fe[b, 0]]),
        np.column_stack([fe[b, 1], np.full(nb, -1, dtype=np.int64)]),
    ])
    face_nodes = np.vstack([
        face_nodes,
        np.column_stack([num + node_rank[fe[b, 0]], new_nodes]),
        np.column_stack([num + node_rank[fe[b, 1]], new_nodes]),
    ]).reshape(-1)

    num_faces = neighbors.shape[0]
    num_cells = int(neighbors.max()) + 1
    h = empty_grid(num_cells, np.vstack([p, new_coords, g_coords]))
    h, _ = add_faces(h, face_nodes,
                     np.full(num_faces, 2, dtype=np.int64), neighbors)

    h = _sort_cell_faces(h)

    art_h = np.concatenate(
        [np.zeros(p.shape[0], dtype=bool), new_nodes_art, g_coords_art]
    )
    h = _remove_nodes(h, np.flatnonzero(art_h))

    assert h.nodes.num == np.unique(h.faces.nodes).size
    assert h.nodes.num == int(h.faces.nodes.max()) + 1
    h = _sort_cell_faces(h)
    h = _merge_boundary_faces(h)
    assert h.nodes.num == np.unique(h.faces.nodes).size
    assert h.nodes.num == int(h.faces.nodes.max()) + 1

    h.type = ["pebi"]
    h.griddim = 2

    h = _remove_duplicate_nodes(h)
    h = _sort_cell_faces(h)
    assert h.nodes.num == np.unique(h.faces.nodes).size
    assert h.nodes.num == int(h.faces.nodes.max()) + 1
    return h


# -- dual-node construction -------------------------------------------------

def _circumcenters(g: Grid) -> np.ndarray:
    """Circumcenter of every (triangular) cell."""
    tri = _tri_nodes(g)
    coords = g.nodes.coords
    b = coords[tri[:, 1]] - coords[tri[:, 0]]
    c = coords[tri[:, 2]] - coords[tri[:, 0]]
    d = 2.0 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
    b2 = np.sum(b**2, axis=1)
    c2 = np.sum(c**2, axis=1)
    return np.column_stack([
        (c[:, 1] * b2 - b[:, 1] * c2) / d + coords[tri[:, 0], 0],
        (b[:, 0] * c2 - c[:, 0] * b2) / d + coords[tri[:, 0], 1],
    ])


def _tri_nodes(g: Grid) -> np.ndarray:
    """Node triple of each cell, from the cell's oriented edges."""
    edges = g.faces.nodes.reshape(-1, 2)
    e = edges[g.cells.faces].copy()
    reverse = g.cell_no() == g.faces.neighbors[g.cells.faces, 1]
    e[reverse] = e[reverse][:, ::-1]
    return e.reshape(-1, 6)[:, [0, 2, 4]]


def _grid_to_tri(g: Grid) -> np.ndarray:
    """Triangle list of a sorted 2D triangular grid (MRST grid2tri)."""
    assert np.all(np.diff(g.faces.nodePos) == 2)
    assert np.all(np.diff(g.cells.facePos) == 3)
    edges = g.faces.nodes.reshape(-1, 2)
    e = edges[g.cells.faces][:, ::-1].copy()
    reverse = g.cell_no() == g.faces.neighbors[g.cells.faces, 1]
    e[reverse] = e[reverse][:, ::-1]
    return e[:, 0].reshape(-1, 3)


def _point_inside(t: np.ndarray, coords: np.ndarray, points: np.ndarray):
    """For each triangle t[k], is points[k] inside it? Also return, for the
    outside points, which edge they are outside of."""
    x, y = coords[:, 0], coords[:, 1]
    t4 = np.column_stack([t, t[:, 0]])
    vx = x[t] - points[:, [0]]
    vy = y[t] - points[:, [1]]
    wx = np.diff(x[t4], axis=1)
    wy = np.diff(y[t4], axis=1)
    a = vx * wy - vy * wx
    inside = np.all(a > -_SQRT_EPS, axis=1)
    neg = np.sign(a[~inside]) == -1
    assert np.all(neg.sum(axis=1) == 1), "Internal error in pebi"
    return inside, np.argmax(neg, axis=1)


def _add_auxiliary_triangles(g: Grid, p: np.ndarray):
    """Reflect vertices of boundary triangles whose circumcenter falls
    outside the domain, so the dual preserves the boundary (MRST
    addAuxillaryTriangles). Returns (grid, circumcenters, artificial-node
    mask)."""
    g = _sort_cell_faces(g)
    t = _grid_to_tri(g)
    inside, col = _point_inside(t, g.nodes.coords, p)
    out = np.flatnonzero(~inside)
    hf = 3 * out + ((col + 1) % 3)
    is_bd = np.any(g.faces.neighbors[g.cells.faces[hf]] == -1, axis=1)

    art = np.zeros(g.nodes.num, dtype=bool)
    if not np.any(is_bd):
        return g, p, art

    f = out[is_bd]
    col = col[is_bd]
    ix1 = t[f, col]
    ix2 = t[f, (col + 1) % 3]
    ix3 = t[f, (col + 2) % 3]

    x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
    u = np.column_stack([x[ix2] - x[ix1], y[ix2] - y[ix1]])
    v = np.column_stack([x[ix3] - x[ix1], y[ix3] - y[ix1]])
    w = v - (np.sum(u * v, axis=1) / np.sum(u**2, axis=1))[:, None] * u
    reflected = np.column_stack([x[ix3], y[ix3]]) - 2.0 * w

    new_ids = g.nodes.num + np.arange(f.size, dtype=np.int64)
    coords = np.vstack([g.nodes.coords, reflected])

    # Split each offending triangle in two along the reflected vertex.
    pair = np.column_stack([t[f], t[f]])
    rows = np.arange(f.size)
    pair[rows, col] = new_ids
    pair[rows, 3 + ((col + 1) % 3)] = new_ids
    keep = np.ones(t.shape[0], dtype=bool)
    keep[f] = False
    t_new = np.vstack([t[keep], pair.reshape(-1, 3)])

    g = triangle_grid(coords, t_new)
    art = np.zeros(g.nodes.num, dtype=bool)
    art[new_ids] = True
    return g, _circumcenters(g), art


# -- topology repair on the dual --------------------------------------------

def _remove_nodes(g: Grid, nodes: np.ndarray) -> Grid:
    """Remove nodes plus the faces using them; close each affected cell with
    a new boundary face between the surviving nodes (MRST removeNodes)."""
    node_removed = np.zeros(g.nodes.num, dtype=bool)
    node_removed[nodes] = True
    face_no = np.repeat(np.arange(g.faces.num, dtype=np.int64),
                        np.diff(g.faces.nodePos))
    face_hit = np.zeros(g.faces.num, dtype=bool)
    np.logical_or.at(face_hit, face_no, node_removed[g.faces.nodes])

    cell_no = g.cell_no()
    edges = g.faces.nodes.reshape(-1, 2)
    hf = face_hit[g.cells.faces]
    cells_rep = np.repeat(cell_no[hf], 2)
    e_flat = edges[g.cells.faces[hf]].reshape(-1)
    keep = ~node_removed[e_flat]

    g, _ = remove_faces(g, np.flatnonzero(face_hit))

    cell_pairs = cells_rep[keep].reshape(-1, 2)
    node_pairs = e_flat[keep].reshape(-1, 2)
    neigh = np.column_stack([
        cell_pairs[:, 0],
        np.full(cell_pairs.shape[0], -1, dtype=np.int64),
    ])
    g, _ = add_faces(g, node_pairs.reshape(-1),
                     np.full(cell_pairs.shape[0], 2, dtype=np.int64), neigh)
    g = remove_cells(g, np.flatnonzero(np.diff(g.cells.facePos) < 3))
    return _remap_nodes(g)


def _remap_nodes(g: Grid) -> Grid:
    """Drop nodes not referenced by any face and renumber."""
    used, inverse = np.unique(g.faces.nodes, return_inverse=True)
    if used.size < g.nodes.num:
        g.faces.nodes = np.asarray(inverse, dtype=np.int64).ravel()
        g.nodes.coords = g.nodes.coords[used]
        g.nodes.num = used.size
    assert g.nodes.num == np.unique(g.faces.nodes).size
    assert g.nodes.num == int(g.faces.nodes.max()) + 1
    return g


def _remove_duplicate_nodes(g: Grid) -> Grid:
    """Merge coincident nodes; drop the faces and cells that collapse."""
    assert g.nodes.num == g.nodes.coords.shape[0]
    coords, node_map = np.unique(g.nodes.coords, axis=0, return_inverse=True)
    if coords.shape[0] == g.nodes.num:
        return g
    node_map = np.asarray(node_map, dtype=np.int64).ravel()
    g.nodes.coords = coords
    g.nodes.num = coords.shape[0]
    g.faces.nodes = node_map[g.faces.nodes]

    collapsed = np.flatnonzero(g.faces.nodes[0::2] == g.faces.nodes[1::2])
    g, _ = remove_faces(g, collapsed)
    return remove_cells(g, np.flatnonzero(np.diff(g.cells.facePos) < 3))


def _merge_boundary_faces(g: Grid) -> Grid:
    """Replace pairs of parallel boundary faces of one cell by one face."""
    cell_no = g.cell_no()
    sgn = cell_no == g.faces.neighbors[g.cells.faces, 0]
    b = np.any(g.faces.neighbors == -1, axis=1)
    i = np.flatnonzero(b[g.cells.faces])
    f = g.cells.faces[i]
    c = g.faces.neighbors[f].sum(axis=1) + 1  # the owning cell
    edges = g.faces.nodes.reshape(-1, 2)
    be = edges[f].copy()
    ns = ~sgn[i]
    be[ns] = be[ns][:, ::-1]

    # Pick out cells with two (consecutive) boundary half-faces. An odd
    # trailing entry is left unpaired and never merged, as in MATLAB.
    same = c[:-1] == c[1:]
    ind = np.concatenate([same, [True]]) | np.concatenate([[True], same])
    e_pairs = be[ind]
    f = f[ind]
    c = c[ind]
    n_pairs = e_pairs.shape[0] // 2

    parallel = _compare_normals(e_pairs[0:2 * n_pairs:2],
                                e_pairs[1:2 * n_pairs:2], g.nodes.coords)
    sel = np.repeat(parallel, 2)
    h, _ = remove_faces(g, f[:2 * n_pairs][sel])

    quads = e_pairs[:2 * n_pairs][sel].reshape(-1, 4)
    face_nodes = quads[:, [0, 3]].reshape(-1)
    n_new = int(parallel.sum())
    neigh = np.column_stack([
        c[0::2][:n_pairs][parallel],
        np.full(n_new, -1, dtype=np.int64),
    ])
    h, _ = add_faces(h, face_nodes, np.full(n_new, 2, dtype=np.int64), neigh)
    return _remap_nodes(h)


def _compare_normals(nn1: np.ndarray, nn2: np.ndarray,
                     coords: np.ndarray) -> np.ndarray:
    """Are the edges nn1 and nn2 (node pairs) parallel?"""
    x, y = coords[:, 0], coords[:, 1]
    n = np.column_stack([np.diff(y[nn1], axis=1), -np.diff(x[nn1], axis=1)])
    e = np.column_stack([np.diff(x[nn2], axis=1), np.diff(y[nn2], axis=1)])
    return np.abs(np.sum(n * e, axis=1)) < 10.0 * _SQRT_EPS


# -- cell-face sorting -------------------------------------------------------

def _sort_cell_faces(g: Grid) -> Grid:
    """Sort each 2D cell's edges back-to-back and counter-clockwise, flipping
    stored edge directions so signs stay consistent (MRST sortCellFaces)."""
    face_pos = g.cells.facePos
    cell_faces = g.cells.faces.copy()
    # cells.tags is parallel to cells.faces, so it has to be permuted with it;
    # leaving it behind silently re-labels which side every boundary face is on
    cell_tags = None if g.cells.tags is None else np.asarray(g.cells.tags).copy()
    stored = g.faces.nodes.reshape(-1, 2).copy()
    x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]

    for i in range(g.cells.num):
        lo, hi = face_pos[i], face_pos[i + 1]
        cf = cell_faces[lo:hi].copy()
        order = np.arange(cf.size, dtype=np.int64)
        reverse = g.faces.neighbors[cf, 1] == i
        e = stored[cf].copy()
        e[reverse] = e[reverse][:, ::-1]

        # Local 1-based node labels in ascending global order.
        uniq = np.unique(e)
        e = np.searchsorted(uniq, e) + 1

        m = np.arange(cf.size)
        s = np.ones(cf.size, dtype=np.int64)
        for k in range(1, cf.size + 1):
            e, m, s = _swap(k, e, m, s)
        cf = cf[m]
        order = order[m]
        assert np.array_equal(g.faces.neighbors[cf, 1] == i, reverse[m])

        e = stored[cf].copy()
        e[reverse[m]] = e[reverse[m]][:, ::-1]
        neg = s == -1
        e[neg] = e[neg][:, ::-1]

        area = 0.5 * np.sum(np.sum(x[e], axis=1) * np.diff(y[e], axis=1).ravel())
        if area < 0:
            cf = cf[::-1].copy()
            order = order[::-1].copy()
            s = -s[::-1]

        assert np.array_equal(e[:, 1], e[np.r_[1:cf.size, 0], 0])

        flip = cf[s == -1]
        stored[flip] = stored[flip][:, ::-1]
        cell_faces[lo:hi] = cf
        if cell_tags is not None:
            cell_tags[lo:hi] = np.asarray(g.cells.tags)[lo:hi][order]

    g.cells.faces = cell_faces
    if cell_tags is not None:
        g.cells.tags = cell_tags
    g.faces.nodes = stored.reshape(-1)
    return g


def _swap(k: int, e: np.ndarray, m: np.ndarray, s: np.ndarray):
    """One step of MRST's edge sort: bring the edge continuing local node k
    into row k, orient it, and relabel so the chain can continue."""
    tail = np.flatnonzero(np.any(e[k:] == k, axis=1))
    if tail.size:
        i = k + tail[0]
        e[[k - 1, i]] = e[[i, k - 1]]
        m[[k - 1, i]] = m[[i, k - 1]]
        s[[k - 1, i]] = s[[i, k - 1]]
    if e[k - 1, 0] != k:
        e[k - 1] = e[k - 1, ::-1]
        s[k - 1] = -1
    if e[k - 1, 1] > k + 1:
        val = e[k - 1, 1]
        u = e == k + 1
        v = e == val
        e[u] = val
        e[v] = k + 1
    return e, m, s


if __name__ == "__main__":
    from scipy.spatial import ConvexHull

    from pymrst.gridprocessing.geometry import compute_geometry

    def _lattice(nx: int, ny: int):
        """Unit square lattice and its (deterministic) two-triangle squares."""
        xx, yy = np.meshgrid(np.arange(nx, dtype=float),
                             np.arange(ny, dtype=float))
        p = np.column_stack([xx.ravel(), yy.ravel()])
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                tris.append([n0, n0 + 1, n0 + nx])
                tris.append([n0 + 1, n0 + nx + 1, n0 + nx])
        return p, np.array(tris)

    def test_square_lattice_gives_unit_squares():
        """The dual of a unit lattice is unit Voronoi squares inside, and
        the domain area is preserved."""
        p, t = _lattice(4, 4)
        h = compute_geometry(pebi(triangle_grid(p, t)))
        h.check()
        assert h.cells.num == p.shape[0]
        interior = np.flatnonzero(
            np.isclose(h.cells.volumes, 1.0, atol=1e-12))
        assert interior.size == 4  # the 4 interior lattice nodes
        assert abs(h.cells.volumes.sum() - 9.0) < 1e-12
        print("ok test_square_lattice_gives_unit_squares")

    def test_perturbed_lattice_covers_hull():
        """A Delaunay dual covers exactly the convex hull of the points."""
        p, _ = _lattice(5, 4)
        interior = ((p[:, 0] > 0) & (p[:, 0] < 4)
                    & (p[:, 1] > 0) & (p[:, 1] < 3))
        p[interior] += 0.2 * np.column_stack([
            np.sin(1.0 + 3.0 * np.arange(interior.sum())),
            np.cos(2.0 + 5.0 * np.arange(interior.sum())),
        ])
        h = compute_geometry(pebi(triangle_grid(p)))
        h.check()
        assert h.cells.num == p.shape[0]
        assert abs(h.cells.volumes.sum() - ConvexHull(p).volume) < 1e-10
        print("ok test_perturbed_lattice_covers_hull")

    def test_obtuse_boundary_triangle_uses_auxiliaries():
        """A circumcenter outside the domain triggers the auxiliary-triangle
        path and the boundary still closes."""
        p = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.2], [1.0, 1.0]])
        t = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 3]])
        h = compute_geometry(pebi(triangle_grid(p, t)))
        h.check()
        assert h.cells.num == 4
        hull_area = 1.0  # triangle (0,0), (2,0), (1,1)
        assert abs(h.cells.volumes.sum() - hull_area) < 1e-10
        print("ok test_obtuse_boundary_triangle_uses_auxiliaries")

    test_square_lattice_gives_unit_squares()
    test_perturbed_lattice_covers_hull()
    test_obtuse_boundary_triangle_uses_auxiliaries()
    print("\nAll tests passed!")
