"""Extrude a 2D grid to a layered 3D grid, ported from MRST makeLayeredGrid.m.

Faces are numbered exactly as in MATLAB: all vertical-normal faces (the
layer interfaces, one n-gon per areal cell per interface) first, then the
horizontal-normal faces (one quad per areal edge per layer). A scalar
layer_spec means that many 1 m layers; an array gives the thickness of
each layer. If the areal grid's coords carry a third column (a co2lab top
surface), the extrusion starts from those depths.
"""

from typing import Union

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes


def make_layered_grid(g: Grid, layer_spec: Union[int, np.ndarray]) -> Grid:
    """Extrude the 2D grid g downward into len(layer_spec) (or layer_spec) layers."""
    if g.griddim != 2:
        raise ValueError("areal grid must be 2D")
    if np.isscalar(layer_spec):
        nlayers = int(layer_spec)
        thickness = np.ones(nlayers, dtype=np.float64)
    else:
        thickness = np.asarray(layer_spec, dtype=np.float64).ravel()
        nlayers = thickness.size

    edges = g.faces.nodes.reshape(-1, 2)
    dz = g.nodes.num
    ne = edges.shape[0]
    nc2 = g.cells.num

    # Horizontal-normal faces: each areal edge extrudes to a quad per layer.
    layer_off = np.repeat(np.arange(nlayers, dtype=np.int64), ne) * dz
    p1 = np.tile(edges[:, 0], nlayers) + layer_off
    p2 = np.tile(edges[:, 1], nlayers) + layer_off
    h_faces = np.column_stack([p1, p2, p2 + dz, p1 + dz])

    bdry = np.tile(g.faces.neighbors == -1, (nlayers, 1))
    cell_off = np.repeat(np.arange(nlayers, dtype=np.int64) * nc2, ne)
    h_neighbors = np.tile(g.faces.neighbors, (nlayers, 1)) + cell_off[:, None]
    h_neighbors[bdry] = -1

    # Vertical-normal faces: each areal cell becomes an n-gon per interface.
    cn = _cell_nodes(g)
    v_faces = (np.tile(cn, nlayers + 1)
               + np.repeat(np.arange(nlayers + 1, dtype=np.int64), cn.size) * dz)
    v_num_nodes = np.tile(np.diff(g.cells.facePos), nlayers + 1)

    nc3 = nlayers * nc2
    v_neighbors = np.full(((nlayers + 1) * nc2, 2), -1, dtype=np.int64)
    cells = np.arange(nc3, dtype=np.int64)
    v_neighbors[cells, 1] = cells
    v_neighbors[cells + nc2, 0] = cells

    neighbors = np.vstack([v_neighbors, h_neighbors])
    face_nodes = np.concatenate([v_faces, h_faces.reshape(-1)])
    num_nodes_per_face = np.concatenate(
        [v_num_nodes, np.full(h_faces.shape[0], 4, dtype=np.int64)]
    )
    node_pos = np.concatenate(
        [[0], np.cumsum(num_nodes_per_face)]
    ).astype(np.int64)
    face_pos = np.concatenate(
        [[0], np.cumsum(np.tile(np.diff(g.cells.facePos) + 2, nlayers))]
    ).astype(np.int64)

    # Cell topology from the neighbor list, as MATLAB's sortrows does.
    nf = neighbors.shape[0]
    face_idx = np.arange(nf, dtype=np.int64)
    cell_col = np.concatenate([neighbors[:, 0], neighbors[:, 1]])
    face_col = np.concatenate([face_idx, face_idx])
    sign_col = np.concatenate([np.ones(nf, dtype=np.int64),
                               -np.ones(nf, dtype=np.int64)])
    order = np.lexsort((sign_col, face_col, cell_col))
    cell_faces = face_col[order][cell_col[order] >= 0]

    out = Grid(
        cells=Cells(
            num=nc3,
            facePos=face_pos,
            faces=cell_faces,
            indexMap=np.arange(nc3, dtype=np.int64),
        ),
        faces=Faces(
            num=nf,
            nodePos=node_pos,
            nodes=face_nodes,
            neighbors=neighbors,
        ),
        nodes=Nodes(
            num=(nlayers + 1) * dz,
            coords=_extrude_coordinates(g.nodes.coords, thickness),
        ),
        griddim=3,
        type=list(g.type) + ["makeLayeredGrid"],
    )
    out.numLayers = nlayers
    out.layerSize = nc2
    return out


def _extrude_coordinates(coords: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Stack the areal nodes once per interface, appending cumulative depth."""
    nn = coords.shape[0]
    z_levels = np.concatenate([[0.0], np.cumsum(thickness)])
    z = np.repeat(z_levels, nn)
    if coords.shape[1] == 3:
        z = z + np.tile(coords[:, 2], thickness.size + 1)
    return np.column_stack([np.tile(coords[:, :2], (thickness.size + 1, 1)), z])


def _cell_nodes(g: Grid) -> np.ndarray:
    """First node of each cell edge, edges sorted back-to-back around the cell."""
    cell_no = g.cell_no()
    edges = g.faces.nodes.reshape(-1, 2)
    cell_edges = edges[g.cells.faces].copy()
    flip = g.faces.neighbors[g.cells.faces, 0] != cell_no
    cell_edges[flip] = cell_edges[flip][:, ::-1]
    for c in range(g.cells.num):
        lo, hi = g.cells.facePos[c], g.cells.facePos[c + 1]
        cell_edges[lo:hi] = _sort_edges(cell_edges[lo:hi])
    return cell_edges[:, 0].copy()


def _sort_edges(e: np.ndarray) -> np.ndarray:
    """Reorder oriented edges so each one starts where the previous ended."""
    e = e.copy()
    n = e.shape[0]
    for i in range(n - 1):
        j = n - 1
        for k in range(i + 1, n):
            if e[i, 1] == e[k, 0]:
                j = k
                break
        e[[i + 1, j]] = e[[j, i + 1]]
    return e


if __name__ == "__main__":
    from pymrst.gridprocessing.geometry import compute_geometry
    from pymrst.gridprocessing.tensor_grid import cart_grid
    from pymrst.gridprocessing.triangle_grid import triangle_grid

    def test_matches_cart_grid_volumes():
        """Extruding a 2x2 areal grid 3 layers reproduces cartGrid([2,2,3])."""
        g = compute_geometry(make_layered_grid(cart_grid([2, 2], [2.0, 2.0]), 3))
        g.check()
        ref = compute_geometry(cart_grid([2, 2, 3], [2.0, 2.0, 3.0]))
        assert g.cells.num == ref.cells.num == 12
        assert g.faces.num == ref.faces.num
        assert abs(g.cells.volumes.sum() - ref.cells.volumes.sum()) < 1e-12
        assert np.allclose(np.sort(g.cells.centroids[:, 2]),
                           np.sort(ref.cells.centroids[:, 2]))
        assert g.numLayers == 3 and g.layerSize == 4
        print("ok test_matches_cart_grid_volumes")

    def test_variable_thickness():
        """Layer volumes follow the given thickness vector."""
        g = compute_geometry(
            make_layered_grid(cart_grid([1, 1], [1.0, 1.0]), [1.0, 2.0, 3.0])
        )
        g.check()
        assert np.allclose(g.cells.volumes, [1.0, 2.0, 3.0])
        assert abs(g.nodes.coords[:, 2].max() - 6.0) < 1e-14
        print("ok test_variable_thickness")

    def test_triangle_areal_grid():
        """Extruded triangles become prisms of the right volume."""
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        t = np.array([[0, 1, 2], [1, 3, 2]])
        g = compute_geometry(make_layered_grid(triangle_grid(p, t), 2))
        g.check()
        assert g.cells.num == 4
        assert np.allclose(g.cells.volumes, 0.5)
        assert np.all(np.diff(g.cells.facePos) == 5)  # 3 quads + 2 triangles
        print("ok test_triangle_areal_grid")

    def test_depth_offset_coords():
        """A third coords column shifts the extrusion datum per node."""
        g2 = cart_grid([1, 1], [1.0, 1.0])
        g2.nodes.coords = np.column_stack(
            [g2.nodes.coords, np.full(g2.nodes.num, 100.0)]
        )
        g = make_layered_grid(g2, 2)
        assert abs(g.nodes.coords[:, 2].min() - 100.0) < 1e-14
        assert abs(g.nodes.coords[:, 2].max() - 102.0) < 1e-14
        print("ok test_depth_offset_coords")

    test_matches_cart_grid_volumes()
    test_variable_thickness()
    test_triangle_areal_grid()
    test_depth_offset_coords()
    print("\nAll tests passed!")
