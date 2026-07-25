"""Convert between face fluxes and cell-centred velocities (MRST `utils/`).

Three small, load-bearing conversions used throughout MRST wherever a flux field
has to change representation:

- `face_flux_to_cell_flux` - a face flux (one value per interface) becomes a
  *half-face* flux, signed so it is positive out of each cell.
- `face_flux_to_cell_velocity` - reconstructs a single constant velocity vector
  per cell from its face fluxes, the lowest-order (Pollock) reconstruction
  `v = (1/|c|) * sum_hf q_hf (x_face - x_cell)`. Needed by `mimetic` and by any
  streamline / velocity post-processing.
- `cell_flux_to_face_flux` - averages the two half-face fluxes back onto a face.

These are pure grid-connectivity operations: they read `cells.faces`, the
`facePos` pointers, `faces.neighbors` and the centroids, and nothing else.
"""

from typing import Tuple

import numpy as np


def _cell_no_faces(g) -> Tuple[np.ndarray, np.ndarray]:
    """Per half-face: the cell it belongs to and the face it is (MRST
    `getCellNoFaces`)."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    counts = np.diff(face_pos)
    cell_no = np.repeat(np.arange(g.cells.num, dtype=np.int64), counts)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    return cell_no, cell_faces


def _outward_sign(g, cell_no: np.ndarray, cell_faces: np.ndarray) -> np.ndarray:
    """`+1` where the cell is the face's first neighbour (flux out of the cell
    aligns with the face normal), `-1` otherwise."""
    n0 = np.asarray(g.faces.neighbors, dtype=np.int64)[cell_faces, 0]
    return np.where(n0 == cell_no, 1.0, -1.0)


def face_flux_to_cell_flux(g, face_flux: np.ndarray) -> np.ndarray:
    """Half-face flux, positive out of each cell (MRST `faceFlux2cellFlux`)."""
    cell_no, cell_faces = _cell_no_faces(g)
    sgn = _outward_sign(g, cell_no, cell_faces)
    ff = np.asarray(face_flux, dtype=float)
    scalar = ff.ndim == 1
    ff = np.atleast_2d(ff.T).T if not scalar else ff
    cf = sgn * ff[cell_faces] if scalar else sgn[:, None] * ff[cell_faces]
    return cf


def face_flux_to_cell_velocity(g, face_flux: np.ndarray) -> np.ndarray:
    """Constant per-cell velocity from face fluxes (MRST `faceFlux2cellVelocity`).

    Returns `(n_cells, dim)`. `v_c = (1/|c|) * sum over the cell's half-faces of
    q_hf * (face_centroid - cell_centroid)`.
    """
    cell_no, cell_faces = _cell_no_faces(g)
    fc = np.asarray(g.faces.centroids, dtype=float)
    cc = np.asarray(g.cells.centroids, dtype=float)
    c = fc[cell_faces] - cc[cell_no]                      # (nhf, dim)
    cf = face_flux_to_cell_flux(g, np.asarray(face_flux, dtype=float).ravel())
    dim = c.shape[1]
    v = np.zeros((g.cells.num, dim))
    for d in range(dim):
        np.add.at(v[:, d], cell_no, cf * c[:, d])
    return v / np.asarray(g.cells.volumes, dtype=float)[:, None]


def cell_flux_to_face_flux(g, cell_flux: np.ndarray) -> np.ndarray:
    """Average half-face fluxes back onto faces (MRST `cellFlux2faceFlux`)."""
    cell_no, cell_faces = _cell_no_faces(g)
    sgn = _outward_sign(g, cell_no, cell_faces)
    cflux = np.asarray(cell_flux, dtype=float)
    scalar = cflux.ndim == 1
    counts = np.bincount(cell_faces, minlength=g.faces.num)
    if scalar:
        ff = np.zeros(g.faces.num)
        np.add.at(ff, cell_faces, sgn * cflux)
        return ff / counts
    ff = np.zeros((g.faces.num, cflux.shape[1]))
    for k in range(cflux.shape[1]):
        np.add.at(ff[:, k], cell_faces, sgn * cflux[:, k])
    return ff / counts[:, None]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_uniform_x_flow_reconstructs_constant_velocity():
        """A uniform flux U across every x-face (zero on y/z) reconstructs a
        cell velocity of exactly [U, 0, 0] everywhere."""
        g = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))
        U = 2.5
        nx = g.faces.normals
        # x-faces have their normal along x; assign flux = U * area there
        area = g.faces.areas
        is_x = np.abs(nx[:, 0]) > 1e-9
        face_flux = np.zeros(g.faces.num)
        face_flux[is_x] = U * area[is_x] * np.sign(nx[is_x, 0])
        v = face_flux_to_cell_velocity(g, face_flux)
        assert np.allclose(v[:, 0], U, atol=1e-9), v[:, 0]
        assert np.allclose(v[:, 1:], 0.0, atol=1e-9)
        print(f"ok test_uniform_x_flow_reconstructs_constant_velocity (vx={v[0,0]:.3f})")

    def test_cell_flux_to_face_flux_inverts_on_interior():
        """Expanding a face flux to half-faces and averaging back returns the
        original flux on interior faces (both half-faces agree)."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        rng = np.random.default_rng(0)
        face_flux = rng.standard_normal(g.faces.num)
        cf = face_flux_to_cell_flux(g, face_flux)
        back = cell_flux_to_face_flux(g, cf)
        interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        assert np.allclose(back[interior], face_flux[interior]), \
            np.abs(back[interior] - face_flux[interior]).max()
        print("ok test_cell_flux_to_face_flux_inverts_on_interior")

    def test_half_face_flux_is_antisymmetric_across_a_face():
        """The two half-face fluxes of an interior face are equal and opposite:
        what leaves one cell enters its neighbour."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        rng = np.random.default_rng(1)
        face_flux = rng.standard_normal(g.faces.num)
        cell_no, cell_faces = _cell_no_faces(g)
        cf = face_flux_to_cell_flux(g, face_flux)
        interior = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        for f in np.flatnonzero(interior):
            hf = cf[cell_faces == f]
            assert hf.size == 2 and np.isclose(hf[0], -hf[1]), (f, hf)
        print("ok test_half_face_flux_is_antisymmetric_across_a_face")

    test_uniform_x_flow_reconstructs_constant_velocity()
    test_cell_flux_to_face_flux_inverts_on_interior()
    test_half_face_flux_is_antisymmetric_across_a_face()
    print("\nAll tests passed!")
