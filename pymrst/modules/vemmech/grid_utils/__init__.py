"""Grid utilities from MRST ``modules/vemmech/grid_utils``."""

from copy import deepcopy
from typing import Dict, Sequence

import numpy as np


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel(order="F")


def grdecl_xyz(grdecl: Dict) -> tuple[np.ndarray, np.ndarray]:
    """Return MRST-style ``xyz`` and ``zcorn`` arrays for a GRDECL dict.

    ``xyz`` has shape ``(6, nx+1, ny+1)`` and stores top/bottom pillar points.
    ``zcorn`` has shape ``(2*nx, 2*ny, 2*nz)``. Both use Fortran ordering,
    matching MRST's ``grdeclXYZ`` layout used by the vemmech helpers.
    """
    dims = np.asarray(grdecl["cartDims"], dtype=np.int64).ravel()
    nx, ny, nz = (int(v) for v in dims)
    xyz = np.asarray(grdecl["COORD"], dtype=float).reshape(
        (6, nx + 1, ny + 1), order="F"
    )
    zcorn = np.asarray(grdecl["ZCORN"], dtype=float).reshape(
        (2 * nx, 2 * ny, 2 * nz), order="F"
    )
    return xyz.copy(), zcorn.copy()


def flip_grid(g):
    """Flip a grid's coordinate axes as MRST ``flipGrid``.

    MRST maps ``(x, y, z) -> (z, x, y)`` and remaps Cartesian half-face tags
    accordingly. PyMRST stores these tags in ``g.cells.tags``.
    """
    out = deepcopy(g)
    out.nodes.coords = out.nodes.coords[:, [2, 0, 1]]
    if getattr(out.cells, "tags", None) is not None:
        tags = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
        new_tag = tags[:, [1, 2, 0]].ravel(order="F")
        old = np.asarray(out.cells.tags, dtype=np.int64)
        keep = (old >= 0) & (old < 6)
        mapped = old.copy()
        mapped[keep] = new_tag[old[keep]]
        out.cells.tags = mapped
    return out


def vertical_grdecl(grdecl: Dict, method: str = "all") -> Dict:
    """Straighten GRDECL pillars as MRST ``verticalGrdecl``."""
    out = dict(grdecl)
    xyz, _ = grdecl_xyz(grdecl)
    top = [0, 1]
    bot = [3, 4]
    if method == "sides":
        xyz[np.ix_(bot, [0], range(xyz.shape[2]))] = \
            xyz[np.ix_(top, [0], range(xyz.shape[2]))]
        xyz[np.ix_(bot, [xyz.shape[1] - 1], range(xyz.shape[2]))] = \
            xyz[np.ix_(top, [xyz.shape[1] - 1], range(xyz.shape[2]))]
        xyz[np.ix_(bot, range(xyz.shape[1]), [0])] = \
            xyz[np.ix_(top, range(xyz.shape[1]), [0])]
        xyz[np.ix_(bot, range(xyz.shape[1]), [xyz.shape[2] - 1])] = \
            xyz[np.ix_(top, range(xyz.shape[1]), [xyz.shape[2] - 1])]
    elif method == "all":
        xyz[top, :, :] = xyz[bot, :, :]
    else:
        raise ValueError(f"unknown vertical_grdecl method {method!r}")
    out["COORD"] = _flat(xyz)
    return out


def pad_grdecl(
    grdecl: Dict,
    dirs: Sequence[bool],
    box: np.ndarray,
    relative: bool = False,
) -> Dict:
    """Pad a corner-point grid with one cell layer in selected directions.

    This is a direct Python counterpart of MRST ``padGrdecl`` for the
    validated Cartesian/corner-point cases used by the vemmech examples.
    """
    dirs = np.asarray(dirs, dtype=bool).ravel()
    box = np.asarray(box, dtype=float).reshape(3, 2)
    gr = dict(grdecl)
    gr_new = dict(gr)

    xyz, zcorn = grdecl_xyz(gr)
    dims = np.asarray(gr["cartDims"], dtype=np.int64).ravel()

    if dirs[2]:
        top_pad = float(box[2, 0])
        bottom_pad = float(box[2, 1])
        cartdims = dims + np.array([0, 0, 2])
        znew = np.empty(tuple(2 * cartdims), dtype=float)
        znew[:, :, 2:-2] = zcorn
        znew[:, :, 1] = zcorn[:, :, 0]
        znew[:, :, 0] = np.nanmin(zcorn) - top_pad
        znew[:, :, -2] = zcorn[:, :, -1]
        znew[:, :, -1] = np.nanmax(zcorn) + bottom_pad
        gr_new = {
            "cartDims": cartdims,
            "COORD": np.asarray(gr["COORD"], dtype=float).copy(),
            "ZCORN": _flat(znew),
        }

    if not np.any(dirs[:2]):
        return gr_new

    gr = gr_new
    xyz, zcorn = grdecl_xyz(gr)
    vecx = xyz[[0, 1], -1, 0] - xyz[[0, 1], 0, 0]
    vecy = xyz[[0, 1], 0, -1] - xyz[[0, 1], 0, 0]
    vecx = vecx / np.linalg.norm(vecx)
    vecy = vecy - np.sum(vecx * vecy) * vecx
    vecy = vecy / np.linalg.norm(vecy)
    U = np.column_stack([vecx, vecy])
    shp = xyz.shape
    xyz[[0, 1], :, :] = (U.T @ xyz[[0, 1], :, :].reshape(2, -1)).reshape(
        (2, shp[1], shp[2]), order="C"
    )
    xyz[[3, 4], :, :] = (U.T @ xyz[[3, 4], :, :].reshape(2, -1)).reshape(
        (2, shp[1], shp[2]), order="C"
    )
    origin = xyz[[0, 1], :, :].reshape(2, -1).min(axis=1)
    xyz[[0, 1], :, :] -= origin[:, None, None]
    xyz[[3, 4], :, :] -= origin[:, None, None]
    org_max = xyz[[0, 1], :, :].reshape(2, -1).max(axis=1)
    org_min = xyz[[0, 1], :, :].reshape(2, -1).min(axis=1)

    dims = np.asarray(gr["cartDims"], dtype=np.int64).ravel()
    if dirs[0]:
        cartdims = dims + np.array([2, 0, 0])
        znew = np.empty(tuple(2 * cartdims), dtype=float)
        znew[2:-2, :, :] = zcorn
        znew[1, :, :] = zcorn[0, :, :]
        znew[0, :, :] = zcorn[0, :, :]
        znew[-2, :, :] = zcorn[-1, :, :]
        znew[-1, :, :] = zcorn[-1, :, :]
        xyz_new = np.empty((xyz.shape[0], xyz.shape[1] + 2, xyz.shape[2]))
        xyz_new[:, 1:-1, :] = xyz
        xyz_new[:, 0, :] = xyz[:, 0, :]
        xyz_new[:, -1, :] = xyz[:, -1, :]
        if relative:
            vec = np.array([1.0, 1.0])
            pos1 = np.array([-box[0, 0], 0.0])
            pos2 = np.array([box[1, 1], 0.0])
        else:
            vec = np.array([0.0, 1.0])
            pos1 = np.array([org_min[0] - box[0, 0], 0.0])
            pos2 = np.array([org_max[0] + box[0, 1], 0.0])
        xyz_new[[0, 1], 0, :] = xyz_new[[0, 1], 0, :] * vec[:, None] \
            + pos1[:, None]
        xyz_new[[0, 1], -1, :] = xyz_new[[0, 1], -1, :] * vec[:, None] \
            + pos2[:, None]
        xyz_new[[3, 4], 0, :] = xyz_new[[0, 1], 0, :]
        xyz_new[[3, 4], -1, :] = xyz_new[[0, 1], -1, :]
        gr_new = {"cartDims": cartdims, "COORD": _flat(xyz_new),
                  "ZCORN": _flat(znew)}
    else:
        gr_new = gr

    if dirs[1]:
        xyz, zcorn = grdecl_xyz(gr_new)
        dims = np.asarray(gr_new["cartDims"], dtype=np.int64).ravel()
        cartdims = dims + np.array([0, 2, 0])
        znew = np.empty(tuple(2 * cartdims), dtype=float)
        znew[:, 2:-2, :] = zcorn
        znew[:, 1, :] = zcorn[:, 0, :]
        znew[:, 0, :] = zcorn[:, 0, :]
        znew[:, -2, :] = zcorn[:, -1, :]
        znew[:, -1, :] = zcorn[:, -1, :]
        xyz_new = np.empty((xyz.shape[0], xyz.shape[1], xyz.shape[2] + 2))
        xyz_new[:, :, 1:-1] = xyz
        xyz_new[:, :, 0] = xyz[:, :, 0]
        xyz_new[:, :, -1] = xyz[:, :, -1]
        if relative:
            vec = np.array([1.0, 1.0])
            pos1 = np.array([0.0, -box[1, 0]])
            pos2 = np.array([0.0, box[1, 1]])
        else:
            vec = np.array([1.0, 0.0])
            pos1 = np.array([0.0, org_min[1] - box[1, 0]])
            pos2 = np.array([0.0, org_max[1] + box[1, 1]])
        xyz_new[[0, 1], :, 0] = xyz_new[[0, 1], :, 0] * vec[:, None] \
            + pos1[:, None]
        xyz_new[[0, 1], :, -1] = xyz_new[[0, 1], :, -1] * vec[:, None] \
            + pos2[:, None]
        xyz_new[[3, 4], :, 0] = xyz_new[[0, 1], :, 0]
        xyz_new[[3, 4], :, -1] = xyz_new[[0, 1], :, -1]
        gr_new = {"cartDims": cartdims, "COORD": _flat(xyz_new),
                  "ZCORN": _flat(znew)}

    return gr_new


if __name__ == "__main__":
    from pymrst.gridprocessing.compute_geometry import compute_geometry
    from pymrst.gridprocessing.process_grdecl import process_grdecl

    def _box_grdecl(nx=2, ny=2, nz=2):
        coord = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                coord.extend([i, j, 0.0, i + 0.1, j + 0.2, float(nz)])
        z = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            z[:, :, 2 * k] = k
            z[:, :, 2 * k + 1] = k + 1
        return {
            "cartDims": np.array([nx, ny, nz]),
            "COORD": np.asarray(coord, dtype=float),
            "ZCORN": z.ravel(order="F"),
        }

    def test_vertical_all():
        gr = _box_grdecl()
        out = vertical_grdecl(gr)
        xyz, _ = grdecl_xyz(out)
        assert np.allclose(xyz[[0, 1]], xyz[[3, 4]])
        print("ok test_vertical_all")

    def test_pad_z_dims():
        gr = _box_grdecl()
        out = pad_grdecl(gr, [False, False, True], np.ones((3, 2)))
        assert np.array_equal(out["cartDims"], [2, 2, 4])
        assert out["ZCORN"].size == 2 * 2 * 2 * 2 * 2 * 4
        print("ok test_pad_z_dims")

    def test_flip_grid_coords():
        gr = vertical_grdecl(_box_grdecl())
        g = compute_geometry(process_grdecl(gr))
        gf = flip_grid(g)
        assert np.allclose(gf.nodes.coords, g.nodes.coords[:, [2, 0, 1]])
        print("ok test_flip_grid_coords")

    test_vertical_all()
    test_pad_z_dims()
    test_flip_grid_coords()
    print("\nAll tests passed!")
