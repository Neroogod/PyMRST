"""MRST ``padGrdecl.m`` corner-point grid padding."""

from typing import Dict, Sequence

import numpy as np

from pymrst.modules.vemmech.grid_utils import _flat, grdecl_xyz


def pad_grdecl(
    grdecl: Dict,
    dirs: Sequence[bool],
    box: np.ndarray,
    relative: bool = False,
) -> Dict:
    """Pad a corner-point grid with one cell layer in selected directions."""
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

__all__ = ["pad_grdecl"]
