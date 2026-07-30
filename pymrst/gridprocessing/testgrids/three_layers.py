"""MRST ``threeLayers`` mapped layered fixture."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import (
    coord_from_xyz,
    corner_indices,
)


def three_layers(nx, ny, nz):
    nx = np.repeat(int(nx), 2) if np.asarray(nx).size == 1 else np.asarray(nx)
    nz = np.repeat(int(nz), 3) if np.asarray(nz).size == 1 else np.asarray(nz)
    if nx.size != 2 or nz.size != 3:
        raise ValueError("nx must have 1 or 2 entries and nz 1 or 3")
    x = np.unique(np.concatenate([
        np.linspace(-1, 0, int(nx[0]) + 1),
        np.linspace(0, 1, int(nx[1]) + 1),
    ]))
    y_axis = np.linspace(0, 1, int(ny) + 1)
    z_axis = np.unique(np.concatenate([
        np.linspace(0, .25, int(nz[0]) + 1),
        np.linspace(.25, .75, int(nz[1]) + 1),
        np.linspace(.75, 1, int(nz[2]) + 1),
    ]))
    xg, yg, zg = np.meshgrid(x, y_axis, z_axis, indexing="ij")
    dims = np.array([x.size - 1, y_axis.size - 1, z_axis.size - 1])
    ix = [corner_indices(int(n)) for n in dims]
    yc = yg[np.ix_(*ix)]
    zc = zg[np.ix_(*ix)]
    zc[dims[0]:, :, :] -= (zc[dims[0]:, :, :] + 1) * yc[dims[0]:, :, :] / 2
    permx = np.zeros(tuple(dims), dtype=float)
    cuts = np.cumsum(nz.astype(int))
    permx[:, :, :cuts[0]] = 100
    permx[:, :, cuts[0]:cuts[1]] = 1
    permx[:, :, cuts[1]:cuts[2]] = 100
    return {
        "cartDims": dims,
        "COORD": coord_from_xyz(xg, yg, zg),
        "ZCORN": zc.ravel(order="F"),
        "ACTNUM": np.ones(int(np.prod(dims)), dtype=np.int32),
        "PERMX": permx.ravel(order="F"),
    }


threeLayers = three_layers
