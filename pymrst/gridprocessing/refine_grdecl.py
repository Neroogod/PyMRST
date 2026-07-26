"""Refine a corner-point GRDECL deck (MRST `refineGrdecl` / `grdeclXYZ`).

Subdivides each cell of a corner-point (COORD/ZCORN) deck by integer factors
`(nx, ny, nz)` in the logical directions, interpolating the pillar geometry
linearly so the refined grid fills exactly the same volume. Cell properties
(PERMX/PORO/ACTNUM/...) are inherited by each child from its parent. This is the
standard way to study grid-resolution effects on a real corner-point model
without re-meshing.

`grdecl_xyz` is the small companion that unpacks the flat COORD/ZCORN vectors
into the structured `(6, nx+1, ny+1)` pillar array and `(2nx, 2ny, 2nz)` corner
array MRST works with.
"""

from typing import Dict, Sequence, Tuple

import numpy as np


def grdecl_xyz(grdecl: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack COORD/ZCORN into `(xyz, zcorn)` structured arrays (MRST
    `grdeclXYZ`)."""
    nx, ny, nz = (int(d) for d in grdecl["cartDims"])
    xyz = np.reshape(np.asarray(grdecl["COORD"], dtype=float),
                     (6, nx + 1, ny + 1), order="F")
    zcorn = np.reshape(np.asarray(grdecl["ZCORN"], dtype=float),
                       (2 * nx, 2 * ny, 2 * nz), order="F")
    return xyz, zcorn


def _refine_z(zcorn: np.ndarray, nz: int) -> np.ndarray:
    dz = zcorn[:, :, 1::2] - zcorn[:, :, 0::2]
    base = zcorn[:, :, 0::2]
    out = np.zeros((zcorn.shape[0], zcorn.shape[1], zcorn.shape[2] * nz))
    for i in range(nz):
        out[:, :, 2 * i::2 * nz] = base + i * dz / nz
        out[:, :, 2 * i + 1::2 * nz] = base + (i + 1) * dz / nz
    return out


def refine_grdecl(grdecl_in: Dict, dim: Sequence[int]) -> Dict:
    """Return a GRDECL deck refined by factors `dim = (nx, ny, nz)`."""
    nx, ny, nz = (int(d) for d in dim)
    xyz, zcorn = grdecl_xyz(grdecl_in)

    if nz > 1:
        zcorn = _refine_z(zcorn, nz)

    if nx > 1:
        dx = xyz[:, 1:, :] - xyz[:, :-1, :]
        base = xyz[:, :-1, :]
        end = xyz[:, -1:, :]
        new = np.zeros((6, (xyz.shape[1] - 1) * nx + 1, xyz.shape[2]))
        for i in range(1, nx + 1):
            new[:, i - 1:-1:nx, :] = base + (i - 1) * dx / nx
        new[:, -1:, :] = end
        xyz = new

        dz = zcorn[1::2, :, :] - zcorn[0::2, :, :]
        base = zcorn[0::2, :, :]
        out = np.zeros((zcorn.shape[0] * nx, zcorn.shape[1], zcorn.shape[2]))
        for i in range(nx):
            out[2 * i::2 * nx, :, :] = base + i * dz / nx
            out[2 * i + 1::2 * nx, :, :] = base + (i + 1) * dz / nx
        zcorn = out

    if ny > 1:
        dy = xyz[:, :, 1:] - xyz[:, :, :-1]
        base = xyz[:, :, :-1]
        end = xyz[:, :, -1:]
        new = np.zeros((6, xyz.shape[1], (xyz.shape[2] - 1) * ny + 1))
        for i in range(1, ny + 1):
            new[:, :, i - 1:-1:ny] = base + (i - 1) * dy / ny
        new[:, :, -1:] = end
        xyz = new

        dz = zcorn[:, 1::2, :] - zcorn[:, 0::2, :]
        base = zcorn[:, 0::2, :]
        out = np.zeros((zcorn.shape[0], zcorn.shape[1] * ny, zcorn.shape[2]))
        for i in range(ny):
            out[:, 2 * i::2 * ny, :] = base + i * dz / ny
            out[:, 2 * i + 1::2 * ny, :] = base + (i + 1) * dz / ny
        zcorn = out

    cart_in = np.asarray(grdecl_in["cartDims"], dtype=np.int64)
    grdecl: Dict = {
        "cartDims": cart_in * np.array([nx, ny, nz]),
        "COORD": xyz.ravel(order="F"),
        "ZCORN": zcorn.ravel(order="F"),
    }

    # inherit cell properties: each child takes its parent's value
    ix = np.repeat(np.arange(cart_in[0]), nx)
    iy = np.repeat(np.arange(cart_in[1]), ny)
    iz = np.repeat(np.arange(cart_in[2]), nz)
    fields = ["ACTNUM", "PERMX", "PERMY", "PERMZ", "PORO", "MULTZ", "MULTX",
              "MULTY", "EQLNUM", "FLUXNUM", "NTG"]
    for f in fields:
        if f in grdecl_in:
            a = np.reshape(np.asarray(grdecl_in[f]), cart_in, order="F")
            grdecl[f] = a[np.ix_(ix, iy, iz)].ravel(order="F")
    return grdecl


if __name__ == "__main__":
    def _box_grdecl(nx, ny, nz):
        """A unit-spaced Cartesian box as a corner-point deck."""
        # pillars: vertical lines at each (i,j) corner, x=i, y=j, z from 0..nz
        xyz = np.zeros((6, nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                xyz[:, i, j] = [i, j, 0.0, i, j, float(nz)]
        # zcorn: each cell spans [k, k+1] in z; corners doubled per direction
        zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            zc[:, :, 2 * k] = k
            zc[:, :, 2 * k + 1] = k + 1
        return {"cartDims": np.array([nx, ny, nz]),
                "COORD": xyz.ravel(order="F"), "ZCORN": zc.ravel(order="F")}

    def test_cartdims_scale_by_the_factors():
        g = _box_grdecl(2, 2, 2)
        r = refine_grdecl(g, (2, 1, 3))
        assert list(r["cartDims"]) == [4, 2, 6], r["cartDims"]
        assert r["ZCORN"].size == np.prod(2 * r["cartDims"])
        assert r["COORD"].size == 6 * (r["cartDims"][0] + 1) * (r["cartDims"][1] + 1)
        print(f"ok test_cartdims_scale_by_the_factors ({list(r['cartDims'])})")

    def test_z_refinement_halves_layer_thickness():
        """Refining a 1-layer box by nz=2 gives two layers of half thickness,
        z-corners at 0, 0.5, 1."""
        g = _box_grdecl(1, 1, 1)
        r = refine_grdecl(g, (1, 1, 2))
        _, zc = grdecl_xyz(r)
        assert np.isclose(zc[0, 0, 0], 0.0)
        assert np.isclose(zc[0, 0, 1], 0.5)
        assert np.isclose(zc[0, 0, 3], 1.0)
        print("ok test_z_refinement_halves_layer_thickness")

    def test_properties_are_inherited_by_children():
        g = _box_grdecl(2, 1, 1)
        g["PERMX"] = np.array([10.0, 20.0])          # two parent cells
        r = refine_grdecl(g, (2, 1, 1))              # split each in x -> 4 cells
        assert np.allclose(r["PERMX"], [10.0, 10.0, 20.0, 20.0]), r["PERMX"]
        print("ok test_properties_are_inherited_by_children")

    def test_refinement_preserves_total_extent():
        g = _box_grdecl(2, 2, 2)
        r = refine_grdecl(g, (2, 2, 2))
        xin, zin = grdecl_xyz(g)
        xr, zr = grdecl_xyz(r)
        assert np.isclose(zr.min(), zin.min()) and np.isclose(zr.max(), zin.max())
        assert np.isclose(xr[0].max(), xin[0].max())   # same x extent
        print("ok test_refinement_preserves_total_extent")

    test_cartdims_scale_by_the_factors()
    test_z_refinement_halves_layer_thickness()
    test_properties_are_inherited_by_children()
    test_refinement_preserves_total_extent()
    print("\nAll tests passed!")
