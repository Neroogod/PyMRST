"""The Norwegian CO2 Storage Atlas (MRST co2lab-common/atlasgrid).

Real formations, not synthetic domes. The Norwegian Petroleum Directorate
mapped the storage potential of the North Sea, the Norwegian Sea and the
Barents Sea, and published the top surface and the thickness of each
candidate formation as ESRI ASCII rasters. This turns a pair of those
rasters into a corner-point grid, which then goes through the ordinary
`process_grdecl` -> `top_surface_grid` -> `trap_analysis` chain like any
other model.

    grdecl = atlas_grid("Johansenfm", nz=1, coarsening=1)
    G  = compute_geometry(process_grdecl(grdecl))
    ts = top_surface_grid(G)
    res = trap_analysis(ts)              # the real Johansen traps

The data is the `CO2Atlas` dataset (a 16 MB download from SINTEF, unpacked
into MRST's data directory). Nothing here downloads it.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# formations whose top surface is stored with negative depths (North Sea and
# Norwegian Sea); Barents Sea rasters already carry positive depths
DEFAULT_ENV = "MRST_DATA_DIR"


def _atlas_dir(data_dir: Optional[str] = None) -> Path:
    """Where the CO2Atlas rasters live."""
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get(DEFAULT_ENV)
    if env:
        return Path(env) / "CO2Atlas"
    raise FileNotFoundError(
        "give data_dir=..., or set the MRST_DATA_DIR environment variable to "
        "MRST's data directory (the one holding CO2Atlas/)")


def read_aai_grid(path) -> Tuple[Dict[str, float], np.ndarray]:
    """Read an ESRI ASCII raster (MRST's readAAIGrid).

    Returns the header and the data as a (ncols, nrows) array with the
    no-data cells set to NaN.

MRST's reader is off by one at *both* ends, and the grids it produces
    depend on it, so both are reproduced here:

      - its header loop reads a line and only then discovers it is data, at
        which point it breaks - so **the first data row is consumed and
        thrown away**;
      - its data loop runs `while i < nrows`, one iteration short, so **the
        last row is never filled** and stays zero.

    The net effect is that the raster is shifted one row against the file. It
    then maps zeros to NaN, which quietly turns that empty last row - and any
    genuine zero, such as a formation of zero thickness - into no-data.

    Reading the file the obvious way instead puts every value one row out
    against MRST, and the resulting grids do not match.
    """
    meta: Dict[str, float] = {}
    rows = []
    with open(path, "r") as fh:
        # Header: lines of `key value`. The line that ends it is a data line,
        # and it is *discarded* - not seeked back over - because that is what
        # MRST does.
        while True:
            line = fh.readline()
            if not line:
                break
            parts = line.replace(",", ".").split()
            if len(parts) != 2:
                break                       # first data row: dropped, as MRST
            try:
                meta[parts[0]] = float(parts[1])
            except ValueError:
                break

        n_rows = int(meta["nrows"])
        n_cols = int(meta["ncols"])
        # and only nrows - 1 rows are read; the last stays zero -> NaN below
        for _ in range(n_rows - 1):
            line = fh.readline()
            if not line:
                break
            vals = np.fromstring(line.replace(",", "."), sep=" ")
            row = np.zeros(n_cols)
            row[:vals.size] = vals[:n_cols]
            rows.append(row)

    data = np.zeros((n_rows, n_cols))
    for i, r in enumerate(rows):
        data[i] = r

    data[data == meta["NODATA_value"]] = np.nan
    data[np.isinf(data)] = np.nan
    data[data == 0.0] = np.nan

    data = data[::-1, :]          # rasters run north -> south
    data = data.T                 # -> (ncols, nrows)
    meta["dims"] = (n_cols, n_rows)
    return meta, data


def read_atlas_dataset(name: str, coarsening: int = 1,
                       data_dir: Optional[str] = None
                       ) -> Tuple[Dict, np.ndarray]:
    """One Atlas raster, coarsened and with depths made positive.

    `name` is a file name such as "Johansenfm_top" or "Johansenfm_thickness".
    """
    meta, data = read_aai_grid(_atlas_dir(data_dir) / name)

    # North Sea and Norwegian Sea top surfaces are stored as negative depths;
    # Barents Sea ones are already positive. Flip so depth is always positive.
    if name.endswith("_top") and np.any(data[~np.isnan(data)] < 0):
        data = -data

    if coarsening > 1:
        data = data[::coarsening, ::coarsening]
    meta = dict(meta)
    meta["cellsize"] = meta["cellsize"] * coarsening
    meta["dims"] = data.shape
    return meta, data


def _interpolator(meta: Dict, data: np.ndarray):
    """Bilinear interpolation of a raster onto arbitrary (x, y)."""
    nx, ny = data.shape
    h = meta["cellsize"]
    xs = np.linspace(meta["xllcorner"], (nx - 1) * h + meta["xllcorner"], nx)
    ys = np.linspace(meta["yllcorner"], (ny - 1) * h + meta["yllcorner"], ny)
    return RegularGridInterpolator((xs, ys), data, method="linear",
                                   bounds_error=False, fill_value=np.nan)


def convert_atlas_to_3d(meta_thick: Dict, meta_top: Dict,
                        data_thick: np.ndarray, data_top: np.ndarray,
                        nz: int = 1) -> Dict:
    """Two rasters -> a corner-point GRDECL (MRST's convertAtlasTo3D).

    The top surface sets the caprock and the thickness raster is divided into
    nz layers below it. Cells touching a hole in the data are switched off:
    a cell needs all four of its corners, so the NaN mask is *dilated* by one
    before it is inverted into ACTNUM.
    """
    ndims = (int(meta_top["dims"][0]), int(meta_top["dims"][1]), nz + 1)
    dims = (ndims[0] - 1, ndims[1] - 1, nz)
    h = meta_top["cellsize"]
    xl, yl = meta_top["xllcorner"], meta_top["yllcorner"]

    xs = np.linspace(xl, dims[0] * h + xl, ndims[0])
    ys = np.linspace(yl, dims[1] * h + yl, ndims[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    pts = np.column_stack([X.ravel(), Y.ravel()])

    # The pillars *are* the top raster's own nodes - the grid is built from
    # meta_top's axes - so the top surface is sampled, not interpolated.
    # Interpolating it would be worse than pointless: a linear interpolator
    # forms sum(w_i v_i) over the corners of the containing cell, and
    # 0 * NaN = NaN, so a zero-weighted no-data neighbour poisons the value at
    # a node that has a perfectly good depth of its own. That wrongly voided
    # 792 cells around the edge of the formation.
    assert data_top.shape == (ndims[0], ndims[1]), (
        f"the grid is built on the top raster's nodes, so its shape must be "
        f"{(ndims[0], ndims[1])}, not {data_top.shape}")
    top = data_top.astype(float)

    # The thickness raster is a different, coarser one, so it does have to be
    # interpolated onto these points.
    f_thick = _interpolator(meta_thick, data_thick)
    thick = f_thick(pts).reshape(ndims[0], ndims[1]) / dims[2]
    thick = np.where(thick <= 0, np.nan, thick)

    Z = np.empty((ndims[0], ndims[1], ndims[2]))
    for k in range(ndims[2]):
        Z[:, :, k] = top + thick * k
    Z = np.sort(Z, axis=2)

    # COORD: one pillar per areal node, from the top layer to the bottom.
    # The pillars run x-fastest (MATLAB's column-major reshape), so these have
    # to be Fortran-ordered. Ravel them C-order and the pillars come out
    # transposed: the grid still builds, with finite coordinates throughout,
    # but its cells are scrambled - faces come out a hundred times too large
    # and a tenth of the cells collapse to zero volume.
    n = ndims[0] * ndims[1]
    lines = np.zeros((n, 6))
    xf, yf = X.ravel(order="F"), Y.ravel(order="F")
    lines[:, 0] = xf
    lines[:, 3] = xf
    lines[:, 1] = yf
    lines[:, 4] = yf
    lines[:, 2] = Z[:, :, 0].ravel(order="F")
    lines[:, 5] = Z[:, :, -1].ravel(order="F")
    coord = lines.ravel()

    # ZCORN: each cell corner is repeated, so index [1,2,2,3,3,...,n,n+1]
    def dup(d):
        return np.arange(1, 2 * dims[d] + 1) // 2

    z = Z[np.ix_(dup(0), dup(1), dup(2))]
    zcorn = z.ravel(order="F")

    # ACTNUM: a cell is off if any of the four corners it needs is missing.
    # Dilating the NaN mask by one node and inverting does exactly that.
    # The mask is taken over every layer rather than the top alone (MRST looks
    # only at the top); the two agree cell for cell, because the thickness
    # raster covers everything the top raster does.
    nan_mask = np.isnan(Z).any(axis=2)
    grown = nan_mask.copy()
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            grown |= np.roll(np.roll(nan_mask, i, axis=0), j, axis=1)
    active2d = ~grown[:-1, :-1]
    actnum = np.repeat(active2d[:, :, None], dims[2], axis=2)

    # MRST writes `inf` into the pillars of the unmapped regions. Those
    # pillars carry only *inactive* cells, so their value is arbitrary - but
    # PyMRST's process_grdecl builds every node before dropping the inactive
    # cells, and an inf there poisons the node coordinates. Fill them with a
    # finite depth instead. The active grid is identical either way.
    fill = float(np.nanmean(Z[:, :, 0])) if np.any(~np.isnan(Z)) else 0.0
    zcorn = np.where(np.isnan(zcorn), fill, zcorn)
    coord = np.where(np.isnan(coord), fill, coord)

    return {
        "cartDims": np.asarray(dims, dtype=np.int64),
        "COORD": coord,
        "ZCORN": zcorn,
        "ACTNUM": actnum.ravel(order="F").astype(np.int32),
    }


def atlas_grid(formation: str, nz: int = 1, coarsening: int = 1,
               data_dir: Optional[str] = None) -> Dict:
    """A corner-point GRDECL for one Atlas formation (MRST's getAtlasGrid).

    `formation` is the base name, e.g. "Johansenfm" or "Utsirafm"; the reader
    looks for `<formation>_top` and `<formation>_thickness`.
    """
    meta_top, data_top = read_atlas_dataset(
        f"{formation}_top", coarsening, data_dir)
    meta_thick, data_thick = read_atlas_dataset(
        f"{formation}_thickness", coarsening, data_dir)
    return convert_atlas_to_3d(meta_thick, meta_top, data_thick, data_top, nz)


def available_formations(data_dir: Optional[str] = None) -> list:
    """Formations with both a top and a thickness raster."""
    d = _atlas_dir(data_dir)
    tops = {p.name[:-4] for p in d.iterdir() if p.name.endswith("_top")}
    thick = {p.name[:-10] for p in d.iterdir()
             if p.name.endswith("_thickness")}
    return sorted(tops & thick)


if __name__ == "__main__":
    import pytest

    from pymrst.gridprocessing import compute_geometry
    from pymrst.gridprocessing.process_grdecl import process_grdecl
    from pymrst.modules.co2lab_spillpoint.trapping import trap_analysis, volumes_of_traps
    from pymrst.modules.co2lab_ve.ve import top_surface_grid

    try:
        DIR = _atlas_dir()
        HAVE = (DIR / "Johansenfm_top").exists()
    except FileNotFoundError:
        HAVE = False

    if not HAVE:
        print("CO2Atlas data not found; set MRST_DATA_DIR to MRST's data "
              "directory (the one holding CO2Atlas/). Skipping.")
        raise SystemExit(0)

    def test_reader_matches_the_header():
        """The raster reads back at the size its own header declares."""
        meta, data = read_aai_grid(DIR / "Johansenfm_top")
        assert data.shape == (int(meta["ncols"]), int(meta["nrows"]))
        assert meta["cellsize"] == 200.0
        # the data is depth, and mostly missing (the formation is a patch in
        # a rectangular raster)
        good = ~np.isnan(data)
        assert 0.05 < good.mean() < 0.9, good.mean()
        # raw North Sea tops are negative depths
        assert np.all(data[good] < 0), "expected negative depths in the raw file"

        # the last row is the reader's phantom one, and must be all NaN
        assert np.all(np.isnan(data[:, 0])), \
            "MRST's reader leaves one row short; that row must come back NaN"
        print(f"ok test_reader_matches_the_header "
              f"({data.shape[0]}x{data.shape[1]} cells, "
              f"{good.mean():.0%} of them mapped)")

    def test_depths_are_made_positive():
        """read_atlas_dataset flips the sign so depth increases downward."""
        _, raw = read_aai_grid(DIR / "Johansenfm_top")
        _, data = read_atlas_dataset("Johansenfm_top")
        good = ~np.isnan(data)
        assert np.all(data[good] > 0), "depths should be positive"
        assert np.allclose(data[good], -raw[~np.isnan(raw)])
        # Johansen sits around 2-3 km down
        assert 1000.0 < np.nanmin(data) < 4000.0
        print(f"ok test_depths_are_made_positive "
              f"(Johansen caprock {np.nanmin(data):.0f}-{np.nanmax(data):.0f} m)")

    def test_johansen_builds_a_real_grid():
        """The full chain: rasters -> GRDECL -> grid -> top surface -> traps."""
        grdecl = atlas_grid("Johansenfm", nz=1, coarsening=1)
        g = compute_geometry(process_grdecl(grdecl))

        assert g.cells.num > 1000, g.cells.num
        # the inactive cells were dropped, so far fewer than the raster's cells
        nx, ny, nz = grdecl["cartDims"]
        assert g.cells.num < nx * ny * nz

        # a real formation: kilometres across, hundreds of metres thick
        extent = np.ptp(g.cells.centroids[:, :2], axis=0)
        assert np.all(extent > 20e3), extent
        depth = g.cells.centroids[:, 2]
        assert 1500.0 < depth.min() < 4000.0, depth.min()

        ts = top_surface_grid(g)
        res = trap_analysis(ts)
        assert res.n_traps > 0, "the real Johansen formation should have traps"
        vols = volumes_of_traps(ts, res, poro=np.full(ts.g.cells.num, 0.2))
        print(f"ok test_johansen_builds_a_real_grid "
              f"({g.cells.num} active cells, {extent[0] / 1e3:.0f}x"
              f"{extent[1] / 1e3:.0f} km, {res.n_traps} traps holding "
              f"{vols.sum() / 1e9:.2f} km3 of pore space)")

    def test_coarsening_shrinks_the_grid():
        """Coarsening subsamples the raster, giving proportionally fewer cells."""
        fine = atlas_grid("Johansenfm", nz=1, coarsening=1)
        coarse = atlas_grid("Johansenfm", nz=1, coarsening=3)

        nf = np.prod(fine["cartDims"])
        ncrs = np.prod(coarse["cartDims"])
        assert ncrs < nf / 5, (nf, ncrs)

        gf = process_grdecl(fine)
        gc = process_grdecl(coarse)
        assert gc.cells.num < gf.cells.num / 5
        print(f"ok test_coarsening_shrinks_the_grid "
              f"({gf.cells.num} -> {gc.cells.num} active cells at 3x)")

    def test_nz_layers_stack_below_the_caprock():
        """More layers means more cells, and the caprock does not move."""
        one = process_grdecl(atlas_grid("Johansenfm", nz=1, coarsening=4))
        five = process_grdecl(atlas_grid("Johansenfm", nz=5, coarsening=4))
        assert five.cells.num == 5 * one.cells.num

        # the top of the model is the same surface either way
        z1 = compute_geometry(one).nodes.coords[:, 2].min()
        z5 = compute_geometry(five).nodes.coords[:, 2].min()
        assert abs(z1 - z5) < 1e-6, (z1, z5)
        print(f"ok test_nz_layers_stack_below_the_caprock "
              f"({one.cells.num} -> {five.cells.num} cells; "
              f"caprock stays at {z1:.0f} m)")

    def test_available_formations():
        """The dataset holds a good number of complete formations."""
        names = available_formations()
        assert "Johansenfm" in names
        assert len(names) >= 5, names
        print(f"ok test_available_formations ({len(names)}: "
              f"{', '.join(names[:6])}...)")

    test_reader_matches_the_header()
    test_depths_are_made_positive()
    test_available_formations()
    test_johansen_builds_a_real_grid()
    test_coarsening_shrinks_the_grid()
    test_nz_layers_stack_below_the_caprock()
    print("\nAll tests passed!")
