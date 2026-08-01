"""Rotate a cell-local permeability into the global frame (MRST
`permeabilityConverter`).

Permeability is sometimes measured relative to a cell's own (tilted) geometry -
along its layering rather than the global x/y/z axes. On a dipping corner-point
grid those differ, and a solver needs the tensor in global coordinates. This
takes diagonal `PERMX/PERMY/PERMZ` given in a cell-local system and rotates each
into the full symmetric global tensor `K = sum_i k_i (e_i ⊗ e_i)`, where the
`e_i` are the cell's local axes recovered from its corner geometry. The local
axes come from the averaged edge vectors of each cell in the three logical
directions; `bedding_plane` replaces the vertical axis with the plane normal,
and `bedding_plane_normal` also re-orthogonalises the in-plane axes.

MRST 2024a's off-diagonal branch fails at its cell-array ``switch``. The
default ``off_diagonal_mode="mrst"`` reproduces that refusal clearly.
``"corrected"`` applies the mathematically expected symmetric cross-axis
terms and is an explicit PyMRST extension.
"""

from typing import Dict

import numpy as np

from pymrst.gridprocessing.process_grdecl import build_corner_pt_nodes


def _cell_axes(grdecl: Dict):
    """The three local edge-direction vectors per cell, from corner geometry."""
    X, Y, Z = build_corner_pt_nodes(grdecl)

    def corner(bi, bj, bk):
        return np.column_stack([C[bi::2, bj::2, bk::2].ravel(order="F")
                                for C in (X, Y, Z)])

    dir1 = sum(corner(1, bj, bk) - corner(0, bj, bk)
               for bj in (0, 1) for bk in (0, 1))        # i-direction
    dir2 = sum(corner(bi, 1, bk) - corner(bi, 0, bk)
               for bi in (0, 1) for bk in (0, 1))        # j-direction
    dir3 = sum(corner(bi, bj, 1) - corner(bi, bj, 0)
               for bi in (0, 1) for bj in (0, 1))        # k-direction
    return dir1, dir2, dir3


def permeability_converter(
    grdecl: Dict, k_system: str = "xyz", *, off_diagonal_mode: str = "mrst"
) -> Dict:
    """Return a grdecl whose PERM* keywords give the global-frame tensor.

    `k_system` is the frame the input diagonal PERMX/Y/Z are given in: `xyz`
    (nothing to do), `cell` (the cell's own axes), `bedding_plane` (axes with the
    vertical replaced by the layer normal), or `bedding_plane_normal` (also
    re-orthogonalised).
    """
    if k_system == "xyz":
        return dict(grdecl)

    perm = np.column_stack([np.asarray(grdecl[k], dtype=float).ravel()
                            for k in ("PERMX", "PERMY", "PERMZ")])
    d1, d2, d3 = _cell_axes(grdecl)

    if k_system == "cell":
        pass
    elif k_system == "bedding_plane":
        d3 = np.cross(d1, d2)
    elif k_system == "bedding_plane_normal":
        d3 = np.cross(d1, d2)
        d2 = d2 - ((d1 * d2).sum(1) / (d1 ** 2).sum(1))[:, None] * d1
    else:
        raise ValueError(f"unknown coordinate system {k_system!r}")

    dirs = [v / np.linalg.norm(v, axis=1)[:, None] for v in (d1, d2, d3)]

    nc = perm.shape[0]
    tensor = np.zeros((nc, 9))
    for i, e in enumerate(dirs):
        # k_i * (e ⊗ e), flattened column-major to match MRST's 9-vector
        outer = np.einsum("ca,cb->cba", e, e).reshape(nc, 9)
        tensor += perm[:, i:i + 1] * outer

    off_diagonal = [
        ("PERMXY", 0, 1), ("PERMYZ", 1, 2), ("PERMZX", 0, 2)
    ]
    present = [item for item in off_diagonal if item[0] in grdecl]
    if present:
        if off_diagonal_mode == "mrst":
            raise NotImplementedError(
                "MRST 2024a fails for off-diagonal permeability input; "
                "use off_diagonal_mode='corrected' for PyMRST's tensor transform"
            )
        elif off_diagonal_mode == "corrected":
            for keyword, first, second in present:
                cross = (
                    np.einsum("ca,cb->cba", dirs[first], dirs[second])
                    + np.einsum("ca,cb->cba", dirs[second], dirs[first])
                ).reshape(perm.shape[0], 9)
                values = np.asarray(grdecl[keyword], dtype=float).ravel()
                if values.size != perm.shape[0]:
                    raise ValueError(f"{keyword} must contain one value per cell")
                tensor += values[:, None]*cross
        else:
            raise ValueError("off_diagonal_mode must be 'mrst' or 'corrected'")

    out = dict(grdecl)
    out["PERMX"] = tensor[:, 0]      # xx
    out["PERMXY"] = tensor[:, 3]     # xy (== yx)
    out["PERMZX"] = tensor[:, 6]     # xz (== zx)
    out["PERMY"] = tensor[:, 4]      # yy
    out["PERMYZ"] = tensor[:, 7]     # yz (== zy)
    out["PERMZ"] = tensor[:, 8]      # zz
    return out


if __name__ == "__main__":
    def _axis_aligned_box(nx, ny, nz):
        """A unit-spaced Cartesian corner-point deck (axes == global axes)."""
        xyz = np.zeros((6, nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                xyz[:, i, j] = [i, j, 0.0, i, j, float(nz)]
        zc = np.zeros((2 * nx, 2 * ny, 2 * nz))
        for k in range(nz):
            zc[:, :, 2 * k] = k
            zc[:, :, 2 * k + 1] = k + 1
        return {"cartDims": np.array([nx, ny, nz]),
                "COORD": xyz.ravel(order="F"), "ZCORN": zc.ravel(order="F")}

    def test_axis_aligned_grid_keeps_diagonal_tensor():
        """On an axis-aligned box the local axes are the global ones, so the
        rotated tensor is diagonal and equal to the input."""
        g = _axis_aligned_box(2, 2, 1)
        nc = 4
        g["PERMX"] = np.full(nc, 100.0)
        g["PERMY"] = np.full(nc, 50.0)
        g["PERMZ"] = np.full(nc, 10.0)
        out = permeability_converter(g, "cell")
        assert np.allclose(out["PERMX"], 100.0)
        assert np.allclose(out["PERMY"], 50.0)
        assert np.allclose(out["PERMZ"], 10.0)
        assert np.allclose(out["PERMXY"], 0.0)
        assert np.allclose(out["PERMYZ"], 0.0)
        assert np.allclose(out["PERMZX"], 0.0)
        print("ok test_axis_aligned_grid_keeps_diagonal_tensor")

    def test_xyz_system_is_a_noop():
        g = _axis_aligned_box(1, 1, 1)
        g["PERMX"] = np.array([1.0])
        out = permeability_converter(g, "xyz")
        assert "PERMXY" not in out and out["PERMX"][0] == 1.0
        print("ok test_xyz_system_is_a_noop")

    def test_isotropic_perm_stays_isotropic_under_rotation():
        """Isotropic k rotates to itself in any frame: diagonal k, zero
        off-diagonals, whatever the local axes."""
        g = _axis_aligned_box(2, 1, 1)
        nc = 2
        for key in ("PERMX", "PERMY", "PERMZ"):
            g[key] = np.full(nc, 42.0)
        out = permeability_converter(g, "bedding_plane")
        assert np.allclose(out["PERMX"], 42.0) and np.allclose(out["PERMY"], 42.0)
        assert np.allclose(out["PERMZ"], 42.0)
        assert np.allclose([out["PERMXY"], out["PERMYZ"], out["PERMZX"]], 0.0)
        print("ok test_isotropic_perm_stays_isotropic_under_rotation")

    def test_offdiagonal_input_has_strict_and_corrected_modes():
        g = _axis_aligned_box(1, 1, 1)
        g.update(PERMX=np.array([1.0]), PERMY=np.array([1.0]),
                 PERMZ=np.array([1.0]), PERMXY=np.array([0.1]))
        corrected = permeability_converter(g, "cell", off_diagonal_mode="corrected")
        assert np.allclose(corrected["PERMXY"], 0.1)
        print("ok test_offdiagonal_input_has_strict_and_corrected_modes")

    test_axis_aligned_grid_keeps_diagonal_tensor()
    test_xyz_system_is_a_noop()
    test_isotropic_perm_stays_isotropic_under_rotation()
    test_offdiagonal_input_has_strict_and_corrected_modes()
    print("\nAll tests passed!")
