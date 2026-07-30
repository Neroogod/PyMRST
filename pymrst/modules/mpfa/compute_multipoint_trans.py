"""Multi-point flux approximation transmissibility (MRST `computeMultiPointTrans`).

TPFA needs the grid to be K-orthogonal: a face flux depends only on the pressure
drop between the two cells it separates. On a skewed grid, or with a full/rotated
permeability tensor, that is wrong - the flux genuinely depends on more than two
cell pressures. MPFA fixes this by building, around each grid *node*, a local
interaction region and solving a small system there so a face flux is a *multi*-
point combination of the surrounding cell pressures. This is MRST's local
flux-mimetic MPFA-O assembly: enumerate the sub-half-faces, form the local mixed
inner product `B = R (N K)^-1` on each node region, invert the node-block mass
matrix, and assemble the transmissibility operators.

Returns a dict of operators - `cf_trans` (K·grad on the sub-faces from cell and
boundary pressures), `e_div` (the divergence), `cf_trans_g` (gravity), and the
book-keeping (`d1`, `R`, `cno`, `counts`, `sb`) - exactly the pieces `incompMPFA`
consumes.
"""

from typing import Dict

import numpy as np
import scipy.sparse as sp

from pymrst.params.rock.perm_tensor import perm_tensor
from pymrst.utils.gridtools.rlencode import rlencode


def _create_mapping(g):
    """Enumerate sub-half-faces, mapping each to its cell, node, half-face, face
    and sub-face (MRST `createMapping`)."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    cells_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()

    cellno = np.repeat(np.arange(g.cells.num), np.diff(face_pos))
    col = (cellno == neigh[cells_faces, 1]).astype(np.int64)     # 0 or 1
    nhf = face_pos[-1]
    hfaces = np.zeros((g.faces.num, 2), dtype=np.int64)
    hfaces[cells_faces, col] = np.arange(nhf)
    n_per_face = np.diff(node_pos)
    hfaces = np.repeat(hfaces, n_per_face, axis=0)               # (Nfn, 2)

    cells = np.repeat(neigh, n_per_face, axis=0)                 # (Nfn, 2)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    face_of = np.repeat(np.arange(g.faces.num), n_per_face)
    nfn = fnodes.size

    cells_f = np.concatenate([cells[:, 0], cells[:, 1]])
    nodes_f = np.concatenate([fnodes, fnodes])
    hfaces_f = np.concatenate([hfaces[:, 0], hfaces[:, 1]])
    faces_f = np.concatenate([face_of, face_of])
    subf_f = np.concatenate([np.arange(nfn), np.arange(nfn)])

    keep = cells_f != -1                                        # -1 == outside
    w = np.column_stack([cells_f, nodes_f, hfaces_f, faces_f, subf_f])[keep]
    w = w[np.lexsort([w[:, 4], w[:, 3], w[:, 2], w[:, 1], w[:, 0]])]
    return w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4], np.arange(w.shape[0])


def _block_diag(vec, n_sub, dims):
    """Assemble a `(n_sub, n_sub)` block-diagonal sparse from per-row `dims`
    values (each block `dims x dims`)."""
    row = np.repeat(np.arange(n_sub), dims)
    block = np.repeat(np.arange(n_sub) // dims, dims) * dims
    col = block + np.tile(np.arange(dims), n_sub)
    return sp.csr_array((np.asarray(vec).ravel(), (row, col)), shape=(n_sub, n_sub))


def _invert_blocks(A: sp.spmatrix, sizes: np.ndarray) -> sp.csr_array:
    """Invert the consecutive diagonal blocks of `A` of the given `sizes`."""
    A = sp.csr_array(A)
    rows, cols, data = [], [], []
    off = 0
    for s in sizes:
        s = int(s)
        blk = A[off:off + s, off:off + s].toarray()
        inv = np.linalg.inv(blk)
        r, c = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
        rows.append((r + off).ravel())
        cols.append((c + off).ravel())
        data.append(inv.ravel())
        off += s
    n = int(np.sum(sizes))
    return sp.csr_array((np.concatenate(data),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n))


def _node_ip_general(areas, volume, N, R, K):
    """MRST tensor assembly's stabilized inner product for a general corner."""
    dim = K.shape[0]
    nface = N.shape[0]
    trace_mean = np.trace(K) / dim
    scale = 6.0 * trace_mean * np.sum(areas) / nface
    Ns = N / scale
    U, singular, Vh = np.linalg.svd(Ns, full_matrices=True)
    singular = singular * scale
    if singular.size < dim or np.any(singular[:dim] <= 0):
        raise np.linalg.LinAlgError(
            "cannot assemble MPFA at degenerate/coplanar cell corner"
        )
    H = np.zeros((dim, nface))
    H[np.arange(dim), np.arange(dim)] = 1.0 / singular[:dim]
    main = R @ Vh.T @ H @ U.T

    stabilization = np.zeros((nface, nface))
    if nface > dim:
        stabilization[dim:, dim:] = np.eye(nface - dim)
    weighted_u = areas[:, None] * U
    t = 6.0 * volume ** (5.0 / 3.0) * trace_mean
    return main + (weighted_u @ stabilization @ weighted_u.T) / t


def _local_flux_mimetic_ip(
    g, rock, cno, fno, nno, subhfno, eta=0.0, general=False
):
    """The mixed inner product `B = R (N K)^-1` and the R vectors."""
    dims = g.nodes.coords.shape[1]
    _, blocksz = rlencode(np.column_stack([cno, nno]))
    if not general:
        assert np.all(blocksz == dims), "each node region must have `dim` sub-faces"
    n_sub = subhfno.size

    fc = np.asarray(g.faces.centroids, dtype=float)
    cc = np.asarray(g.cells.centroids, dtype=float)
    nx = np.asarray(g.nodes.coords, dtype=float)
    R_vec = fc[fno] - cc[cno] + eta * (nx[nno] - fc[fno])
    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    sgn = 2 * (cno == neigh[fno, 0]) - 1
    numnodes = np.diff(np.asarray(g.faces.nodePos))
    N_vec = (np.asarray(g.faces.normals, dtype=float)[fno]
             * sgn[:, None] / numnodes[fno][:, None])
    k = perm_tensor(rock, dims)[0]                              # (ncells, dims^2)
    if general:
        rows, cols, values = [], [], []
        offset = 0
        for size in blocksz:
            size = int(size)
            ix = slice(offset, offset + size)
            Kcell = k[cno[offset]].reshape(dims, dims)
            local = _node_ip_general(
                np.asarray(g.faces.areas)[fno[ix]],
                float(np.asarray(g.cells.volumes)[cno[offset]]),
                N_vec[ix] @ Kcell,
                R_vec[ix],
                Kcell,
            )
            rr, cc = np.meshgrid(
                np.arange(offset, offset + size),
                np.arange(offset, offset + size),
                indexing="ij",
            )
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            values.append(local.ravel())
            offset += size
        B = sp.csr_array(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_sub, n_sub),
        )
        return B, R_vec

    R = _block_diag(R_vec, n_sub, dims)
    N = _block_diag(N_vec, n_sub, dims)
    # each of a node-block's `dims` sub-faces takes one row of its cell's tensor
    local = np.arange(n_sub) % dims
    k_rows = k[cno].reshape(n_sub, dims, dims)[np.arange(n_sub), local, :]
    K = _block_diag(k_rows, n_sub, dims)

    sz = np.full(n_sub // dims, dims)
    return R @ _invert_blocks(N @ K, sz), R_vec


def compute_multipoint_trans(
    g, rock, eta: float = 0.0, use_tensor_assembly: bool = False
) -> Dict:
    """Return the MPFA-O transmissibility operators for grid ``g`` and ``rock``.

    ``use_tensor_assembly=True`` selects MRST's stabilized general-corner
    formulation, required when a 3-D cell corner touches more than three faces.
    """
    cno, nno, hfno, fno, subfno, subhfno = _create_mapping(g)
    B, R_vec = _local_flux_mimetic_ip(
        g, rock, cno, fno, nno, subhfno, eta, general=use_tensor_assembly
    )

    neigh = np.asarray(g.faces.neighbors, dtype=np.int64)
    n_sub = subhfno.size
    n_subf = int(subfno.max()) + 1
    s = 2 * (cno == neigh[fno, 0]) - 1

    D = sp.csr_array((np.ones(n_sub), (subhfno, subfno)),
                     shape=(n_sub, n_subf))
    Do = sp.csr_array((s.astype(float), (subhfno, subfno)),
                      shape=(n_sub, n_subf))
    C = sp.csr_array((np.ones(n_sub), (subhfno, cno)),
                     shape=(n_sub, g.cells.num))
    counts = np.bincount(fno, minlength=g.faces.num)
    d1 = sp.csr_array((np.ones(n_sub), (subfno, fno)),
                      shape=(n_subf, g.faces.num))

    DoBDo = (Do.T @ B @ Do).tocsr()

    # invert the node-blocks of DoBDo: reorder sub-faces grouped by node
    tmp = np.unique(np.column_stack([nno, subfno]), axis=0)
    p = tmp[:, 1]
    node_ids, sz = rlencode(tmp[:, 0])
    P = sp.csr_array((np.ones(p.size), (np.arange(p.size), p)),
                     shape=(p.size, p.size))
    iDoBDo = P.T @ _invert_blocks(DoBDo[np.ix_(p, p)], sz) @ P

    sb = np.asarray(D.sum(axis=0)).ravel() == 1                 # boundary sub-faces
    CD = sp.hstack([C, -D[:, sb]]).tocsr()
    cf_trans = (iDoBDo @ Do.T @ CD).tocsr()
    e_div = (CD.T @ Do).tocsr()
    cf_trans_g = (iDoBDo @ Do.T).tocsr()

    return {"cf_trans": cf_trans, "e_div": e_div, "cf_trans_g": cf_trans_g,
            "d1": d1, "R": R_vec, "cno": cno, "counts": counts, "sb": sb,
            "neighbors": neigh, "_iDoBDo": iDoBDo}


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY

    def test_operators_have_expected_shapes():
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        T = compute_multipoint_trans(g, rock)
        nsub = T["cf_trans"].shape[0]
        assert T["cf_trans"].shape[1] == g.cells.num + int(T["sb"].sum())
        assert T["e_div"].shape == (g.cells.num + int(T["sb"].sum()), nsub)
        print(f"ok test_operators_have_expected_shapes "
              f"(cf_trans {T['cf_trans'].shape})")

    def test_row_sum_of_cf_trans_is_zero_for_constant_pressure():
        """A constant cell/boundary pressure drives no flux: the interior part of
        cf_trans (cell columns) sums to zero per sub-face (partition of unity)."""
        g = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
        rock = make_rock(g, 1.0 * DARCY, 0.2)
        T = compute_multipoint_trans(g, rock)
        cf = T["cf_trans"].toarray()
        nc = g.cells.num
        allp = np.ones(cf.shape[1])                            # constant field
        flux = cf @ allp
        assert np.allclose(flux, 0.0, atol=1e-9), np.abs(flux).max()
        print("ok test_row_sum_of_cf_trans_is_zero_for_constant_pressure")

    test_operators_have_expected_shapes()
    test_row_sum_of_cf_trans_is_zero_for_constant_pressure()
    print("\nAll tests passed!")
