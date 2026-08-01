"""Phase 410: VEM grid geometry (edges/diameters) against MRST.

Pins `compute_vem_geometry` against MRST's `computeVEMGeometry` on a twisted 3-D
grid: the deduplicated global edge list (nodes, lengths, centroids), the per-face
edge maps and in-plane edge normals, cell edges, and cell/face diameters.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vem.utils.compute_vem_geometry import (
    compute_vem_geometry,
)

REF = Path(__file__).with_name("ref_p410_vem_geometry.mat")


def _twisted_grid(nx=3):
    g = cart_grid([nx, nx, nx], [float(nx)] * 3)
    c = np.asarray(g.nodes.coords, dtype=float)
    x, y, z, L = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy(), float(nx)
    c[:, 0] += 0.2 * np.sin(np.pi * x / L) * np.sin(np.pi * y / L)
    c[:, 1] += 0.15 * np.sin(np.pi * x / L) * np.sin(np.pi * z / L)
    c[:, 2] += 0.1 * np.sin(np.pi * y / L) * np.sin(np.pi * z / L)
    g.nodes.coords = c
    return compute_geometry(g)


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _face_edge_sets(faces_edges, edge_pos, nf):
    return [frozenset(faces_edges[edge_pos[f]:edge_pos[f + 1]]) for f in range(nf)]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase410_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    g = compute_vem_geometry(_twisted_grid())

    failed = 0
    print("computeVEMGeometry (twisted 3-D grid) vs MRST:")

    # Global edge list: same count and same canonical node pairs (row order too,
    # both being the lexicographic unique of the same pair set).
    ne_ref = int(ref["edgesNum"].ravel()[0])
    failed += _check(f"edge count ({g.edges.num})", g.edges.num == ne_ref,
                     f"MRST {ne_ref}")
    en_ref = ref["edgesNodes"].ravel().reshape(-1, 2) - 1     # 1- -> 0-indexed
    en = g.edges.nodes.reshape(-1, 2)
    failed += _check("edge node pairs match", np.array_equal(en, en_ref))
    failed += _check("edge lengths match",
                     np.allclose(g.edges.lengths, ref["edgesLengths"].ravel()),
                     f"max|dL| {np.max(np.abs(g.edges.lengths - ref['edgesLengths'].ravel())):.2e}")
    failed += _check("edge centroids match",
                     np.allclose(g.edges.centroids, ref["edgesCentroids"]))

    # Per-face edge structure (order-independent: compare edge sets per face).
    fe_ref = ref["facesEdges"].ravel().astype(np.int64) - 1
    fep_ref = ref["facesEdgePos"].ravel().astype(np.int64) - 1
    sets_py = _face_edge_sets(g.faces.edges, g.faces.edgePos, g.faces.num)
    sets_ref = _face_edge_sets(fe_ref, fep_ref, g.faces.num)
    failed += _check("per-face edge sets match", sets_py == sets_ref)

    # Edge normals are consistent per-face-edge only if the face-edge ordering
    # matches; compare directly when it does, else fall back to the set test.
    same_order = np.array_equal(g.faces.edges, fe_ref)
    if same_order:
        failed += _check("in-plane edge normals match (same ordering)",
                         np.allclose(g.faces.edgeNormals, ref["facesEdgeNorm"]),
                         f"max|dn| {np.max(np.abs(g.faces.edgeNormals - ref['facesEdgeNorm'])):.2e}")
    else:
        print("  note face-edge ordering differs from MRST; edge sets already "
              "verified, normals checked for unit length + in-plane below")
        en_unit = np.allclose(np.linalg.norm(g.faces.edgeNormals, axis=1), 1.0)
        foe = np.repeat(np.arange(g.faces.num), np.diff(g.faces.edgePos))
        fn = g.faces.normals[foe]
        fn = fn / np.linalg.norm(fn, axis=1, keepdims=True)
        in_plane = np.max(np.abs((g.faces.edgeNormals * fn).sum(1))) < 1e-12
        failed += _check("edge normals unit + in-plane", en_unit and in_plane)

    # Cell edges (order-independent sets) and diameters (direct).
    ce_ref = ref["cellsEdges"].ravel().astype(np.int64) - 1
    cep_ref = ref["cellsEdgePos"].ravel().astype(np.int64) - 1
    csets_py = _face_edge_sets(g.cells.edges, g.cells.edgePos, g.cells.num)
    csets_ref = _face_edge_sets(ce_ref, cep_ref, g.cells.num)
    failed += _check("per-cell edge sets match", csets_py == csets_ref)
    failed += _check("cell diameters match",
                     np.allclose(g.cells.diameters, ref["cellDiameters"].ravel()),
                     f"max|dd| {np.max(np.abs(g.cells.diameters - ref['cellDiameters'].ravel())):.2e}")
    failed += _check("face diameters match",
                     np.allclose(g.faces.diameters, ref["faceDiameters"].ravel()))

    print("\nVEM grid geometry matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
