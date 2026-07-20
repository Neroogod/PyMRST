"""Phase 72: vemmech deformed plotting geometry vs MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.plotting import (
    deformed_grid,
    plot_faces_2d_geometry,
    plot_node_data_deformed_geometry,
)
from pymrst.modules.vemmech.vem import create_augmented_grid

HERE = Path(__file__).parent
REF = HERE / "ref_p72_vemmech_plotting.mat"


def _field(s, name):
    return getattr(s, name)


def _coords(gstruct):
    return np.asarray(_field(_field(gstruct, "nodes"), "coords"), dtype=float)


def compare(name, py, ml, rtol=1e-12, atol=1e-12) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ml = np.asarray(ml, dtype=float).ravel()
    scale = max(float(np.nanmax(np.abs(ml))), 1.0)
    diff = float(np.nanmax(np.abs(py - ml)))
    ok = diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        raise SystemExit(
            f"Missing {REF.name}. Run dump_phase72_reference.m in MATLAB first."
        )
    ref = scipy.io.loadmat(REF, squeeze_me=True, struct_as_record=False)

    g2 = create_augmented_grid(compute_geometry(cart_grid([2, 1],
                                                          [2.0, 1.0])))
    x2 = g2.nodes.coords[:, 0]
    y2 = g2.nodes.coords[:, 1]
    u2 = np.column_stack([0.1 * x2, -0.2 * y2])
    node_data2 = np.column_stack([
        np.arange(1, g2.nodes.num + 1),
        10 + np.arange(1, g2.nodes.num + 1),
    ])
    gd2 = deformed_grid(g2, u2)
    patch2 = plot_node_data_deformed_geometry(g2, node_data2, u2)
    faces2 = np.arange(g2.faces.num)
    shift2 = np.linspace(0.01, 0.05, g2.faces.num)
    lines2 = plot_faces_2d_geometry(g2, faces2, uu=u2, shift=shift2)

    g3 = compute_geometry(cart_grid([1, 1, 1], [1.0, 2.0, 3.0]))
    x3 = g3.nodes.coords[:, 0]
    y3 = g3.nodes.coords[:, 1]
    z3 = g3.nodes.coords[:, 2]
    u3 = np.column_stack([0.01 * x3, -0.02 * y3, 0.03 * z3])
    node_data3 = np.arange(1, g3.nodes.num + 1)
    gd3 = deformed_grid(g3, u3)
    faces3 = np.asarray(ref["faces3"], dtype=np.int64).ravel() - 1
    patch3 = plot_node_data_deformed_geometry(g3, node_data3, u3)
    patch3_faces = plot_node_data_deformed_geometry(g3, node_data3, u3)
    # Recompute with MATLAB's boundary-face order for a strict topology check.
    from pymrst.modules.vemmech.plotting import node_patch_topology
    f3, verts3 = node_patch_topology(gd3, faces3)
    v3 = gd3.nodes.coords[verts3]
    cdata3 = node_data3[verts3]

    failures = 0
    print("Phase 72: vemmech deformed plotting geometry vs MATLAB MRST")
    failures += compare("2D deformed coords", gd2.nodes.coords,
                        _coords(ref["G2d"]))
    failures += compare("2D node patch faces", patch2["faces"], ref["f2"],
                        rtol=0, atol=0)
    failures += compare("2D node patch verts", patch2["verts"] + 1,
                        ref["verts2"], rtol=0, atol=0)
    failures += compare("2D node patch vertices", patch2["vertices"],
                        ref["v2"])
    failures += compare("2D node patch cdata", patch2["cdata"],
                        ref["cdata2"], rtol=0, atol=0)
    failures += compare("2D face line x", lines2[:, :, 0], ref["line_x2"])
    failures += compare("2D face line y", lines2[:, :, 1], ref["line_y2"])

    failures += compare("3D deformed coords", gd3.nodes.coords,
                        _coords(ref["G3d"]))
    failures += compare("3D node patch faces", f3, ref["f3"], rtol=0, atol=0)
    failures += compare("3D node patch verts", verts3 + 1, ref["verts3"],
                        rtol=0, atol=0)
    failures += compare("3D node patch vertices", v3, ref["v3"])
    failures += compare("3D node patch cdata", cdata3, ref["cdata3"],
                        rtol=0, atol=0)
    failures += compare("3D default patch cdata", patch3["cdata"],
                        patch3_faces["cdata"], rtol=0, atol=0)

    if failures:
        print(f"\nvemmech plotting validation failed ({failures} checks).")
    else:
        print("\nPyMRST matches MRST vemmech deformed plotting geometry.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
