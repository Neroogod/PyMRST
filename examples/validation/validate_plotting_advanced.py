"""Validate advanced plotting geometry against MATLAB/MRST fixtures."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import (
    compute_geometry,
    make_layered_grid,
    process_grdecl,
    remove_cells,
)
from pymrst.gridprocessing.testgrids import one_sloping_fault
from pymrst.gridprocessing.triangle_grid import triangle_grid
from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_slice import slice_entities
from pymrst.plotting.private.plot_patches import build_patch_geometry


HERE = Path(__file__).parent


def _face_signatures(face_pos, face_nodes, coords, *, matlab=False):
    pos = np.asarray(face_pos, dtype=np.int64).ravel() - (1 if matlab else 0)
    nodes = np.asarray(face_nodes, dtype=np.int64).ravel() - (1 if matlab else 0)
    xyz = np.asarray(coords, dtype=float)
    signatures = []
    for start, stop in zip(pos[:-1], pos[1:]):
        vertices = xyz[nodes[start:stop]]
        signatures.append(tuple(sorted(tuple(np.round(v, 11)) for v in vertices)))
    return sorted(signatures)


def _compare_patch(name, grid, reference, prefix):
    geometry = build_patch_geometry(grid, boundary_faces(grid))
    got = sorted(
        tuple(sorted(tuple(np.round(v, 11)) for v in geometry.vertices[loop]))
        for loop in geometry.connectivity
    )
    expected = _face_signatures(
        reference[f"{prefix}_face_pos"],
        reference[f"{prefix}_face_nodes"],
        reference[f"{prefix}_node_coords"],
        matlab=True,
    )
    if got != expected:
        raise AssertionError(f"{name} patch topology differs from MATLAB")
    print(f"  ok   {name} patch topology ({len(got)} faces)")


def main():
    path = HERE / "ref_plotting_advanced.mat"
    if not path.exists():
        raise SystemExit("Run dump_plotting_advanced_reference.m in MATLAB first")
    ref = scipy.io.loadmat(path, squeeze_me=True)

    fault = compute_geometry(process_grdecl(one_sloping_fault([4, 2, 2], 2)))
    _compare_patch("faulted corner-point", fault, ref, "fault")

    points = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], float)
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    poly = compute_geometry(make_layered_grid(triangle_grid(points, triangles), 2))
    _compare_patch("general polyhedral", poly, ref, "poly")

    from pymrst.gridprocessing import cart_grid
    inactive = remove_cells(
        compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0])), [1, 7]
    )
    for case, (slice_index, dimension) in enumerate(
        ((1, 1), (0, 2), (1, 3)), start=1
    ):
        cells, faces, owners = slice_entities(
            inactive, slice_index, dimension
        )
        np.testing.assert_array_equal(
            cells + 1, np.asarray(ref[f"inactive_slice_cells{case}"]).ravel()
        )
        np.testing.assert_array_equal(
            faces + 1, np.asarray(ref[f"inactive_slice_faces{case}"]).ravel()
        )
        np.testing.assert_array_equal(
            owners + 1, np.asarray(ref[f"inactive_slice_values{case}"]).ravel()
        )
        print(f"  ok   inactive-cell slice dimension {dimension}")

    import pyvista as pv
    axis = np.asarray(ref["axis_values"], dtype=float).ravel()
    x, y, z = np.meshgrid(axis, axis, axis, indexing="xy")
    structured = pv.StructuredGrid(x, y, z)
    structured.point_data["v"] = (x+y+z).ravel(order="F")
    surface = structured.contour([float(ref["iso_level"])], scalars="v")
    bounds = np.asarray(surface.bounds).reshape(3, 2).T
    np.testing.assert_allclose(bounds, ref["iso_bounds"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(surface.area, float(ref["iso_area"]), rtol=0.03)
    # Counts are recorded for diagnosis, but triangulation legitimately
    # differs between MATLAB and VTK. Require only the same order of scale.
    ratio = surface.n_cells/float(ref["iso_face_count"])
    assert 0.5 <= ratio <= 2.0
    print(f"  ok   isosurface statistics ({surface.n_cells} VTK faces)")

    print("Advanced plotting fixtures match MATLAB/MRST")


if __name__ == "__main__":
    main()
