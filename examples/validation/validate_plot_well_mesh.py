"""Compare PyMRST plotWell meshes with fixtures dumped by MATLAB/MRST."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params.wells_and_bc import Well
from pymrst.plotting.plot_well import plot_well


class CapturePlotter:
    def add_mesh(self, mesh, **kwargs):
        return mesh

    def add_point_labels(self, *args, **kwargs):
        return None


def well(cells, direction):
    cells = np.asarray(cells, dtype=np.int64)
    return Well(cells, "rate", 1.0, np.ones(cells.size), direction,
                np.ones(cells.size), np.zeros(cells.size), "W",
                np.array([1.0, 0.0]), 0.0, 1)


def main():
    path = Path(__file__).with_name("plot_well_mesh_reference.mat")
    if not path.exists():
        raise SystemExit("Run dump_plot_well_mesh_reference.m in MATLAB first")
    refs = loadmat(path, squeeze_me=True, struct_as_record=False)
    cases = {
        "x": ([2, 1, 1], [2, 1, 1], [0, 1], "xx", 8),
        "y": ([1, 2, 1], [1, 2, 1], [0, 1], "yy", 8),
        "z": ([1, 1, 2], [1, 1, 2], [0, 1], "zz", 8),
        "slanted": ([2, 1, 2], [2, 1, 2], [0, 3], "qq", 8),
        "single": ([1, 1, 1], [2, 3, 4], [0], "z", 10),
    }
    for name, (dims, extent, cells, direction, cylpts) in cases.items():
        grid = compute_geometry(cart_grid(dims, extent))
        got = plot_well(grid, well(cells, direction), CapturePlotter(),
                        cylpts=cylpts, fontsize=0).geometry[0]
        ref = refs[name]
        np.testing.assert_allclose(got.surface_vertices, ref.vertices,
                                   rtol=0, atol=2e-15)
        np.testing.assert_array_equal(got.surface_faces + 1, ref.faces)
        np.testing.assert_allclose(got.head_vertices, ref.head_vertices,
                                   rtol=0, atol=2e-15)
        np.testing.assert_allclose(got.heel_vertices, ref.heel_vertices,
                                   rtol=0, atol=2e-15)
        print(f"  ok   {name} well mesh")
    print("plotWell meshes match MRST")


if __name__ == "__main__":
    main()
