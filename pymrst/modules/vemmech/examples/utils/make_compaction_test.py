"""Compaction boundary conditions and load (MRST `makeCompactionTest`).

Sets up the classic reservoir-compaction problem: a block held on rollers at
its sides and base, loaded by its own weight and optionally by an overburden
force on the top. It is the driver behind `compactionTest2D` and
`compactionTest3D`.

"Rolling" is the important part. A side is pinned only in its own normal
direction and left free to slide tangentially, which is what lets the block
compact vertically without the sides fighting it. Getting a mask wrong here
does not error - it quietly produces a stiffer answer.

Note on orientation: MRST loads side ``2*griddim - 1`` and pins side
``2*griddim``, which in its ``XMin, XMax, YMin, YMax, ZMin, ZMax`` ordering
puts the overburden on **ZMin** and the support on **ZMax**. That is the
reservoir depth convention, where z grows downwards. The indices here are
transcribed from MRST rather than reasoned from "top", so a grid built z-up
will see the load on its z-minimum face.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.vemmech.vem import ElasticBC
from pymrst.params.wells_and_bc.pside import pside
from pymrst.utils.mcolon import mcolon

SIDES_3D = ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax")


@dataclass
class CompactionOptions:
    """Which parts of the compaction problem are switched on."""

    gravity_load: bool = True
    top_load: bool = True
    free_top: bool = True
    hanging: bool = False
    islinear: bool = False


def make_compaction_test(
    grid,
    opt: CompactionOptions,
    *,
    gravity: float = 10.0,
    density: float = 3000.0,
    top_force: float = 30000.0,
    rolling_vertical: bool = False,
    tol: Optional[float] = None,
) -> Tuple[ElasticBC, Callable[[np.ndarray], np.ndarray]]:
    """Return ``(el_bc, load)`` for the compaction problem on ``grid``."""
    dim = grid.griddim
    tol = float(np.sqrt(np.finfo(float).eps)) if tol is None else float(tol)
    side_faces = _side_faces(grid, tol)
    side_nodes = [_nodes_of(grid, faces) for faces in side_faces]
    masks = [np.ones((nodes.size, dim), dtype=bool) for nodes in side_nodes]

    keep = [True] * (2 * dim)
    if not opt.hanging or opt.islinear:
        for i in range(2 * (dim - 1)):
            # vertical sides roll: free in the vertical direction ...
            masks[i][:, dim - 1] = False
            if dim == 3 and not rolling_vertical:
                # ... and free in the in-plane tangential direction too
                masks[i][:, 1 if i <= 1 else 0] = False
        masks[2 * dim - 1][:, :dim - 1] = False          # base: vertical only
        if opt.free_top:
            keep[2 * dim - 2] = False                    # top carries no pin
        else:
            masks[2 * dim - 2][:, :dim - 1] = False
    else:
        for i in range(2 * dim - 3, 2 * dim):            # hanging: release them
            keep[i] = False

    nodes = np.concatenate([side_nodes[i] for i in range(2 * dim) if keep[i]]) \
        if any(keep) else np.zeros(0, dtype=np.int64)
    mask = np.vstack([masks[i] for i in range(2 * dim) if keep[i]]) \
        if any(keep) else np.zeros((0, dim), dtype=bool)

    bcdisp = _boundary_displacement(grid, opt)
    values = bcdisp(grid.nodes.coords[nodes, :dim]) if nodes.size \
        else np.zeros((0, dim))

    force_faces = np.zeros(0, dtype=np.int64)
    force = np.zeros((0, dim))
    if opt.top_load:
        force_faces = side_faces[2 * dim - 2]
        coords = np.asarray(grid.nodes.coords, dtype=float)[:, dim - 1]
        height = coords.max() - coords.min()
        direction = np.zeros(dim)
        direction[dim - 1] = -1.0
        force = np.tile(height * top_force * direction, (force_faces.size, 1))

    el_bc = ElasticBC(disp_nodes=nodes, disp_mask=mask, disp_vals=values,
                      force_faces=force_faces, force=force)
    return el_bc, _gravity_load(dim, opt, gravity, density)


def _side_faces(grid, tol: float):
    """The 2*griddim outer sides, in MRST's XMin, XMax, YMin, ... order."""
    dim = grid.griddim
    centroids = np.asarray(grid.faces.centroids, dtype=float)
    out = []
    for axis in range(dim):
        extremes = (centroids[:, axis].min(), centroids[:, axis].max())
        for value in extremes:
            if grid.cartDims is not None and grid.cells.tags is not None:
                side = SIDES_3D[2 * axis + (0 if value == extremes[0] else 1)]
                faces = np.asarray(pside(None, grid, side, 0.0).face,
                                   dtype=np.int64)
            else:
                faces = np.flatnonzero(np.abs(centroids[:, axis] - value) < tol)
                if not np.all(np.any(grid.faces.neighbors[faces] < 0, axis=1)):
                    raise ValueError("a side face is not on the boundary")
            out.append(faces)
    return out


def _nodes_of(grid, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64)
    inodes = mcolon(grid.faces.nodePos[faces], grid.faces.nodePos[faces + 1])
    return np.unique(np.asarray(grid.faces.nodes, dtype=np.int64)[inodes])


def _boundary_displacement(grid, opt: CompactionOptions):
    """The prescribed displacement: zero, or a linear field for `islinear`."""
    dim = grid.griddim
    if not opt.islinear:
        return lambda x: np.zeros_like(np.asarray(x, dtype=float))
    centroids = np.asarray(grid.cells.centroids, dtype=float)[:, :dim]
    origin = centroids.mean(axis=0)
    span = centroids[:, dim - 1].max() - centroids[:, dim - 1].min()
    factor = span * 1000.0
    axis = np.zeros(dim)
    axis[dim - 1] = 1.0
    return lambda x: (np.asarray(x, dtype=float) * axis - origin) / factor


def _gravity_load(dim: int, opt: CompactionOptions, gravity: float,
                  density: float):
    magnitude = (gravity * density) if opt.gravity_load else 0.0
    direction = np.zeros(dim)
    direction[dim - 1] = 1.0
    return lambda x: -magnitude * np.tile(direction,
                                          (np.asarray(x).shape[0], 1))


makeCompactionTest = make_compaction_test


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import create_augmented_grid, enu2c, enu2c_2d
    from pymrst.modules.vemmech.vem_lin_elast import vem_lin_elast

    def _grid(dims, size):
        return create_augmented_grid(compute_geometry(cart_grid(dims, size)))

    def _C(grid, E=1e9, nu=0.25):
        build = enu2c if grid.griddim == 3 else enu2c_2d
        return build(np.full(grid.cells.num, E), np.full(grid.cells.num, nu))

    def test_sides_roll_and_the_base_is_pinned_vertically() -> None:
        grid = _grid([3, 3, 3], [1.0, 1.0, 1.0])
        el_bc, _ = make_compaction_test(grid, CompactionOptions())
        assert el_bc.disp_mask.shape[1] == 3
        # something must be pinned, but never everything
        assert el_bc.disp_mask.any() and not el_bc.disp_mask.all()
        print("ok test_sides_roll_and_the_base_is_pinned_vertically")

    def test_gravity_load_points_down_and_scales() -> None:
        grid = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        _, load = make_compaction_test(grid, CompactionOptions(),
                                       gravity=10.0, density=3000.0)
        value = load(grid.cells.centroids)
        assert np.allclose(value[:, :2], 0.0)
        assert np.allclose(value[:, 2], -30000.0)
        print("ok test_gravity_load_points_down_and_scales")

    def test_gravity_can_be_switched_off() -> None:
        grid = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        _, load = make_compaction_test(
            grid, CompactionOptions(gravity_load=False))
        assert np.allclose(load(grid.cells.centroids), 0.0)
        print("ok test_gravity_can_be_switched_off")

    def test_overburden_loads_the_side_mrst_selects() -> None:
        """MRST loads side 2*griddim-1, which is ZMin in its ordering."""
        grid = _grid([2, 2, 2], [1.0, 1.0, 1.0])
        el_bc, _ = make_compaction_test(grid, CompactionOptions(top_load=True))
        assert el_bc.force_faces.size > 0
        assert np.all(el_bc.force[:, 2] < 0), "overburden acts along -z"
        centroids = grid.faces.centroids[el_bc.force_faces]
        assert np.allclose(centroids[:, 2], centroids[:, 2].min())
        assert np.allclose(centroids[:, 2], 0.0), "ZMin in this grid"
        print("ok test_overburden_loads_the_side_mrst_selects")

    def test_free_top_leaves_the_top_unpinned() -> None:
        grid = _grid([3, 3, 3], [1.0, 1.0, 1.0])
        free, _ = make_compaction_test(grid, CompactionOptions(free_top=True))
        held, _ = make_compaction_test(grid, CompactionOptions(free_top=False))
        assert held.disp_nodes.size > free.disp_nodes.size
        print("ok test_free_top_leaves_the_top_unpinned")

    def test_compaction_solves_with_the_supported_side_held() -> None:
        """Gravity and overburden both act along -z; the supported side (ZMax
        in MRST's ordering) must not move in z, and the loaded side must."""
        grid = _grid([4, 4, 4], [10.0, 10.0, 10.0])
        el_bc, load = make_compaction_test(grid, CompactionOptions())
        u, _ = vem_lin_elast(grid, _C(grid), el_bc, load)
        loaded = np.isclose(grid.nodes.coords[:, 2], 0.0)
        supported = np.isclose(grid.nodes.coords[:, 2], 10.0)
        assert np.all(np.isfinite(u))
        assert abs(u[supported, 2]).max() < 1e-12, "the supported side is held"
        assert u[loaded, 2].mean() < 0, "the loaded side moves along -z"
        print(f"ok test_compaction_solves_with_the_supported_side_held "
              f"(loaded side moves {u[loaded, 2].mean():.4e} m)")

    def test_two_dimensional_compaction_also_deforms() -> None:
        grid = _grid([4, 4], [10.0, 10.0])
        el_bc, load = make_compaction_test(grid, CompactionOptions())
        u, _ = vem_lin_elast(grid, _C(grid), el_bc, load)
        loaded = np.isclose(grid.nodes.coords[:, 1], 0.0)
        assert np.all(np.isfinite(u))
        assert u[loaded, 1].mean() < 0
        print("ok test_two_dimensional_compaction_also_deforms")

    def test_more_overburden_settles_further() -> None:
        grid = _grid([4, 4, 4], [10.0, 10.0, 10.0])
        settle = []
        for force in (30000.0, 60000.0):
            el_bc, load = make_compaction_test(
                grid, CompactionOptions(gravity_load=False), top_force=force)
            u, _ = vem_lin_elast(grid, _C(grid), el_bc, load)
            settle.append(u[np.isclose(grid.nodes.coords[:, 2], 0.0), 2].mean())
        assert settle[1] < settle[0] < 0
        assert np.isclose(settle[1], 2.0 * settle[0], rtol=1e-9), settle
        print("ok test_more_overburden_settles_further")

    test_sides_roll_and_the_base_is_pinned_vertically()
    test_gravity_load_points_down_and_scales()
    test_gravity_can_be_switched_off()
    test_overburden_loads_the_side_mrst_selects()
    test_free_top_leaves_the_top_unpinned()
    test_compaction_solves_with_the_supported_side_held()
    test_two_dimensional_compaction_also_deforms()
    test_more_overburden_settles_further()
    print("\nAll tests passed!")
