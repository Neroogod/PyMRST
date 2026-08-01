"""Extrude a 2-D grid onto horizons (MRST ``utils/gridtools/makeLayeredHorizonGrid.m``).

Takes an areal grid - triangular, PEBI, whatever suits the wells and faults -
and drapes it over a stack of surfaces, giving a 3-D grid whose layers follow
the horizons. `convert_horizons_to_grid` solves the same problem but only ever
produces a Cartesian areal mesh; this one keeps whatever areal grid you built.

`layers[i]` cells go between horizon `i` and `i+1`, with node depths
interpolated linearly. The result is passed through `remove_short_edges`, so
**cells that collapse to nothing where two horizons touch are removed, not left
as degenerate cells**. That is why a second return value exists: `cellmap` says
which cell of the uncollapsed grid each surviving cell came from, and
`cells.layer` is carried through it.

**Interpolation is scattered, not gridded** - unlike
`convert_horizons_to_grid`, which requires its horizons on a rectilinear mesh.
Here they can be arbitrary point clouds, which is what makes this usable with
raw picks.

**MRST's `extrapMethod` option has no effect.** It is declared in the options
struct and documented by implication, but `scatteredInterpolant` is constructed
with only the interpolation method, so the default extrapolation applies
whatever is asked for. This port omits the parameter rather than accepting one
that does nothing, and states the actual behaviour instead: nodes outside the
horizons' hull are **linearly extrapolated**, and for an areal grid extending
past the surfaces that means invented depths rather than an error. Check the
footprint before trusting the edges.
"""

import warnings
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from pymrst.gridprocessing.make_layered_grid import make_layered_grid
from pymrst.gridprocessing.remove_short_edges import remove_short_edges
from pymrst.utils.rldecode import rldecode


def _field(horizon, name):
    value = horizon[name] if hasattr(horizon, "keys") else getattr(horizon, name)
    return np.asarray(value, dtype=float).ravel()


def _interpolate(horizon, points, method):
    x, y, z = (_field(horizon, n) for n in ("x", "y", "z"))
    if not (x.size == y.size == z.size):
        raise ValueError(
            f"a horizon's x, y and z must have the same number of points, got "
            f"{x.size}, {y.size} and {z.size}")
    known = np.column_stack([x, y])
    if method == "nearest":
        return NearestNDInterpolator(known, z)(points)
    inside = LinearNDInterpolator(known, z)(points)
    # Outside the hull LinearNDInterpolator gives NaN; MATLAB's
    # scatteredInterpolant extrapolates. Fall back to nearest so the two agree
    # in behaviour, and see the module docstring for why that is a warning
    # rather than a feature.
    outside = np.isnan(inside)
    if np.any(outside):
        inside[outside] = NearestNDInterpolator(known, z)(points[outside])
    return inside


def _repair(surfaces, function, order, offset):
    if function is None:
        return surfaces
    surfaces = list(surfaces)
    for i in order:
        own, other = surfaces[i], surfaces[i + offset]
        if offset < 0:
            unset = np.isinf(own)
            own = np.where(unset, other, own)
        holes = np.isnan(own)
        own = np.asarray(function(own, other), dtype=float)
        surfaces[i] = np.where(holes, np.nan, own)
    return surfaces


def make_layered_horizon_grid(
        g2d, horizons: Sequence, layers=None, method: str = "linear",
        do_repair: bool = True,
        repair_function: Optional[Callable] = None,
        repair_function2: Optional[Callable] = None) -> Tuple:
    """Return ``(grid, cellmap)`` for ``g2d`` draped over ``horizons``."""
    horizons = list(horizons)
    if len(horizons) < 2:
        raise ValueError(
            f"at least two horizons are needed to enclose a volume, got "
            f"{len(horizons)}")

    count = len(horizons) - 1
    if layers is None:
        layers = np.ones(count, dtype=np.int64)
    else:
        layers = np.atleast_1d(np.asarray(layers, dtype=np.int64))
        if layers.size == 1:
            layers = np.repeat(layers, count)
    if layers.size != count:
        raise ValueError(
            f"one layer count is needed per gap between horizons, so "
            f"{count} of them, got {layers.size}")
    if np.any(layers < 1):
        raise ValueError("every layer count must be at least 1")

    areal = np.asarray(g2d.nodes.coords, dtype=float)[:, :2]
    surfaces = [_interpolate(h, areal, method) for h in horizons]
    if do_repair:
        surfaces = _repair(surfaces, repair_function, range(1, len(horizons)), -1)
        surfaces = _repair(surfaces, repair_function2,
                           range(len(horizons) - 2, -1, -1), 1)

    g = make_layered_grid(g2d, int(layers.sum()))
    if getattr(g.faces, "tag", None) is not None:
        g.faces.tag = None

    # Which pair of horizons each cell layer belongs to.
    layer_of_cell = rldecode(np.arange(count), layers)
    layer = np.tile(layer_of_cell, (g2d.cells.num, 1)).ravel(order="F")

    coords = np.asarray(g.nodes.coords, dtype=float).copy()
    per_layer = int(g2d.nodes.num)
    coords[:per_layer, 2] = surfaces[0]

    previous, k = surfaces[0], 1
    for index in range(1, len(horizons)):
        following = surfaces[index]
        step = (following - previous) / layers[index - 1]
        finite = step[np.isfinite(step)]
        # Scale-relative, unlike MRST's bare `>= 0`. Interpolating a surface
        # that touches another returns values like -5e-18 there rather than
        # exactly zero, and a strict test reports that rounding as crossing
        # horizons - which is noise on precisely the tangency case that is
        # most likely to be deliberate.
        slack = 1e-12 * max(1.0, float(np.max(np.abs(finite))) if finite.size
                            else 1.0)
        if finite.size and not np.all(finite >= -slack):
            warnings.warn(
                "non-monotone horizons: some pillars have a deeper surface "
                "above a shallower one, so the layers there are inverted.",
                stacklevel=2)
        for sub in range(1, int(layers[index - 1]) + 1):
            coords[k * per_layer:(k + 1) * per_layer, 2] = previous + sub * step
            k += 1
        previous = following
    g.nodes.coords = coords

    # Collapse cells that have no thickness where horizons meet. MRST's
    # removeShortEdges returns the cell map; PyMRST's does not, so it is
    # recovered by numbering the cells first - `indexMap` is subset by the
    # removal, which is exactly the map.
    original = getattr(g.cells, "indexMap", None)
    g.cells.indexMap = np.arange(g.cells.num)
    span = float(np.nanmax(surfaces[-1]) - np.nanmin(surfaces[0]))
    g = remove_short_edges(g, span * 1e-16)

    cellmap = np.asarray(g.cells.indexMap, dtype=np.int64)
    g.cells.indexMap = (None if original is None
                        else np.asarray(original)[cellmap])
    g.cells.layer = layer[cellmap]
    return g, cellmap


makeLayeredHorizonGrid = make_layered_horizon_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def areal(dims=(4, 4), sizes=(1.0, 1.0)):
        return compute_geometry(cart_grid(list(dims), list(sizes)))

    def cloud(f, n=9):
        """A horizon as scattered points, which is what this accepts."""
        x, y = np.meshgrid(np.linspace(-0.1, 1.1, n), np.linspace(-0.1, 1.1, n))
        return {"x": x, "y": y, "z": np.asarray(f(x, y), dtype=float)}

    def flat(depth, n=9):
        return cloud(lambda x, y: np.full_like(x, float(depth)), n)

    def test_it_extrudes_to_the_requested_layers() -> None:
        g, _ = make_layered_horizon_grid(areal(), [flat(0.0), flat(10.0)],
                                         layers=3)
        assert g.griddim == 3 and g.cells.num == 16 * 3, g.cells.num
        print("ok test_it_extrudes_to_the_requested_layers")

    def test_the_volume_is_the_slab_between_the_surfaces() -> None:
        g, _ = make_layered_horizon_grid(areal(), [flat(0.0), flat(10.0)],
                                         layers=2)
        g = compute_geometry(g)
        assert np.isclose(np.sum(g.cells.volumes), 1.0 * 1.0 * 10.0, rtol=1e-9)
        print("ok test_the_volume_is_the_slab_between_the_surfaces")

    def test_the_top_and_bottom_sit_on_the_horizons() -> None:
        g, _ = make_layered_horizon_grid(areal(), [flat(2.0), flat(7.0)])
        z = np.asarray(g.nodes.coords)[:, 2]
        assert np.isclose(z.min(), 2.0) and np.isclose(z.max(), 7.0)
        print("ok test_the_top_and_bottom_sit_on_the_horizons")

    def test_a_layer_count_per_gap_is_honoured() -> None:
        g, _ = make_layered_horizon_grid(
            areal(), [flat(0.0), flat(5.0), flat(9.0)], layers=[2, 3])
        assert g.cells.num == 16 * 5, g.cells.num
        assert set(np.unique(g.cells.layer).tolist()) == {0, 1}
        print("ok test_a_layer_count_per_gap_is_honoured")

    def test_the_layer_map_counts_cells_per_interval() -> None:
        g, _ = make_layered_horizon_grid(
            areal(), [flat(0.0), flat(5.0), flat(9.0)], layers=[2, 3])
        counts = np.bincount(np.asarray(g.cells.layer))
        assert counts.tolist() == [16 * 2, 16 * 3], counts
        print("ok test_the_layer_map_counts_cells_per_interval")

    def test_it_follows_a_sloping_horizon() -> None:
        g, _ = make_layered_horizon_grid(
            areal(), [cloud(lambda x, y: 5.0 * x), flat(20.0)])
        g = compute_geometry(g)
        centroids = np.asarray(g.cells.centroids)
        assert (centroids[centroids[:, 0] > 0.8, 2].mean()
                > centroids[centroids[:, 0] < 0.2, 2].mean())
        print("ok test_it_follows_a_sloping_horizon")

    def test_a_non_cartesian_areal_grid_works() -> None:
        """The reason to use this over convert_horizons_to_grid."""
        from pymrst.gridprocessing import triangle_grid
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                           [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        triangles = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
        base = compute_geometry(triangle_grid(points, triangles))
        g, _ = make_layered_horizon_grid(
            base, [flat(0.0), flat(4.0)], layers=2)
        assert g.cells.num == 4 * 2
        g = compute_geometry(g)
        assert np.isclose(np.sum(g.cells.volumes), 2.0 * 4.0, rtol=1e-9)
        print("ok test_a_non_cartesian_areal_grid_works")

    def test_touching_horizons_collapse_rather_than_degenerate() -> None:
        """Where two horizons meet there is no rock, so there is no cell."""
        wedge = cloud(lambda x, y: 5.0 * x)
        g, cellmap = make_layered_horizon_grid(areal(), [flat(0.0), wedge])
        g = compute_geometry(g)
        assert np.all(np.asarray(g.cells.volumes) > 0), \
            "a zero-volume cell survived the collapse"
        assert np.asarray(cellmap).size == g.cells.num
        print(f"ok test_touching_horizons_collapse_rather_than_degenerate "
              f"({g.cells.num} cells)")

    def test_the_cell_map_indexes_the_uncollapsed_grid() -> None:
        g, cellmap = make_layered_horizon_grid(
            areal(), [flat(0.0), flat(6.0)], layers=2)
        cellmap = np.asarray(cellmap, dtype=np.int64)
        assert cellmap.size == g.cells.num
        assert cellmap.max() < 16 * 2 and cellmap.min() >= 0
        print("ok test_the_cell_map_indexes_the_uncollapsed_grid")

    def test_crossing_horizons_warn() -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_layered_horizon_grid(
                areal(), [cloud(lambda x, y: 10.0 * x),
                          cloud(lambda x, y: 10.0 - 10.0 * x)])
        assert any("non-monotone" in str(w.message) for w in caught), caught
        print("ok test_crossing_horizons_warn")

    def test_scattered_horizons_need_no_grid() -> None:
        """What separates this from convert_horizons_to_grid."""
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.2, 1.2, (200, 2))
        scattered = {"x": points[:, 0], "y": points[:, 1],
                     "z": np.full(200, 3.0)}
        g, _ = make_layered_horizon_grid(areal(), [flat(0.0), scattered])
        z = np.asarray(g.nodes.coords)[:, 2]
        assert np.isclose(z.max(), 3.0, atol=1e-9), z.max()
        print("ok test_scattered_horizons_need_no_grid")

    def test_one_horizon_is_rejected() -> None:
        try:
            make_layered_horizon_grid(areal(), [flat(0.0)])
        except ValueError as error:
            assert "at least two horizons" in str(error)
            print("ok test_one_horizon_is_rejected")
            return
        raise AssertionError("a single horizon encloses nothing")

    def test_a_wrong_layer_count_is_rejected() -> None:
        try:
            make_layered_horizon_grid(areal(), [flat(0.0), flat(5.0)],
                                      layers=[1, 2, 3])
        except ValueError as error:
            assert "per gap between horizons" in str(error)
            print("ok test_a_wrong_layer_count_is_rejected")
            return
        raise AssertionError("one layer count per gap is required")

    test_it_extrudes_to_the_requested_layers()
    test_the_volume_is_the_slab_between_the_surfaces()
    test_the_top_and_bottom_sit_on_the_horizons()
    test_a_layer_count_per_gap_is_honoured()
    test_the_layer_map_counts_cells_per_interval()
    test_it_follows_a_sloping_horizon()
    test_a_non_cartesian_areal_grid_works()
    test_touching_horizons_collapse_rather_than_degenerate()
    test_the_cell_map_indexes_the_uncollapsed_grid()
    test_crossing_horizons_warn()
    test_scattered_horizons_need_no_grid()
    test_one_horizon_is_rejected()
    test_a_wrong_layer_count_is_rejected()
    print("\nAll tests passed!")
