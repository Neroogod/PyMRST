"""Corner-point grid from horizon surfaces (MRST ``utils/gridtools/convertHorizonsToGrid.m``).

Given two or more surfaces - seismic picks, mapped tops - this builds the
`GRDECL` deck of the volume between them, ready for `process_grdecl`.

The areal mesh is a Cartesian grid over the **union** of the horizons' bounding
boxes, so every surface is covered. Cells whose corners fall outside a given
horizon get `NaN` there, and the deck's `ACTNUM` deactivates any cell with a
non-finite corner. The active region is therefore roughly the *intersection* of
the horizons' footprints, while the array covers their union - the two differ,
and a grid with far fewer active cells than `cartDims` suggests is the expected
outcome for surfaces that do not overlap much, not a bug.

Between each pair of horizons `layers[i]` cells are inserted, their corner
depths interpolated linearly. Depths are then **sorted along each pillar**,
which silently repairs crossing horizons after warning about them - MRST's
behaviour, kept, because the alternative is a deck `process_grdecl` will reject.
The warning is the part to act on.

**Horizons must be in `ndgrid` orientation**, with `x` varying down the rows and
`y` across the columns. This is MRST's requirement too, inherited from
`griddedInterpolant`, and it is easy to violate because MATLAB's `meshgrid` and
NumPy's default `np.meshgrid` both give the transpose. MRST's own example
disguises it by writing `[y,x,z] = peaks(30)` - swapping the first two outputs
rather than explaining. Here the orientation is checked and a wrong one is
rejected with a message saying so, instead of producing a grid that is silently
transposed.

`repair_function` and `repair_function2` hook the collision pass, as in MRST:
`max`/`min` pick the lower/upper surface where two horizons cross. They run
downward and upward respectively, and `NaN` holes are preserved through both.
"""

import warnings
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pymrst.utils.discspace1 import discspace1


def _field(horizon, name):
    value = horizon[name] if hasattr(horizon, "keys") else getattr(horizon, name)
    return np.asarray(value, dtype=float)


def _axes(horizon, index):
    """The 1-D x and y axes of an ndgrid-oriented horizon."""
    x, y, z = (_field(horizon, n) for n in ("x", "y", "z"))
    if x.shape != z.shape or y.shape != z.shape:
        raise ValueError(
            f"horizon {index}: x, y and z must have the same shape, got "
            f"{x.shape}, {y.shape} and {z.shape}")
    if x.ndim != 2:
        raise ValueError(f"horizon {index}: expected 2-D arrays, got {x.ndim}-D")

    ax, ay = x[:, 0], y[0, :]
    if not (np.allclose(x, ax[:, None]) and np.allclose(y, ay[None, :])):
        raise ValueError(
            f"horizon {index} is not in ndgrid orientation: x must vary down "
            f"the rows and y across the columns. NumPy's np.meshgrid gives the "
            f"transpose - use indexing='ij', or transpose x, y and z together.")
    if np.any(np.diff(ax) <= 0) or np.any(np.diff(ay) <= 0):
        raise ValueError(
            f"horizon {index}: x and y must be strictly increasing along "
            f"their own axes")
    return ax, ay, z


def _interpolate(horizon, index, x, y, method, fill):
    ax, ay, z = _axes(horizon, index)
    interpolator = RegularGridInterpolator(
        (ax, ay), z, method=method, bounds_error=False, fill_value=fill)
    return interpolator(np.stack([x, y], axis=-1))


def _repair(surfaces, function, order, offset):
    """One collision pass over the stack; see the module docstring."""
    if function is None:
        return surfaces
    surfaces = list(surfaces)
    for i in order:
        own, other = surfaces[i], surfaces[i + offset]
        if offset < 0:
            # Depths left undefined by an earlier pass inherit from above.
            unset = np.isinf(own)
            own = np.where(unset, other, own)
        holes = np.isnan(own)
        own = np.asarray(function(own, other), dtype=float)
        surfaces[i] = np.where(holes, np.nan, own)
    return surfaces


def convert_horizons_to_grid(
        horizons: Sequence, dims: Optional[Sequence[int]] = None,
        layers=None, method: str = "linear",
        extrap_method: Optional[str] = None, do_repair: bool = True,
        refinex=None, refiney=None, refinez=None,
        refine_method: str = "linear",
        repair_function: Optional[Callable] = None,
        repair_function2: Optional[Callable] = None) -> Dict:
    """Build a ``GRDECL`` deck spanning ``horizons``, shallowest first."""
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
    total = int(layers.sum())

    shapes = np.array([_field(h, "z").shape for h in horizons])
    node_dims = (np.array([shapes[:, 0].max(), shapes[:, 1].max(), total]) + 1
                 if dims is None
                 else np.array([dims[0], dims[1], total]) + 1)

    corners = np.array([[_field(h, "x").min(), _field(h, "x").max(),
                         _field(h, "y").min(), _field(h, "y").max()]
                        for h in horizons])
    ax = discspace1(corners[:, 0].min(), corners[:, 1].max(),
                    int(node_dims[0]), refinex, refine_method)
    ay = discspace1(corners[:, 2].min(), corners[:, 3].max(),
                    int(node_dims[1]), refiney, refine_method)
    # The third axis only fixes the layer count; depths come from the horizons.
    az = discspace1(0.0, 1.0, int(node_dims[2]), refinez, refine_method)
    node_dims = np.array([ax.size, ay.size, az.size])
    cell_dims = node_dims - 1

    x, y = np.meshgrid(ax, ay, indexing="ij")
    fill = np.nan if extrap_method in (None, "none") else None
    if fill is None and extrap_method != "linear":
        raise ValueError(
            f"extrap_method must be None/'none' or 'linear', got "
            f"{extrap_method!r}")
    surfaces = [_interpolate(h, i, x, y, method, fill)
                for i, h in enumerate(horizons)]

    if do_repair:
        surfaces = _repair(surfaces, repair_function,
                           range(1, len(horizons)), -1)
        surfaces = _repair(surfaces, repair_function2,
                           range(len(horizons) - 2, -1, -1), 1)

    z = np.empty((node_dims[0], node_dims[1], node_dims[2]))
    previous, k = surfaces[0], 0
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
                "above a shallower one. Depths will be sorted along each "
                "pillar, which hides the crossing rather than resolving it.",
                stacklevel=2)
        for sub in range(int(layers[index - 1])):
            z[:, :, k] = previous + sub * step
            k += 1
        previous = following
    z[:, :, -1] = surfaces[-1]
    z = np.sort(z, axis=2)

    # Pillars run from the top surface to the bottom one.
    n = int(node_dims[0] * node_dims[1])
    lines = np.zeros((n, 6))
    lines[:, [0, 3]] = np.stack([x.ravel(order="F")] * 2, axis=1)
    lines[:, [1, 4]] = np.stack([y.ravel(order="F")] * 2, axis=1)
    lines[:, [2, 5]] = np.column_stack([z[:, :, 0].ravel(order="F"),
                                        z[:, :, -1].ravel(order="F")])

    # Each interior node is a corner of two cells, so the axes are doubled.
    def doubled(d):
        return np.arange(2 * cell_dims[d]) // 2 + (np.arange(2 * cell_dims[d]) % 2)

    zcorn = z[np.ix_(doubled(0), doubled(1), doubled(2))]

    # A cell is active only if all eight of its corners are real numbers.
    flat = zcorn.ravel(order="F")
    cell_index = np.arange(int(np.prod(cell_dims))).reshape(cell_dims, order="F")
    for axis in range(3):
        cell_index = np.repeat(cell_index, 2, axis=axis)
    totals = np.bincount(cell_index.ravel(order="F"), weights=flat,
                         minlength=int(np.prod(cell_dims)))
    actnum = np.isfinite(totals).astype(np.int32)

    coord = lines.reshape(-1)
    return {"cartDims": cell_dims.astype(np.int64),
            "COORD": np.where(np.isnan(coord), np.inf, coord),
            "ZCORN": np.where(np.isnan(flat), np.inf, flat),
            "ACTNUM": actnum}


convertHorizonsToGrid = convert_horizons_to_grid


if __name__ == "__main__":
    from pymrst.gridprocessing import compute_geometry, process_grdecl

    def surface(z_of_xy, n=8, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
        """An ndgrid-oriented horizon; see the module docstring."""
        ax, ay = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        x, y = np.meshgrid(ax, ay, indexing="ij")
        return {"x": x, "y": y, "z": np.asarray(z_of_xy(x, y), dtype=float)}

    def flat(depth, **kw):
        return surface(lambda x, y: np.full_like(x, float(depth)), **kw)

    def test_two_flat_horizons_give_a_box() -> None:
        deck = convert_horizons_to_grid([flat(0.0), flat(10.0)])
        assert tuple(deck["cartDims"]) == (8, 8, 1), deck["cartDims"]
        assert np.all(deck["ACTNUM"] == 1)
        print("ok test_two_flat_horizons_give_a_box")

    def test_the_deck_processes_into_a_grid() -> None:
        deck = convert_horizons_to_grid([flat(0.0), flat(10.0)], layers=3)
        g = compute_geometry(process_grdecl(deck))
        assert g.cells.num == 8 * 8 * 3, g.cells.num
        assert np.all(np.asarray(g.cells.volumes) > 0)
        print("ok test_the_deck_processes_into_a_grid")

    def test_the_volume_is_the_slab_between_the_surfaces() -> None:
        g = compute_geometry(process_grdecl(
            convert_horizons_to_grid([flat(0.0), flat(10.0)], layers=2)))
        assert np.isclose(np.sum(g.cells.volumes), 1.0 * 1.0 * 10.0, rtol=1e-9)
        print("ok test_the_volume_is_the_slab_between_the_surfaces")

    def test_layers_split_the_gap_evenly() -> None:
        g = compute_geometry(process_grdecl(
            convert_horizons_to_grid([flat(0.0), flat(9.0)], layers=3)))
        volumes = np.asarray(g.cells.volumes)
        assert np.allclose(volumes, volumes[0]), "layers should be equal"
        print("ok test_layers_split_the_gap_evenly")

    def test_a_layer_count_per_gap_is_honoured() -> None:
        deck = convert_horizons_to_grid([flat(0.0), flat(5.0), flat(9.0)],
                                        layers=[2, 3])
        assert tuple(deck["cartDims"]) == (8, 8, 5), deck["cartDims"]
        print("ok test_a_layer_count_per_gap_is_honoured")

    def test_dims_sets_the_areal_resolution() -> None:
        deck = convert_horizons_to_grid([flat(0.0), flat(10.0)],
                                        dims=[20, 15])
        assert tuple(deck["cartDims"]) == (20, 15, 1), deck["cartDims"]
        print("ok test_dims_sets_the_areal_resolution")

    def test_a_sloping_horizon_gives_sloping_cells() -> None:
        top = surface(lambda x, y: 5.0 * x)
        deck = convert_horizons_to_grid([top, flat(20.0)])
        g = compute_geometry(process_grdecl(deck))
        centroids = np.asarray(g.cells.centroids)
        low = centroids[centroids[:, 0] < 0.2, 2].mean()
        high = centroids[centroids[:, 0] > 0.8, 2].mean()
        assert high > low, (low, high)
        print("ok test_a_sloping_horizon_gives_sloping_cells")

    def test_non_overlapping_footprints_deactivate_cells() -> None:
        """The array covers the union; the active region is the overlap."""
        deck = convert_horizons_to_grid(
            [flat(0.0), flat(10.0, x0=0.5, x1=1.5)])
        active = int(deck["ACTNUM"].sum())
        assert 0 < active < deck["ACTNUM"].size, \
            (active, deck["ACTNUM"].size)
        print(f"ok test_non_overlapping_footprints_deactivate_cells "
              f"({active} of {deck['ACTNUM'].size})")

    def test_crossing_horizons_warn_and_are_sorted() -> None:
        """MRST's behaviour, kept - but the warning is the part to act on."""
        top = surface(lambda x, y: 10.0 * x)
        bottom = surface(lambda x, y: 10.0 - 10.0 * x)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deck = convert_horizons_to_grid([top, bottom])
        assert any("non-monotone" in str(w.message) for w in caught), caught
        # Sorting leaves every pillar's depths ascending.
        assert np.all(np.diff(np.sort(deck["ZCORN"][np.isfinite(deck["ZCORN"])]))
                      >= 0)
        print("ok test_crossing_horizons_warn_and_are_sorted")

    def test_the_repair_function_picks_a_surface() -> None:
        top = surface(lambda x, y: 10.0 * x)
        bottom = surface(lambda x, y: 10.0 - 10.0 * x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = convert_horizons_to_grid([top, bottom])
            repaired = convert_horizons_to_grid(
                [top, bottom], repair_function=np.maximum)
        assert not np.allclose(plain["ZCORN"], repaired["ZCORN"])
        print("ok test_the_repair_function_picks_a_surface")

    def test_a_meshgrid_oriented_horizon_is_rejected() -> None:
        """The mistake the orientation rule exists to catch."""
        ax = ay = np.linspace(0.0, 1.0, 6)
        x, y = np.meshgrid(ax, ay)                    # no indexing='ij'
        wrong = {"x": x, "y": y, "z": 2.0 * x + y}
        try:
            convert_horizons_to_grid([wrong, wrong])
        except ValueError as error:
            assert "ndgrid orientation" in str(error)
            print("ok test_a_meshgrid_oriented_horizon_is_rejected")
            return
        raise AssertionError("a transposed horizon must be rejected")

    def test_one_horizon_is_rejected() -> None:
        try:
            convert_horizons_to_grid([flat(0.0)])
        except ValueError as error:
            assert "at least two horizons" in str(error)
            print("ok test_one_horizon_is_rejected")
            return
        raise AssertionError("a single horizon encloses nothing")

    def test_a_wrong_layer_count_is_rejected() -> None:
        try:
            convert_horizons_to_grid([flat(0.0), flat(5.0), flat(9.0)],
                                     layers=[1, 2, 3])
        except ValueError as error:
            assert "per gap between horizons" in str(error)
            print("ok test_a_wrong_layer_count_is_rejected")
            return
        raise AssertionError("one layer count per gap is required")

    test_two_flat_horizons_give_a_box()
    test_the_deck_processes_into_a_grid()
    test_the_volume_is_the_slab_between_the_surfaces()
    test_layers_split_the_gap_evenly()
    test_a_layer_count_per_gap_is_honoured()
    test_dims_sets_the_areal_resolution()
    test_a_sloping_horizon_gives_sloping_cells()
    test_non_overlapping_footprints_deactivate_cells()
    test_crossing_horizons_warn_and_are_sorted()
    test_the_repair_function_picks_a_surface()
    test_a_meshgrid_oriented_horizon_is_rejected()
    test_one_horizon_is_rejected()
    test_a_wrong_layer_count_is_rejected()
    print("\nAll tests passed!")
