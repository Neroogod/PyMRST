"""Order horizons by depth (MRST ``utils/gridtools/sortHorizons.m``).

Horizons arrive in whatever order they were read or picked, and everything that
builds a grid from them - `convert_horizons_to_grid`,
`make_layered_horizon_grid` - assumes they run top to bottom. This sorts them
and reports the permutation, so any per-horizon data the caller holds alongside
can be reordered to match.

A horizon is a surface given as three arrays `x`, `y`, `z` on a mesh. Depth
order is by each surface's **shallowest** point, `min(z)`, which is MRST's
choice and worth stating because it is not the only reasonable one: two
horizons that interpenetrate have no well-defined order, and ranking by minimum
resolves that arbitrarily rather than reporting it. Ranking by mean would be
just as arbitrary and would disagree. If your surfaces cross, sorting them is
not the step that will tell you.

Accepts horizons as mappings (`{"x": ..., "z": ...}`) or as objects with `.z`,
since both turn up depending on whether the surfaces came from a file reader or
were built in code.
"""

from typing import List, Sequence, Tuple

import numpy as np


def _depth(horizon) -> float:
    """The shallowest z of one horizon, however it is packaged."""
    try:
        z = horizon["z"] if hasattr(horizon, "keys") else horizon.z
    except (KeyError, AttributeError):
        raise TypeError(
            "a horizon must have a 'z' field, as a mapping key or an "
            "attribute; got an object with none") from None
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        raise ValueError("a horizon has no points")
    return float(np.nanmin(z))


def sort_horizons(horizons: Sequence) -> Tuple[List, np.ndarray]:
    """Return ``(sorted_horizons, order)``, shallowest first."""
    horizons = list(horizons)
    if not horizons:
        return [], np.zeros(0, dtype=np.int64)
    order = np.argsort([_depth(h) for h in horizons], kind="stable")
    return [horizons[i] for i in order], order.astype(np.int64)


sortHorizons = sort_horizons


if __name__ == "__main__":

    def surface(z, n=3):
        x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
        return {"x": x, "y": y, "z": np.full((n, n), float(z))}

    def test_they_come_back_shallowest_first() -> None:
        got, order = sort_horizons([surface(30), surface(10), surface(20)])
        assert [h["z"][0, 0] for h in got] == [10.0, 20.0, 30.0]
        assert np.array_equal(order, [1, 2, 0])
        print("ok test_they_come_back_shallowest_first")

    def test_the_order_reindexes_the_input() -> None:
        """What the second return value is for: reordering parallel data."""
        horizons = [surface(30), surface(10), surface(20)]
        names = np.array(["deep", "shallow", "middle"])
        got, order = sort_horizons(horizons)
        assert list(names[order]) == ["shallow", "middle", "deep"]
        print("ok test_the_order_reindexes_the_input")

    def test_already_sorted_is_left_alone() -> None:
        horizons = [surface(1), surface(2), surface(3)]
        got, order = sort_horizons(horizons)
        assert np.array_equal(order, [0, 1, 2])
        assert all(a is b for a, b in zip(got, horizons))
        print("ok test_already_sorted_is_left_alone")

    def test_it_ranks_by_the_shallowest_point_not_the_mean() -> None:
        """MRST's choice, and the two disagree; see the module docstring."""
        tilted = surface(0)
        tilted["z"] = np.array([[0.0, 100.0, 100.0]] * 3)     # min 0, mean 66.7
        flat = surface(50)                                     # min and mean 50
        got, _ = sort_horizons([flat, tilted])
        assert got[0] is tilted, "min(z) must decide, not the mean"
        assert np.mean(tilted["z"]) > np.mean(flat["z"])
        print("ok test_it_ranks_by_the_shallowest_point_not_the_mean")

    def test_objects_work_as_well_as_mappings() -> None:
        class Surface:
            def __init__(self, z):
                self.z = np.full((2, 2), float(z))
        got, order = sort_horizons([Surface(5), Surface(1)])
        assert np.array_equal(order, [1, 0])
        print("ok test_objects_work_as_well_as_mappings")

    def test_ties_keep_their_input_order() -> None:
        a, b = surface(7), surface(7)
        got, order = sort_horizons([a, b])
        assert got[0] is a and got[1] is b and np.array_equal(order, [0, 1])
        print("ok test_ties_keep_their_input_order")

    def test_nan_holes_are_ignored() -> None:
        """Surfaces from real data have gaps; a NaN must not swallow the rank."""
        holed = surface(4)
        holed["z"][0, 0] = np.nan
        got, _ = sort_horizons([surface(9), holed])
        assert got[0] is holed
        print("ok test_nan_holes_are_ignored")

    def test_an_empty_list_is_fine() -> None:
        got, order = sort_horizons([])
        assert got == [] and order.size == 0
        print("ok test_an_empty_list_is_fine")

    def test_a_horizon_without_z_is_rejected() -> None:
        try:
            sort_horizons([{"x": 1, "y": 2}])
        except TypeError as error:
            assert "'z' field" in str(error)
            print("ok test_a_horizon_without_z_is_rejected")
            return
        raise AssertionError("a horizon with no z must be rejected")

    def test_an_empty_horizon_is_rejected() -> None:
        try:
            sort_horizons([{"z": np.zeros(0)}])
        except ValueError as error:
            assert "no points" in str(error)
            print("ok test_an_empty_horizon_is_rejected")
            return
        raise AssertionError("an empty horizon must be rejected")

    test_they_come_back_shallowest_first()
    test_the_order_reindexes_the_input()
    test_already_sorted_is_left_alone()
    test_it_ranks_by_the_shallowest_point_not_the_mean()
    test_objects_work_as_well_as_mappings()
    test_ties_keep_their_input_order()
    test_nan_holes_are_ignored()
    test_an_empty_list_is_fine()
    test_a_horizon_without_z_is_rejected()
    test_an_empty_horizon_is_rejected()
    print("\nAll tests passed!")
