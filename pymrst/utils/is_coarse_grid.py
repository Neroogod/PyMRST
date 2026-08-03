"""Is this an upscaled grid? (MRST ``utils/isCoarseGrid.m``).

A coarse grid carries the fine grid it came from (`parent`) and the map from
fine cells to coarse ones (`partition`). MRST checks field presence only, so
objects with both fields are classified as coarse even when their values are
empty. This reproduces that structural predicate.
"""


def is_coarse_grid(g) -> bool:
    """True when ``g`` has both a ``parent`` grid and a ``partition``."""
    return hasattr(g, "parent") and hasattr(g, "partition")


isCoarseGrid = is_coarse_grid


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_plain_grid_is_not_coarse() -> None:
        assert not is_coarse_grid(compute_geometry(cart_grid([2, 2], [2.0, 2.0])))
        print("ok test_a_plain_grid_is_not_coarse")

    def test_a_grid_with_both_fields_is_coarse() -> None:
        g = SimpleNamespace(parent="fine", partition=[0, 0, 1, 1])
        assert is_coarse_grid(g)
        print("ok test_a_grid_with_both_fields_is_coarse")

    def test_one_field_alone_is_not_enough() -> None:
        """Half a coarse grid is malformed, not coarse."""
        assert not is_coarse_grid(SimpleNamespace(parent="fine"))
        assert not is_coarse_grid(SimpleNamespace(partition=[0, 1]))
        print("ok test_one_field_alone_is_not_enough")

    def test_fields_present_but_none_do_not_count() -> None:
        g = SimpleNamespace(parent=None, partition=[0, 1])
        assert is_coarse_grid(g)
        print("ok test_fields_present_but_none_do_not_count")

    test_a_plain_grid_is_not_coarse()
    test_a_grid_with_both_fields_is_coarse()
    test_one_field_alone_is_not_enough()
    test_fields_present_but_none_do_not_count()
    print("\nAll tests passed!")
