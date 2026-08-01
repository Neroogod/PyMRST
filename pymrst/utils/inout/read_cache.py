"""Retrieve a cached result (MRST ``utils/inout/readCache.m``).

The other half of `write_cache`: hashes the arguments and loads the entry if it
exists. Returns `(found, variables)` so a miss is a value rather than an
exception - the caller's next step is to compute and cache, not to handle an
error.

**Only read a cache this process wrote.** Entries are pickles, so loading one
executes code in it. `write_cache`'s docstring explains why pickle rather than
`.npz`; the safety rule follows from it and is repeated here because this is
the function where it bites.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pymrst.utils.inout.write_cache import cache_path


def read_cache(arg, directory: Optional[Path] = None
               ) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return ``(found, variables)`` for the entry keyed by ``arg``."""
    path = cache_path(arg, directory)
    if not path.is_file():
        return False, None
    try:
        with path.open("rb") as handle:
            return True, pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError) as error:
        raise ValueError(
            f"the cache entry at {path} could not be read ({error}); delete it "
            f"and recompute") from None


readCache = read_cache


if __name__ == "__main__":
    import tempfile

    import numpy as np

    from pymrst.utils.inout.write_cache import write_cache

    def scratch():
        return Path(tempfile.mkdtemp())

    def test_a_miss_is_a_value_not_an_error() -> None:
        found, data = read_cache("never written", scratch())
        assert found is False and data is None
        print("ok test_a_miss_is_a_value_not_an_error")

    def test_a_round_trip() -> None:
        d = scratch()
        write_cache(["case", 1], {"x": np.arange(3), "n": 7}, d)
        found, data = read_cache(["case", 1], d)
        assert found is True
        assert data["x"].tolist() == [0, 1, 2] and data["n"] == 7
        print("ok test_a_round_trip")

    def test_a_different_key_misses() -> None:
        d = scratch()
        write_cache(["case", 1], {"x": 1}, d)
        assert read_cache(["case", 2], d)[0] is False
        print("ok test_a_different_key_misses")

    def test_rewriting_replaces_the_entry() -> None:
        d = scratch()
        write_cache("k", {"x": 1}, d)
        write_cache("k", {"x": 2}, d)
        assert read_cache("k", d)[1]["x"] == 2
        print("ok test_rewriting_replaces_the_entry")

    def test_it_survives_a_grid() -> None:
        """The reason it is pickle and not npz: real objects, not arrays."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        d = scratch()
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        write_cache("grid", {"G": g}, d)
        found, data = read_cache("grid", d)
        assert found and data["G"].cells.num == 6
        assert np.allclose(data["G"].cells.volumes, g.cells.volumes)
        print("ok test_it_survives_a_grid")

    def test_a_corrupt_entry_is_reported() -> None:
        from pymrst.utils.inout.write_cache import cache_path
        d = scratch()
        write_cache("k", {"x": 1}, d)
        cache_path("k", d).write_bytes(b"not a pickle at all")
        try:
            read_cache("k", d)
        except ValueError as error:
            assert "delete it and recompute" in str(error)
            print("ok test_a_corrupt_entry_is_reported")
            return
        raise AssertionError("a corrupt cache entry must be reported")

    test_a_miss_is_a_value_not_an_error()
    test_a_round_trip()
    test_a_different_key_misses()
    test_rewriting_replaces_the_entry()
    test_it_survives_a_grid()
    test_a_corrupt_entry_is_reported()
    print("\nAll tests passed!")
