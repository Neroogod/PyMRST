"""Cache a computed result (MRST ``utils/inout/writeCache.m``).

Stores values under a hash of the arguments that produced them, so an expensive
step - processing a large deck, building a grid - is done once per input rather
than once per run. `read_cache` is the other half.

**Stored with `pickle`, not `.npz`.** MRST writes `.mat`, which holds arbitrary
structs; the things worth caching here are `Grid` objects and rock structures,
not bare arrays, and `np.savez` cannot represent them. That choice has a
consequence worth stating plainly: **a pickle file executes code when loaded**,
so `read_cache` must only ever be pointed at a directory this process wrote.
It is a local scratch cache, not an interchange format.

The key is hashed rather than used directly because it is usually a filename
plus options, which does not make a safe path component.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from pymrst.utils.inout.cache_dir import cache_dir

CACHE_NAME = ".cache"


def cache_key(arg: Union[str, Sequence]) -> str:
    """A stable hash of the arguments identifying a cached result."""
    if isinstance(arg, (str, bytes)):
        material = [arg if isinstance(arg, bytes) else arg.encode()]
    else:
        material = [repr(item).encode() for item in arg]
    digest = hashlib.md5()
    for piece in material:
        digest.update(piece)
        digest.update(b"\0")          # so ["ab","c"] and ["a","bc"] differ
    return digest.hexdigest()


def cache_path(arg, directory: Optional[Path] = None) -> Path:
    """Where the entry for ``arg`` lives."""
    root = Path(directory) if directory is not None else cache_dir() / CACHE_NAME
    return root / f"{cache_key(arg)}.pkl"


def write_cache(arg, variables: Dict[str, Any],
                directory: Optional[Path] = None) -> Path:
    """Store ``variables`` under a hash of ``arg``; return the file written."""
    if not isinstance(variables, dict):
        raise TypeError(
            f"cached variables must be a name-to-value mapping, got "
            f"{type(variables).__name__}")
    path = cache_path(arg, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(variables, handle)
    return path


writeCache = write_cache


if __name__ == "__main__":
    import tempfile

    import numpy as np

    def scratch():
        return Path(tempfile.mkdtemp())

    def test_it_writes_a_file() -> None:
        d = scratch()
        path = write_cache(["case", 1], {"x": np.arange(3)}, d)
        assert path.exists() and path.parent == d
        print("ok test_it_writes_a_file")

    def test_the_same_key_gives_the_same_path() -> None:
        d = scratch()
        a = write_cache(["case", 1], {"x": 1}, d)
        b = write_cache(["case", 1], {"x": 2}, d)
        assert a == b, "the same arguments must map to one entry"
        print("ok test_the_same_key_gives_the_same_path")

    def test_different_keys_give_different_paths() -> None:
        d = scratch()
        assert write_cache(["case", 1], {"x": 1}, d) != \
            write_cache(["case", 2], {"x": 1}, d)
        print("ok test_different_keys_give_different_paths")

    def test_the_key_is_not_ambiguous() -> None:
        """['ab','c'] and ['a','bc'] must not collide."""
        assert cache_key(["ab", "c"]) != cache_key(["a", "bc"])
        print("ok test_the_key_is_not_ambiguous")

    def test_a_string_key_works() -> None:
        d = scratch()
        assert write_cache("mycase", {"x": 1}, d).exists()
        print("ok test_a_string_key_works")

    def test_it_creates_the_directory() -> None:
        d = scratch() / "nested" / "deeper"
        assert write_cache("k", {"x": 1}, d).exists()
        print("ok test_it_creates_the_directory")

    def test_a_non_mapping_is_rejected() -> None:
        try:
            write_cache("k", [1, 2, 3], scratch())
        except TypeError as error:
            assert "name-to-value mapping" in str(error)
            print("ok test_a_non_mapping_is_rejected")
            return
        raise AssertionError("a non-dict payload must be rejected")

    test_it_writes_a_file()
    test_the_same_key_gives_the_same_path()
    test_different_keys_give_different_paths()
    test_the_key_is_not_ambiguous()
    test_a_string_key_works()
    test_it_creates_the_directory()
    test_a_non_mapping_is_rejected()
    print("\nAll tests passed!")
