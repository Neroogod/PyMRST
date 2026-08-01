"""Where cached results live (MRST ``utils/inout/cacheDir.m``).

MRST returns `/tmp/` on unix and `./` elsewhere. Hard-coding `/tmp` is wrong on
Windows and rude on a shared machine, so this asks the platform - which is what
MRST was approximating.
"""

import tempfile
from pathlib import Path


def cache_dir() -> Path:
    """The directory for cached results."""
    return Path(tempfile.gettempdir())


cacheDir = cache_dir


if __name__ == "__main__":

    def test_it_exists_and_is_a_directory() -> None:
        assert cache_dir().is_dir()
        print("ok test_it_exists_and_is_a_directory")

    def test_it_is_writable() -> None:
        probe = cache_dir() / "pymrst_cache_probe"
        probe.write_text("x")
        assert probe.read_text() == "x"
        probe.unlink()
        print("ok test_it_is_writable")

    def test_it_is_absolute() -> None:
        assert cache_dir().is_absolute()
        print("ok test_it_is_absolute")

    test_it_exists_and_is_a_directory()
    test_it_is_writable()
    test_it_is_absolute()
    print("\nAll tests passed!")
