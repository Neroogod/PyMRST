"""Where an MRST dataset lives on this machine (MRST `getDatasetPath`).

MRST's `getDatasetPath(name)` points into `mrstDataDirectory()`, whose
default is the MRST tree's `examples/data`. PyMRST's `get_dataset_path` points
into its own data directory and offers to download. The examples try that
first and then the MRST checkout next to this repository, which ships SPE1,
SPE3, SPE9, Egg and Norne, so they run without a download.
"""

from __future__ import annotations

from pathlib import Path

MRST_DATA = Path(__file__).resolve().parents[6]/"mrst-2024a-Geomech"/"examples"/"data"


def local_dataset_path(name: str, filename: str) -> Path:
    """The directory holding `filename` for dataset `name`."""
    from pymrst.utils.dataset_manager.get_dataset_path import get_dataset_path
    candidates = []
    try:
        candidates.append(Path(get_dataset_path(name, download=False,
                                                skip_available_check=True)))
    except (KeyError, ValueError, FileNotFoundError):
        pass
    candidates += [MRST_DATA/name.upper(), MRST_DATA/name]
    for candidate in candidates:
        if (candidate/filename).is_file():
            return candidate
    raise FileNotFoundError(f"{filename} was not found; looked in "
                            + ", ".join(str(c) for c in candidates))


__all__ = ["local_dataset_path"]


if __name__ == "__main__":
    def test_the_mrst_checkout_provides_spe3():
        assert (local_dataset_path("spe3", "BENCH_SPE3.DATA")/"BENCH_SPE3.DATA").is_file()
        print("ok test_the_mrst_checkout_provides_spe3")

    def test_a_missing_file_is_reported_with_the_places_searched():
        try:
            local_dataset_path("spe3", "NO_SUCH.DATA")
        except FileNotFoundError as missing:
            assert "NO_SUCH.DATA" in str(missing)
        else:
            raise AssertionError("expected FileNotFoundError")
        print("ok test_a_missing_file_is_reported_with_the_places_searched")

    test_the_mrst_checkout_provides_spe3()
    test_a_missing_file_is_reported_with_the_places_searched()
    print("\nAll tests passed!")
