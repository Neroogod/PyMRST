"""Locating and installing the EGG benchmark dataset.

No MRST counterpart, and the directory it sits in has none either - MRST has no
`modules/deckformat/egg`. MRST registers the dataset in
`utils/dataset_manager/datasets/dataset_egg.m` (mirrored at
`pymrst/utils/dataset_manager/datasets/dataset_egg.py`) and locates it with
`getDatasetPath('egg')`, which looks only under `mrstDataDirectory()`. The Egg
data ships *inside the MRST tree* in this workspace, at
`mrst-2024a-Geomech/examples/data/Egg`, so `get_dataset_path` alone does not
find it. That gap is what these two functions close, and they check MRST's own
location first.

This file used to also carry `read_egg_deck`, a bespoke Egg-only parser
returning a flat dictionary rather than a deck. It has been removed. Its output
could not be handed to `init_eclipse_grid`, `init_eclipse_rock`,
`init_deck_adi_fluid` or `init_eclipse_problem_ad_ad` - the four functions MRST's
own Egg workflow is built out of - which is what made the old `eggExample` port
a re-implementation instead of a transcription. The deck is now read by
`pymrst.modules.test_suite.dataset_functions.egg.get_deck_egg`, which is
`readEclipseDeck` + `convertDeckUnits` exactly as `getDeckEGG.m` calls them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request
import zipfile

ARTICLE = "0e53ddd5-33b7-44f2-9ca4-921880825e47"
FILE = "5867bc9e-4bf6-4e07-827c-4a6669a539ce"
DOWNLOAD_URL = f"https://data.4tu.nl/file/{ARTICLE}/{FILE}"
MD5 = "50d270d9f769f2ca4bb1a8c10b49852e"

#: The one file whose presence means the dataset is installed rather than
#: half-extracted. The archive unpacks to two nested layouts depending on how
#: it was extracted, so the marker is checked rather than the directory.
MARKER = Path("MRST")/"Egg_Model_ECL.DATA"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bundled_root() -> Path:
    return _workspace_root().parent/"mrst-2024a-Geomech"/"examples"/"data"/"Egg"


def _candidates(root=None):
    """Where to look, MRST's own location first."""
    if root is not None:
        base = Path(root)
        yield from (base, base/"Egg_Model_Data_Files_v2")
        return
    try:
        from pymrst.utils.dataset_manager.get_dataset_path import (
            get_dataset_path,
        )
        registered = Path(get_dataset_path("egg", download=False,
                                           skip_available_check=True))
        yield from (registered, registered/"Egg_Model_Data_Files_v2")
    except Exception:
        # An unregistered dataset or an unreadable settings file must not stop
        # the bundled copy from being found.
        pass
    bundled = _bundled_root()
    yield from (bundled, bundled/"Egg_Model_Data_Files_v2")


def egg_data_directory(root=None) -> Path:
    """The installed dataset directory, accepting both archive layouts."""
    seen = []
    for candidate in _candidates(root):
        seen.append(candidate)
        if (candidate/MARKER).is_file():
            return candidate
    raise FileNotFoundError(
        "the EGG dataset is not installed. Looked in "
        + ", ".join(str(path) for path in seen)
        + ". Call fetch_egg_dataset(), or extract the official data.zip so "
        f"that {MARKER} exists under one of them."
    )


def fetch_egg_dataset(destination=None, *, overwrite=False) -> Path:
    """Download and verify the official public 4TU EGG archive."""
    destination = Path(destination) if destination else _bundled_root()
    if not overwrite:
        try:
            return egg_data_directory(destination)
        except FileNotFoundError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/"data.zip"
    urllib.request.urlretrieve(DOWNLOAD_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise IOError("downloaded EGG archive failed its official MD5 checksum")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    return egg_data_directory(destination)


__all__ = ["DOWNLOAD_URL", "MD5", "egg_data_directory", "fetch_egg_dataset"]


if __name__ == "__main__":
    import tempfile

    def test_the_bundled_dataset_is_found():
        try:
            found = egg_data_directory()
        except FileNotFoundError:
            print("skip test_the_bundled_dataset_is_found (not installed)")
            return
        assert (found/MARKER).is_file(), found
        print(f"ok test_the_bundled_dataset_is_found ({found.name})")

    def test_both_archive_layouts_are_accepted():
        """The zip unpacks with or without the `Egg_Model_Data_Files_v2` level
        depending on the extractor. Accepting only one turns a correctly
        downloaded dataset into a missing one."""
        for inner in ("", "Egg_Model_Data_Files_v2"):
            root = Path(tempfile.mkdtemp())
            (root/inner/"MRST").mkdir(parents=True)
            (root/inner/MARKER).write_text("RUNSPEC\n")
            assert egg_data_directory(root) == (root/inner if inner else root)
        print("ok test_both_archive_layouts_are_accepted")

    def test_a_missing_dataset_names_where_it_looked():
        """A bare "not found" leaves the user guessing which of three
        directories to extract into."""
        root = Path(tempfile.mkdtemp())
        try:
            egg_data_directory(root)
        except FileNotFoundError as refused:
            assert str(root) in str(refused), refused
            assert "fetch_egg_dataset" in str(refused)
            print("ok test_a_missing_dataset_names_where_it_looked")
            return
        raise AssertionError("an empty directory is not an installed dataset")

    def test_an_explicit_root_does_not_fall_back_to_the_bundled_copy():
        """A caller pointing at their own copy must get theirs or an error -
        silently reading the workspace's copy would make a per-realization
        study read the wrong permeability and still run."""
        root = Path(tempfile.mkdtemp())
        try:
            egg_data_directory(root)
        except FileNotFoundError:
            print("ok test_an_explicit_root_does_not_fall_back_to_the_bundled_copy")
            return
        raise AssertionError("an explicit root must not fall back")

    def test_the_bespoke_parser_is_gone():
        """It returned a flat dictionary rather than a deck, so nothing in the
        MRST chain could consume it. `get_deck_egg` replaced it."""
        import pymrst.modules.deckformat as deckformat
        assert not hasattr(deckformat, "read_egg_deck")
        from pymrst.modules.test_suite.dataset_functions.egg import (
            get_deck_egg,
        )
        assert callable(get_deck_egg)
        print("ok test_the_bespoke_parser_is_gone")

    test_the_bundled_dataset_is_found()
    test_both_archive_layouts_are_accepted()
    test_a_missing_dataset_names_where_it_looked()
    test_an_explicit_root_does_not_fall_back_to_the_bundled_copy()
    test_the_bespoke_parser_is_gone()
    print("\nAll tests passed!")
