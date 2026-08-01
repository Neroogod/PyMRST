"""Official SPE10 Model I property loader."""

from __future__ import annotations

from pathlib import Path
import hashlib
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat

SPE10_URL = ("https://www.sintef.no/contentassets/"
             "124f261f170947a6bc51dd76aea66129/SPE10.zip")
SPE10_SHA256 = "f4063515f426238a486141424107b112d546a75aa5d3c5defa368783c119abe6"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def spe10_data_directory(root=None) -> Path:
    """Locate the official archive after MRST-style extraction."""
    base = (Path(root) if root is not None else
            _workspace_root().parent/"mrst-2024a-Geomech/examples/data/SPE10")
    for candidate in (base, base/"SPE10"):
        if (candidate/"model1_data.mat").is_file():
            return candidate
    raise FileNotFoundError(
        f"SPE10 Model I data is missing under {base}. "
        "Call fetch_spe10_dataset() to install the official SINTEF archive."
    )


def fetch_spe10_dataset(destination=None, *, overwrite=False) -> Path:
    """Download, checksum, and extract SINTEF's official SPE10 archive."""
    destination = (Path(destination) if destination is not None else
                   _workspace_root().parent/"mrst-2024a-Geomech/examples/data/SPE10")
    if not overwrite:
        try:
            return spe10_data_directory(destination)
        except FileNotFoundError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination/"SPE10.zip"
    urllib.request.urlretrieve(SPE10_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SPE10_SHA256:
        raise IOError(f"SPE10 archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    return spe10_data_directory(destination)


def load_spe10_model1(*, layers=None, active=None, data_root=None) -> dict:
    """Load Model I with optional one-based layers and active-cell selection."""
    source = spe10_data_directory(data_root)/"model1_data.mat"
    data = loadmat(source, squeeze_me=True, struct_as_record=False)
    rock = data["rock"]
    dims = (100, 1, 20)
    count = int(np.prod(dims))
    selected = np.ones(count, dtype=bool)
    if layers is not None:
        layer_ids = np.atleast_1d(np.asarray(layers, dtype=int))
        if np.any((layer_ids < 1) | (layer_ids > dims[2])):
            raise ValueError("SPE10 Model I layers must be in 1..20")
        k = np.arange(count)//(dims[0]*dims[1]) + 1
        selected &= np.isin(k, layer_ids)
    if active is not None:
        raw = np.asarray(active)
        if raw.dtype == bool:
            if raw.size != count:
                raise ValueError("Boolean active mask must contain 2000 entries")
            selected &= raw.ravel()
        else:
            mask = np.zeros(count, dtype=bool)
            ids = np.atleast_1d(raw).astype(int)
            if np.any((ids < 0) | (ids >= count)):
                raise ValueError("active cell IDs must be zero-based in 0..1999")
            mask[ids] = True
            selected &= mask
    return {
        "cartDims": dims, "active": selected,
        "indexMap": np.flatnonzero(selected),
        "perm": np.asarray(rock.perm, dtype=float)[selected],
        "poro": np.asarray(rock.poro, dtype=float).ravel()[selected],
        "sgof": np.asarray(data["kr_deck"].PROPS.SGOF, dtype=float),
        "source": source,
    }
