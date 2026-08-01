"""Official MRST ``simplecomp`` E300 and AD-GPRS reference results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat


SIMPLECOMP_URL = (
    "https://www.sintef.no/contentassets/124f261f170947a6bc51dd76aea66129/"
    "simple_comp.zip"
)
SIMPLECOMP_SHA256 = "5df7b7ab54927b2828b561572aed3b40eca0f3a4b33374c8ed2558b963053150"


@dataclass(frozen=True)
class SimpleCompReferenceState:
    """Value-only state imported from MRST's comparison MAT-file."""

    pressure: np.ndarray
    components: np.ndarray
    temperature: np.ndarray | None = None
    saturation: np.ndarray | None = None
    liquid_fraction: np.ndarray | None = None
    liquid_composition: np.ndarray | None = None
    vapor_composition: np.ndarray | None = None
    viscosity: np.ndarray | None = None
    density: np.ndarray | None = None


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[5]


def simplecomp_data_directory(root=None) -> Path:
    """Return the installed official dataset directory."""
    base = Path(root) if root is not None else (
        _workspace_root().parent / "mrst-2024a-Geomech/examples/data/simplecomp"
    )
    for candidate in (base, base / "simplecomp"):
        if (candidate / "comparison.mat").is_file() and (candidate / "SIMPLE_COMP.DATA").is_file():
            return candidate
    raise FileNotFoundError(
        f"simplecomp data is missing under {base}. Call fetch_simplecomp_dataset() "
        "to install MRST's official SINTEF archive."
    )


def fetch_simplecomp_dataset(destination=None, *, overwrite: bool = False) -> Path:
    """Download, verify, and extract MRST's official simplecomp archive."""
    destination = Path(destination) if destination is not None else (
        _workspace_root().parent / "mrst-2024a-Geomech/examples/data/simplecomp"
    )
    if not overwrite:
        try:
            return simplecomp_data_directory(destination)
        except FileNotFoundError:
            pass
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "simple_comp.zip"
    urllib.request.urlretrieve(SIMPLECOMP_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SIMPLECOMP_SHA256:
        raise IOError(f"simplecomp archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as bundle:
        expected = {"comparison.mat", "SIMPLE_COMP.DATA"}
        if set(bundle.namelist()) != expected:
            raise IOError("simplecomp archive has an unexpected layout")
        bundle.extractall(destination)
    archive.unlink()
    return simplecomp_data_directory(destination)


def _matrix(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        return np.column_stack([
            np.asarray(column, dtype=float).reshape(-1) for column in array.reshape(-1)
        ])
    return np.asarray(array, dtype=float)


def _optional(raw, name, *, matrix=False):
    if not hasattr(raw, name):
        return None
    value = getattr(raw, name)
    return _matrix(value) if matrix else np.asarray(value, dtype=float).copy()


def _convert_state(raw) -> SimpleCompReferenceState:
    return SimpleCompReferenceState(
        pressure=np.asarray(raw.pressure, dtype=float).reshape(-1).copy(),
        components=_matrix(raw.components),
        temperature=_optional(raw, "T"),
        saturation=_optional(raw, "s"),
        liquid_fraction=_optional(raw, "L"),
        liquid_composition=_optional(raw, "x", matrix=True),
        vapor_composition=_optional(raw, "y", matrix=True),
        viscosity=_optional(raw, "mu"),
        density=_optional(raw, "rho"),
    )


def load_simplecomp_references(*, data_root=None) -> dict:
    """Load MRST-aligned 499-step E300 and AD-GPRS histories."""
    source = simplecomp_data_directory(data_root) / "comparison.mat"
    raw = loadmat(source, squeeze_me=True, struct_as_record=False)
    eclipse_all = tuple(_convert_state(state) for state in np.asarray(raw["statesECL"]).reshape(-1))
    gprs = tuple(_convert_state(state) for state in np.asarray(raw["statesGPRS"]).reshape(-1))
    eclipse = eclipse_all[1:]
    if len(eclipse) != len(gprs):
        raise ValueError(
            f"unaligned simplecomp references: E300={len(eclipse)}, AD-GPRS={len(gprs)}"
        )
    return {
        "E300": eclipse,
        "AD-GPRS": gprs,
        "E300_initial": eclipse_all[0],
        "source": source,
        "report_steps": len(gprs),
    }


__all__ = [
    "SIMPLECOMP_SHA256",
    "SIMPLECOMP_URL",
    "SimpleCompReferenceState",
    "fetch_simplecomp_dataset",
    "load_simplecomp_references",
    "simplecomp_data_directory",
]
