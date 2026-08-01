"""Read SAM simulator rock files (MRST ``readSimRock.m``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pymrst.utils.units import DARCY, MILLI


def _read_records(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"SAM rock file does not exist: {path}")
    values = np.fromstring(path.read_text(), sep=" ")
    if values.size < 1:
        raise ValueError(f"SAM rock file is empty: {path}")
    rows = int(values[0])
    payload = values[1:]
    if rows < 0 or (rows and payload.size % rows):
        raise ValueError(f"invalid SAM record count in {path}")
    columns = 0 if rows == 0 else payload.size // rows
    return payload.reshape(rows, columns)


def read_sim_rock(casename):
    """Read ``-perm.dat`` and ``-poro.dat`` for a SAM simulation case."""
    # Local import keeps pymrst.utils.inout safe while params/gridprocessing
    # packages are mutually initializing through the legacy deck I/O exports.
    from pymrst.params.rock.rock import Rock

    base = Path(casename)
    permeability = _read_records(Path(f"{base}-perm.dat")) * (MILLI * DARCY)
    porosity = _read_records(Path(f"{base}-poro.dat"))
    if permeability.shape[0] != porosity.shape[0]:
        raise ValueError("permeability and porosity record counts do not match")
    return Rock(permeability, porosity.reshape(-1))


readSimRock = read_sim_rock

__all__ = ["readSimRock", "read_sim_rock"]
