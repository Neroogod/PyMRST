"""Read and write ECLIPSE unformatted keyword records (M2.6).

The record layer is *not* reimplemented here. `read_summary_local` already
reads real ECLIPSE SMSPEC/UNSMRY files, and its `_fields` generator carries
the parts that are fiddly and easy to get subtly wrong: Fortran unformatted
block markers, the big-endian INTE/REAL/DOUB/LOGI map, CHAR width handling and
chunked-block reassembly. Reusing it means the layer under every reader below
has already been exercised against files this project did not write.

The writer here exists for one reason: **there is no ECLIPSE binary output
anywhere in this repository or its dataset cache to test against.** Without a
writer, a reader could only be verified by reading its own imagination.

That is still not sufficient on its own. A reader and writer that share a
misunderstanding round-trip perfectly - the same trap the transport Jacobian's
finite-difference check turned out to have, where residual and Jacobian agreed
because both used the same wrong upwind. So round-trip is the weaker half, and
the readers above this are additionally cross-checked against the *text*
GRDECL path, which is independently validated against MATLAB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
import struct

import numpy as np

# The one import from the replica layer, and deliberately narrow: this reads a
# validated record parser rather than copying it. Nothing here is imported by
# the replica layer, so the covenant holds in the direction that matters.
from pymrst.modules.ad_core.utils.read_summary_local import _fields

_WRITE_DTYPES = {
    "INTE": ">i4",
    "REAL": ">f4",
    "DOUB": ">f8",
    "LOGI": ">i4",
}

# ECLIPSE writes long arrays in blocks. These limits are part of the format,
# not a tuning choice: a reader that assumes one block per keyword fails on
# any real field case.
_NUMERIC_BLOCK = 1000
_CHAR_BLOCK = 105


def read_records(path: str | Path) -> list[tuple[str, str, Any]]:
    """Every `(keyword, type, values)` record in an unformatted file."""
    return list(_fields(Path(path)))


def read_keywords(path: str | Path) -> dict[str, Any]:
    """Records as a mapping, keeping the *first* of any repeated keyword.

    Repetition is meaningful in UNRST, where each report step repeats the same
    keywords, so anything needing per-step data must use `read_records` and
    split on SEQNUM. This helper is for files where a keyword appears once.
    """
    out: dict[str, Any] = {}
    for name, _, values in _fields(Path(path)):
        out.setdefault(name, values)
    return out


def _write_record(stream, payload: bytes) -> None:
    marker = struct.pack(">i", len(payload))
    stream.write(marker)
    stream.write(payload)
    stream.write(marker)


def write_records(path: str | Path, records) -> Path:
    """Write `(keyword, type, values)` records in ECLIPSE unformatted form."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        for name, value_type, values in records:
            if len(name) > 8:
                raise ValueError(f"keyword {name!r} exceeds eight characters")
            if value_type == "CHAR":
                items = [str(v) for v in values]
                count = len(items)
            elif value_type == "MESS":
                items, count = [], 0
            else:
                items = np.asarray(values, dtype=_WRITE_DTYPES[value_type])
                count = items.size

            header = (f"{name:<8}".encode("ascii")
                      + struct.pack(">i", count)
                      + f"{value_type:<4}".encode("ascii"))
            _write_record(stream, header)

            if value_type == "MESS":
                continue
            if value_type == "CHAR":
                for start in range(0, count, _CHAR_BLOCK):
                    block = items[start:start + _CHAR_BLOCK]
                    _write_record(
                        stream,
                        b"".join(f"{v:<8}".encode("latin-1")[:8] for v in block),
                    )
                if count == 0:
                    _write_record(stream, b"")
            else:
                for start in range(0, count, _NUMERIC_BLOCK):
                    _write_record(stream, items[start:start + _NUMERIC_BLOCK].tobytes())
                if count == 0:
                    _write_record(stream, b"")
    return destination


__all__ = ["read_keywords", "read_records", "write_records"]
