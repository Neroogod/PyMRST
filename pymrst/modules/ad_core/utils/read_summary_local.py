"""Read ECLIPSE unformatted SMSPEC/UNSMRY summary pairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


_DTYPES = {
    "INTE": ">i4",
    "REAL": ">f4",
    "DOUB": ">f8",
    "LOGI": ">i4",
}


def _record(stream):
    lead = stream.read(4)
    if not lead:
        return None
    if len(lead) != 4:
        raise EOFError("truncated Fortran record marker")
    size = struct.unpack(">i", lead)[0]
    payload = stream.read(size)
    trail = stream.read(4)
    if len(payload) != size or len(trail) != 4:
        raise EOFError("truncated Fortran record")
    if struct.unpack(">i", trail)[0] != size:
        raise ValueError("mismatched Fortran record markers")
    return payload


def _fields(path):
    with Path(path).open("rb") as stream:
        while True:
            header = _record(stream)
            if header is None:
                return
            if len(header) != 16:
                raise ValueError(f"invalid ECLIPSE keyword header ({len(header)} bytes)")
            name = header[:8].decode("ascii").strip()
            count = struct.unpack(">i", header[8:12])[0]
            encoded_type = header[12:16].decode("ascii")
            if encoded_type.startswith("C") and encoded_type[1:].isdigit():
                width = int(encoded_type[1:])
                value_type = "CHAR"
            elif encoded_type == "CHAR":
                width, value_type = 8, "CHAR"
            else:
                width, value_type = None, encoded_type

            chunks = []
            remaining = count
            while remaining:
                payload = _record(stream)
                if payload is None:
                    raise EOFError(f"missing data for {name}")
                if value_type == "CHAR":
                    if len(payload) % width:
                        raise ValueError(f"invalid character record for {name}")
                    number = len(payload) // width
                    chunks.extend(
                        payload[offset:offset + width].decode(
                            "latin-1"
                        ).rstrip()
                        for offset in range(0, len(payload), width)
                    )
                else:
                    dtype = np.dtype(_DTYPES[value_type])
                    if len(payload) % dtype.itemsize:
                        raise ValueError(f"invalid numeric record for {name}")
                    values = np.frombuffer(payload, dtype=dtype)
                    number = values.size
                    chunks.append(values)
                remaining -= number
                if remaining < 0:
                    raise ValueError(f"too many values for {name}")

            if value_type == "CHAR":
                values = chunks
            else:
                values = np.concatenate(chunks) if chunks else np.array([])
                if value_type == "LOGI":
                    values = values != 0
                elif value_type == "INTE":
                    values = values.astype(np.int64)
                else:
                    values = values.astype(float)
            yield name, value_type, values


def _read_spec(path):
    output = {}
    for name, value_type, values in _fields(path):
        entry = output.setdefault(name, {"type": value_type, "values": []})
        entry["values"].extend(values if isinstance(values, list) else values.tolist())
    for entry in output.values():
        if entry["type"] != "CHAR":
            entry["values"] = np.asarray(entry["values"])
    return output


@dataclass
class Summary:
    WGNAMES: list[str]
    KEYWORDS: list[str]
    nInx: np.ndarray
    kInx: np.ndarray
    data: np.ndarray

    def get_index(self, name=None, keyword=None):
        selected = np.ones(self.nInx.size, dtype=bool)
        if name is not None:
            selected &= self.nInx == self.WGNAMES.index(name)
        if keyword is not None:
            selected &= self.kInx == self.KEYWORDS.index(keyword)
        return selected

    getInx = get_index

    def get(self, name=None, keyword=None, ministeps=None):
        rows = self.get_index(name, keyword)
        columns = slice(None) if ministeps is None else ministeps
        return self.data[rows, columns]

    def get_names(self, keyword):
        rows = self.get_index(keyword=keyword)
        return [self.WGNAMES[index] for index in np.unique(self.nInx[rows])]

    getNms = get_names

    def get_keywords(self, name):
        rows = self.get_index(name=name)
        return [self.KEYWORDS[index] for index in np.unique(self.kInx[rows])]

    getKws = get_keywords


def read_summary_local(prefix, keywords=None):
    """Read ``prefix.SMSPEC`` and ``prefix.UNSMRY``.

    Returns ``(summary, smspec)``. Keyword filtering follows MRST and retains
    the original SMSPEC row order.
    """
    prefix = Path(prefix)
    smspec = _read_spec(prefix.with_suffix(".SMSPEC"))
    names_full = list(smspec["WGNAMES"]["values"])
    keywords_full = list(smspec["KEYWORDS"]["values"])
    if len(names_full) != len(keywords_full):
        raise ValueError("SMSPEC WGNAMES and KEYWORDS have different lengths")

    if keywords is None:
        row_index = np.arange(len(names_full))
    else:
        wanted = set(np.asarray(keywords, dtype=object).ravel().tolist())
        row_index = np.flatnonzero([keyword in wanted for keyword in keywords_full])
    names_selected = [names_full[index] for index in row_index]
    keywords_selected = [keywords_full[index] for index in row_index]

    unique_names, name_inverse = np.unique(names_selected, return_inverse=True)
    unique_keywords, keyword_inverse = np.unique(
        keywords_selected, return_inverse=True
    )

    columns = {}
    ministep = None
    for name, _, values in _fields(prefix.with_suffix(".UNSMRY")):
        if name == "MINISTEP":
            ministep = int(np.asarray(values).ravel()[0])
        elif name == "PARAMS":
            if ministep is None:
                raise ValueError("PARAMS encountered before MINISTEP")
            values = np.asarray(values)
            if values.size < len(names_full):
                raise ValueError("PARAMS shorter than SMSPEC variable list")
            columns[ministep] = values[row_index].astype(float)

    steps = sorted(columns)
    data = (
        np.column_stack([columns[step] for step in steps])
        if steps else np.empty((len(row_index), 0))
    )
    summary = Summary(
        WGNAMES=unique_names.tolist(),
        KEYWORDS=unique_keywords.tolist(),
        nInx=name_inverse.astype(np.int64),
        kInx=keyword_inverse.astype(np.int64),
        data=data,
    )
    return summary, smspec


readSummaryLocal = read_summary_local


__all__ = ["Summary", "read_summary_local", "readSummaryLocal"]
