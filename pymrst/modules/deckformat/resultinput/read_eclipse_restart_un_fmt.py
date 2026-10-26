"""MRST ``deckformat/resultinput/readEclipseRestartUnFmt.m``.

The restart data of an ECLIPSE run, as `{keyword: [value per step]}`. With
a restart specification (`spec`, or `<prefix>.RSSPEC` through
`process_eclipse_restart_spec`) each requested step's keywords are read at
the positions the specification gives; out-of-range `steps` are dropped
with a warning. Without one, everything is read
(`read_eclipse_restart_un_fmt_fallback`) and the requested steps kept.
Keyword names are made MATLAB-valid as in MRST (`FLROILI+` -> `FLROILIp`).

`steps` are MRST's 1-based step numbers.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
    fix_var_name,
)

_DTYPES = {"float64": ">f8", "float32": ">f4", "int32": ">i4"}


def read_eclipse_restart_un_fmt(prefix: Any, spec: Any = None,
                                steps: Optional[Sequence[int]] = None
                                ) -> Optional[Dict[str, List[Any]]]:
    """MRST `output = readEclipseRestartUnFmt(prefix, spec, steps)`."""
    from pymrst.modules.deckformat.resultinput.process_eclipse_restart_spec import (
        process_eclipse_restart_spec,
    )
    prefix = str(prefix)
    if spec is None and os.path.exists(prefix + ".RSSPEC"):
        spec = process_eclipse_restart_spec(prefix)
    if spec is None:
        n_steps = np.inf if steps is None or len(steps) == 0 else max(steps)
        output = read_eclipse_restart_un_fmt_fallback(prefix, nSteps=n_steps)
        out: Dict[str, List[Any]] = {}
        for name, field in output.items():
            values = field.values if hasattr(field, "values") else field
            out[name] = (list(values) if steps is None or len(steps) == 0
                         else [values[s - 1] if s - 1 < len(values) else None for s in steps])
        return out
    n_spec = np.size(spec.time)
    steps = list(range(1, n_spec + 1)) if steps is None or len(steps) == 0 else list(steps)
    ok = [1 <= s <= n_spec for s in steps]
    if not all(ok):
        warnings.warn("Ignoring out-of-range entries of input-vector 'steps'.")
        steps = [s for s, o in zip(steps, ok) if o]
        if not steps:
            return None
    longest = max(range(len(steps)), key=lambda i: len(spec.keywords[steps[i] - 1]))
    out = {fix_var_name(n): [None]*len(steps) for n in spec.keywords[steps[longest] - 1]}
    unified = spec.type == "unified"
    fid = open(prefix + ".UNRST", "rb") if unified else None
    try:
        for ks, step in enumerate(steps):
            if not unified:
                fid = open(spec.fnames[step - 1], "rb")
            names = [fix_var_name(n) for n in spec.keywords[step - 1]]
            pointers = spec.pointers[step - 1]
            counts = spec.num[step - 1]
            precs = spec.prec[step - 1]
            for kf, name in enumerate(names):
                if counts[kf] > 0:
                    fid.seek(int(pointers[kf]) + 28)
                    out.setdefault(name, [None]*len(steps))[ks] = _fread(fid, int(counts[kf]),
                                                                         precs[kf])
            if not unified:
                fid.close()
    finally:
        if unified and fid is not None:
            fid.close()
    return out


readEclipseRestartUnFmt = read_eclipse_restart_un_fmt


def read_eclipse_restart_un_fmt_fallback(prefix: Any, *, RestartFields: Sequence[str] = (),
                                         nSteps: float = np.inf):
    """MRST `[rstrt, rsspec] = readEclipseRestartUnFmt_fallback(prefix, ...)`: the
    whole restart, from the `.X####` files or else the unified `.UNRST`."""
    import re
    from pathlib import Path
    from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
        EclipseField,
        read_eclipse_output_file_un_fmt,
    )
    path = Path(str(prefix))
    dname = path.parent
    files = sorted(f for f in os.listdir(dname)
                   if re.fullmatch(re.escape(path.name) + r"\.X\d{4}", f))
    if files:
        rstrt: Dict[str, Any] = {}
        for f in files:
            item = read_eclipse_output_file_un_fmt(dname/f)
            fields = list(RestartFields) if RestartFields else list(item)
            for fld in fields:
                rstrt.setdefault(fld, EclipseField(fld, "", []))
                if fld in item:
                    rstrt[fld].values.append(item[fld].values)
        return rstrt
    return read_eclipse_output_file_un_fmt(dname/(path.name + ".UNRST"), cellOutput=True,
                                           maxCellSize=nSteps)


readEclipseRestartUnFmt_fallback = read_eclipse_restart_un_fmt_fallback


def _fread(fid, n: int, prec: str) -> Any:
    """MATLAB `fread(fid, n, prec, 8)`: blocks of `prec`'s count, then 8 bytes skipped."""
    block, _, kind = prec.partition("*")
    block = int(block)
    if kind == "uchar=>char":
        chars = _read_blocks(fid, 8*n, block, 1, "S1")
        text = b"".join(chars.tolist()).decode("latin-1")
        return [text[8*i:8*i + 8].rstrip() for i in range(n)]
    dtype = np.dtype(_DTYPES[kind])
    return _read_blocks(fid, n, block, dtype.itemsize, dtype).astype(float)


def _read_blocks(fid, n: int, block: int, size: int, dtype) -> np.ndarray:
    parts = []
    left = n
    while left > 0:
        m = min(block, left)
        parts.append(np.frombuffer(fid.read(m*size), dtype=dtype, count=m))
        left -= m
        if left > 0:
            fid.seek(8, 1)
    return np.concatenate(parts) if parts else np.zeros(0)


__all__ = ["read_eclipse_restart_un_fmt", "read_eclipse_restart_un_fmt_fallback",
           "readEclipseRestartUnFmt", "readEclipseRestartUnFmt_fallback"]


if __name__ == "__main__":
    import io
    import struct
    from pathlib import Path

    NORNE = (Path(__file__).resolve().parents[5] / "mrst-2024a-Geomech" / "examples" / "data"
             / "norne_field_model" / "opm-tests" / "norne" / "opm-simulation-reference"
             / "flow_legacy" / "NORNE_ATW2013")

    def test_fread_skips_record_markers_between_blocks():
        values = np.arange(1500, dtype=">f8")
        raw = values[:1000].tobytes() + b"\0"*8 + values[1000:].tobytes()
        got = _fread(io.BytesIO(raw), 1500, "1000*float64")
        np.testing.assert_array_equal(got, np.arange(1500))
        raw = b"PRESSURE" + b"SWAT    "
        assert _fread(io.BytesIO(raw), 2, "840*uchar=>char") == ["PRESSURE", "SWAT"]
        print("ok test_fread_skips_record_markers_between_blocks")

    def test_norne_restart_without_rsspec_reads_every_step():
        if not Path(str(NORNE) + ".UNRST").exists():
            print("skip test_norne_restart_without_rsspec_reads_every_step")
            return
        rst = read_eclipse_restart_un_fmt(NORNE)
        assert len(rst["PRESSURE"]) == 65 and np.size(rst["PRESSURE"][0]) == 44431
        two = read_eclipse_restart_un_fmt(NORNE, None, [1, 3])
        np.testing.assert_array_equal(two["PRESSURE"][1], rst["PRESSURE"][2])
        print("ok test_norne_restart_without_rsspec_reads_every_step")

    test_fread_skips_record_markers_between_blocks()
    test_norne_restart_without_rsspec_reads_every_step()
    print("\nAll tests passed!")
