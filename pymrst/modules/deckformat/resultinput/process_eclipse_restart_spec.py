"""MRST ``deckformat/resultinput/processEclipseRestartSpec.m``.

The index of an ECLIPSE restart: from `<prefix>.RSSPEC`, the report times
and dates, the unit system, whether the restart is unified (`.UNRST`) or
one file per step (`.X####`), and per step the keywords present with their
byte positions, read precisions and element counts. With `restartAmount`
'default', only MRST's default keywords are kept. `ZTRACER` entries take the
name of the tracer that follows, repeated aquifer keywords are numbered, and
LGR blocks are split off (into `specLGR` when asked for).

Precisions keep MRST's `fread` strings ('1000*float64', '840*uchar=>char',
...): values come in blocks of 1000 (strings 105 x 8 characters) between
8-byte record markers.
"""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np

from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
    read_eclipse_output_file_un_fmt,
)

PRECISIONS = {"CHAR": "840*uchar=>char", "DOUB": "1000*float64", "INTE": "1000*int32",
              "LOGI": "1000*int32", "MESS": "blunder", "REAL": "1000*float32"}

DEFAULT_FIELDS = (
    "ACAQ", "CNV_OIL", "CNV_GAS", "CNV_WAT", "CNV_DOIL", "CNV_DGAS", "CNV_DWAT", "DEN",
    "DLYTIM", "DOUBHEAD", "DUDF", "DUDW", "FLR", "IAAQ", "IACN", "IACT", "IAQN", "ICAQ",
    "ICON", "IDWL", "IGPH", "IGRP", "ILBR", "ILBS", "INTEHEAD", "ISEG", "IUAD", "IUAP",
    "IUDQ", "IWEL", "IWLS", "LGR", "LOGIHEAD", "PADS", "PBUB", "POLYMAX", "POLYMER",
    "PRESSURE", "RAQN", "REGDIMS", "REGRPT", "RS", "RSEG", "RV", "SAAQ", "SACN", "SACT",
    "SCAQ", "SCON", "SDWL", "SEQNUM", "SGAS", "SGRP", "SOIL", "SWAT", "SWEL", "VGAS",
    "VISC", "VOIL", "XAAQ", "XCON", "XGRP", "XMF", "XWEL", "YMF", "ZACN", "ZACT", "ZDWL",
    "ZGRP", "ZLACT", "ZMF", "ZUDL", "ZUDN", "ZWEL", "ZWLS")


def process_eclipse_restart_spec(prefix: Any, restartAmount: str = "default",
                                 nargout: int = 1):
    """MRST `[spec, specLGR] = processEclipseRestartSpec(prefix, restartAmount)`."""
    path = Path(str(prefix))
    ext = path.suffix if path.suffix else ".RSSPEC"
    if ext != ".RSSPEC":
        raise ValueError(f"Unexpected file format: {ext}")
    filename = path.with_suffix(".RSSPEC") if path.suffix else Path(str(prefix) + ".RSSPEC")
    rsspec = read_eclipse_output_file_un_fmt(filename)
    ih = np.ravel(rsspec["INTEHEAD"].values)
    unit = ("metric", "field", "lab")[int(ih[2]) - 1]
    t = np.ravel(rsspec["TIME"].values)
    d = np.ravel(rsspec["ITIME"].values)
    n_steps = t.size
    date = d.reshape(n_steps, d.size//n_steps)
    rep_num = date[:, 0]
    date = date[:, 1:4]
    names = [str(x).strip() for x in rsspec["NAME"].values]
    first = names[0]
    if first == "SEQNUM":
        kind = "unified"
    elif first == "INTEHEAD":
        kind = "multiple"
    else:
        kind = "unknown"
        warnings.warn(f"Unexpected format of file: {filename}")
    nf = [i for i, n in enumerate(names) if n == first]
    types = [str(x).strip() for x in rsspec["TYPE"].values]
    pointer = np.ravel(rsspec["POINTER"].values).astype(np.int64)
    pointerb = np.ravel(rsspec["POINTERB"].values).astype(np.int64)
    number = np.ravel(rsspec["NUMBER"].values).astype(np.int64)
    if len(nf) != n_steps:
        warnings.warn("Unable to output keyword-info")
        keywords, pointers, prec, num = [], [], [], []
    else:
        keywords, pointers, prec, num = ([None]*n_steps for _ in range(4))
        ix = nf + [len(names)]
        for k in range(n_steps):
            subix = np.arange(ix[k], ix[k + 1])
            if restartAmount == "default":
                subix = subix[_field_ix([names[i] for i in subix], ())]
            keywords[k] = [names[i] for i in subix]
            pointers[k] = pointerb[subix]*2**31 + pointer[subix]
            prec[k] = [PRECISIONS.get(types[i], types[i]) for i in subix]
            num[k] = number[subix]
    spec = SimpleNamespace(time=t, date=date, unit=unit, type=kind, keywords=keywords,
                           pointers=pointers, prec=prec, num=num, repNum=rep_num)
    if kind == "multiple":
        spec.fnames = _restart_files(prefix, spec)
    for k in range(t.size):
        kws = spec.keywords[k]
        for kz in [i for i, n in enumerate(kws) if n == "ZTRACER"]:
            kws[kz] = "ZTRACER_" + kws[kz + 1]
    for k in range(t.size):
        kws = spec.keywords[k]
        for fld in ("ICAQNUM", "SCAQNUM", "ACAQNUM"):
            for count, i in enumerate([i for i, n in enumerate(kws) if n == fld], 1):
                kws[i] = f"{kws[i]}_{count}"
                kws[i + 1] = f"{kws[i + 1]}_{count}"
    spec_lgr = None
    if nargout > 1:
        nlgr = max(sum(n == "LGR" for n in kws) for kws in spec.keywords)
        spec_lgr = [copy.deepcopy(spec) for _ in range(nlgr)]
        lgr_names: List[str] = []
        ix_lgr = 0
    for k in range(t.size):
        kws = spec.keywords[k]
        ixl = [i for i, n in enumerate(kws) if n == "LGR"]
        if not ixl:
            continue
        ixr = [i for i, n in enumerate(kws) if n == "ENDLGR"]
        keep = np.ones(len(kws), dtype=bool)
        for lgr_num, start in enumerate(ixl):
            ix = np.arange(start, ixr[lgr_num] + 1)
            keep[ix] = False
            if nargout > 1:
                cur = str(rsspec["LGRNAME"].values[ix_lgr]).strip()
                ix_lgr += 1
                if cur not in lgr_names:
                    lgr_names.append(cur)
                s_lgr = spec_lgr[lgr_names.index(cur)]
                for f in ("keywords", "prec", "pointers", "num"):
                    getattr(s_lgr, f)[k] = _take(getattr(s_lgr, f)[k], ix)
                s_lgr.name = cur
        for f in ("keywords", "prec", "pointers", "num"):
            getattr(spec, f)[k] = _take(getattr(spec, f)[k], np.flatnonzero(keep))
    return (spec, spec_lgr) if nargout > 1 else spec


processEclipseRestartSpec = process_eclipse_restart_spec


def _restart_files(prefix: Any, spec: SimpleNamespace) -> List[Any]:
    directory = Path(str(prefix)).parent
    stem = Path(str(prefix)).name
    fnms = sorted(p.name for p in directory.glob(stem + ".X*"))
    mtch = [re.search(r".X[0-9][0-9][0-9][0-9]", f) for f in fnms]
    fnms = [f for f, m in zip(fnms, mtch) if m]
    num = np.array([int(m.group(0)[2:]) for m in mtch if m])
    if num.size and num.min() == 0:
        num = num + 1
    order = np.argsort(num, kind="stable")
    num = num[order]
    fnms = [str(directory/fnms[i]) for i in order]
    nspec = spec.time.size
    if len(fnms) == nspec:
        return sorted(fnms)
    subix = np.isin(spec.repNum, num)
    assert np.count_nonzero(subix) == num.size, \
        "Unable to match multiple restart-files to RSSPEC"
    out: List[Any] = [None]*nspec
    for i, f in zip(np.flatnonzero(subix), fnms):
        out[i] = f
    return out


def _field_ix(nms: Sequence[str], additional: Sequence[str]) -> np.ndarray:
    inc = tuple(DEFAULT_FIELDS) + tuple(additional)
    return np.array([any(n.startswith(f) for f in inc) for n in nms], dtype=bool)


def _take(values: Any, ix: np.ndarray) -> Any:
    if isinstance(values, np.ndarray):
        return values[ix]
    return [values[i] for i in ix]


__all__ = ["process_eclipse_restart_spec", "processEclipseRestartSpec"]


if __name__ == "__main__":
    def test_default_fields_match_by_prefix():
        ix = _field_ix(["PRESSURE", "FLROILI+", "HIDDEN", "SWAT"], ())
        np.testing.assert_array_equal(ix, [True, True, False, True])
        print("ok test_default_fields_match_by_prefix")

    def test_a_non_rsspec_file_is_refused():
        try:
            process_eclipse_restart_spec("CASE.UNRST")
        except ValueError:
            print("ok test_a_non_rsspec_file_is_refused")
            return
        raise AssertionError("only .RSSPEC is accepted, as in MRST")

    def test_precisions_are_mrsts_fread_strings():
        assert PRECISIONS["DOUB"] == "1000*float64"
        assert PRECISIONS["CHAR"] == "840*uchar=>char"
        print("ok test_precisions_are_mrsts_fread_strings")

    test_default_fields_match_by_prefix()
    test_a_non_rsspec_file_is_refused()
    test_precisions_are_mrsts_fread_strings()
    print("\nAll tests passed!")
