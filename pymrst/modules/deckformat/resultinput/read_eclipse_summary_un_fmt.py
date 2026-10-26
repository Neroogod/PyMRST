"""MRST ``deckformat/resultinput/readEclipseSummaryUnFmt.m``.

Read an unformatted ECLIPSE summary: the specification `<prefix>.SMSPEC`
and the vectors from the unified `<prefix>.UNSMRY` or, failing that, the
separate `<prefix>.S####` files. Returns `(smry, smspec)`, where `smry` has
MRST's fields (`WGNAMES`, `KEYWORDS`, `UNITS`, `STARTDAT`, `NUMS`, `nInx`,
`nInxOrder`, `kInx`, `kInxOrder`, `data`, `ministep`, `cartDims`) and
accessors:
- `get(nm, kw, ms, trimData)`: the rows for entity `nm` and keyword `kw`
  (either may be empty for "all", at most one a list) at ministep columns
  `ms` (empty or ':' for all); `nargout=2`/`3` also returns the unit and
  the spec (cell `[i j k]` for B/C keywords, region pairs for R);
- `getInx(nm, kw)`, `getNms(kw)`, `getKws(nm)`, `getUnit(nm, kw)`.

Port conventions: `nInx`, `kInx`, `nInxOrder`, `kInxOrder` and the
ministep positions `ms` are 0-based; `NUMS` and the cell spec stay the
1-based ECLIPSE numbers MRST returns. Empty names and keywords read as
'empty', as in MRST. MRST keeps `UNITS` and `NUMS` unfiltered when
`keyWords` is given, so units and specs then come from the leading rows;
that is reproduced. MRST prints progress under `mrstVerbose` (off by
default); PyMRST has no such setting, so nothing is printed.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.deckformat.resultinput.read_eclipse_output_file_un_fmt import (
    iter_eclipse_output_records, read_eclipse_output_file_un_fmt,
)
from pymrst.utils.block_diag_index import block_diag_index


def read_eclipse_summary_un_fmt(prefix: Any, keyWords: Optional[Sequence[str]] = None
                                ) -> Tuple[SimpleNamespace, dict]:
    """MRST `[smry, smspec] = readEclipseSummaryUnFmt(prefix, keyWords)`."""
    prefix = _normalise_prefix(prefix)
    smspec = read_eclipse_output_file_un_fmt(prefix + ".SMSPEC")
    if "WGNAMES" in smspec:
        name_list = list(smspec["WGNAMES"].values)
    elif "NAMES" in smspec:
        name_list = list(smspec["NAMES"].values)
    else:
        raise KeyError("No known summary vector name list in SMSPEC fields\n\t{ "
                       + " ".join(sorted(smspec)) + " }")
    kwrd_list = list(smspec["KEYWORDS"].values)
    name_list = [n if n != "" else "empty" for n in name_list]
    kwrd_list = [k if k != "" else "empty" for k in kwrd_list]
    num_full = len(name_list)
    if keyWords is not None:
        wanted = set([keyWords] if isinstance(keyWords, str) else keyWords)
        row_inx = np.flatnonzero([k in wanted for k in kwrd_list])
    else:
        row_inx = np.arange(num_full)
    name_list = [name_list[i] for i in row_inx]
    kwrd_list = [kwrd_list[i] for i in row_inx]
    names, n_inx_order, n_inx = _unique(name_list)
    kwrds, k_inx_order, k_inx = _unique(kwrd_list)
    smry_files = _summary_files(prefix)
    columns: List[np.ndarray] = []
    ministeps: List[float] = []
    for fname in smry_files:
        for name, _, values in iter_eclipse_output_records(fname):
            if name == "MINISTEP":
                ministeps.append(float(np.asarray(values).ravel()[0]))
            elif name == "PARAMS":
                field = np.asarray(values, dtype=float).ravel()
                while len(columns) < len(ministeps):
                    columns.append(np.zeros(row_inx.size))
                columns[len(ministeps) - 1] = field[row_inx]
    while len(columns) < len(ministeps):
        columns.append(np.zeros(row_inx.size))
    data = np.column_stack(columns) if columns else np.zeros((row_inx.size, 0))
    smry = SimpleNamespace(
        WGNAMES=names, KEYWORDS=kwrds, UNITS=list(smspec["UNITS"].values),
        STARTDAT=np.asarray(smspec["STARTDAT"].values)[[2, 1, 0]],
        NUMS=np.asarray(smspec["NUMS"].values), nInx=n_inx, nInxOrder=n_inx_order,
        kInx=k_inx, kInxOrder=k_inx_order, data=data, ministep=np.asarray(ministeps),
        cartDims=np.asarray(smspec["DIMENS"].values)[1:4])
    return _add_smry_funcs(smry), smspec


def _summary_files(prefix: str) -> List[str]:
    smry_file = prefix + ".UNSMRY"
    if Path(smry_file).is_file():
        return [smry_file]
    dname, fp = str(Path(prefix).parent), Path(prefix).name
    return match_result_files(dname or ".", re.escape(fp) + r"\.S\d{4}")


def match_result_files(dname: str, patt: str) -> List[str]:
    """MRST private `matchResultFiles`: the sorted files in `dname` whose
    names contain a match of `patt`, as paths."""
    found = []
    for entry in Path(dname).iterdir():
        hit = re.search(patt, entry.name)
        if hit:
            found.append(hit.group(0))
    return [str(Path(dname)/f) for f in sorted(found)]


def _unique(values: Sequence[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """MATLAB `[u, iFirst, iInv] = unique(cellstr)`, 0-based."""
    u, first, inv = np.unique(np.asarray(values, dtype=object).astype(str),
                              return_index=True, return_inverse=True)
    return u.tolist(), first.astype(np.int64), inv.astype(np.int64).ravel()


def _add_smry_funcs(smry: SimpleNamespace) -> SimpleNamespace:
    smry.get = lambda *args, **kw: _get_data(smry, *args, **kw)
    smry.getInx = lambda nm, kw: _get_row_inx(smry, nm, kw)[0]
    smry.getNms = lambda kw: _get_names(smry, kw)
    smry.getKws = lambda nm: _get_keywords(smry, nm)
    smry.getUnit = lambda nm, kw: _get_unit(smry, nm, kw)
    return smry


def _get_names(smry: SimpleNamespace, kw: Any) -> List[str]:
    r_inx, _ = _get_row_inx(smry, None, kw)
    return [smry.WGNAMES[i] for i in np.unique(smry.nInx[r_inx])]


def _get_keywords(smry: SimpleNamespace, nm: Any) -> List[str]:
    r_inx, _ = _get_row_inx(smry, nm, None)
    return [smry.KEYWORDS[i] for i in np.unique(smry.kInx[r_inx])]


def _get_data(smry: SimpleNamespace, nm: Any, kw: Any, ms: Any = None,
              trimData: bool = False, nargout: int = 1):
    """MRST `[s, unit, spec] = smry.get(nm, kw, ms, trimData)`."""
    r_inx, order = _get_row_inx(smry, nm, kw)
    if _is_empty(ms) or (isinstance(ms, str) and ms == ":"):
        ms = np.arange(smry.data.shape[1])
    s = smry.data[r_inx][:, np.asarray(ms, dtype=np.int64)]
    s = s[order, :]
    unit = _get_unit(smry, nm, kw) if nargout >= 2 else None
    spec = None
    if nargout >= 3 or trimData:
        spec, is_valid = _get_spec(smry, nm, kw)
    if trimData:
        s, spec = s[is_valid, :], spec[is_valid]
    return (s, unit, spec)[:nargout] if nargout > 1 else s


def _get_row_inx(smry: SimpleNamespace, nm: Any, kw: Any) -> Tuple[np.ndarray, np.ndarray]:
    nm_multi = isinstance(nm, (list, tuple)) and len(nm) > 1
    kw_multi = isinstance(kw, (list, tuple)) and len(kw) > 1
    assert not (nm_multi and kw_multi), (
        "At most one of Keyword or Entity may be a multi-element string array "
        "or a cell array of character vectors")
    nlist = smry.kInx.size
    nm_order = kw_order = None
    if not _is_empty(nm):
        pick, nm_order = _string_lookup(smry.WGNAMES, smry.nInxOrder, _as_list(nm))
        r_inx_n = pick[smry.nInx]
    else:
        r_inx_n = np.ones(nlist, dtype=bool)
    if not _is_empty(kw):
        pick, kw_order = _string_lookup(smry.KEYWORDS, smry.kInxOrder, _as_list(kw))
        r_inx_k = pick[smry.kInx]
    else:
        r_inx_k = np.ones(nlist, dtype=bool)
    r_inx = r_inx_n & r_inx_k
    if nm_multi:
        order = nm_order
    elif kw_multi:
        order = kw_order
    else:
        order = np.arange(int(r_inx.sum()))
    return r_inx, order


def _get_unit(smry: SimpleNamespace, nm: Any, kw: Any) -> str:
    r_inx, _ = _get_row_inx(smry, nm, kw)
    return str(_masked(np.asarray(smry.UNITS, dtype=object), r_inx)[0]).strip()


def _get_spec(smry: SimpleNamespace, nm: Any, kw: str) -> Tuple[np.ndarray, np.ndarray]:
    r_inx, _ = _get_row_inx(smry, nm, kw)
    nums = _masked(np.asarray(smry.NUMS), r_inx)
    spec = nums
    if kw == "STEPTYPE":
        return spec, np.ones(nums.shape, dtype=bool)
    is_valid = nums > 0
    if kw[0] in ("B", "C"):
        # `[i, j, k] = ind2sub(cartDims, nums)`, 1-based like `nums`.
        i, j, k = np.unravel_index(nums - 1, tuple(int(d) for d in smry.cartDims), order="F")
        spec = np.column_stack([i + 1, j + 1, k + 1])
    elif kw[0] == "R":
        d = 32768
        if np.all(nums > d):
            spec = np.column_stack([np.mod(nums, d), nums//d - 10])
    elif kw[0] not in ("A", "S"):
        is_valid = np.ones(nums.shape, dtype=bool)
    return spec, is_valid


def _string_lookup(lst: Sequence[str], list_order: np.ndarray,
                   search: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """MRST `string_lookup`: which of `lst` are searched for, and the order
    that puts the selected rows in search order."""
    ncol = len(search)
    # `[i, j] = blockDiagIndex(numel(list), ncol)`: one numel(list)-by-ncol
    # block, listed column by column.
    i, j = block_diag_index(len(lst), ncol)
    lst_arr, search_arr = np.asarray(lst, dtype=object), np.asarray(search, dtype=object)
    match = (lst_arr[i] == search_arr[j]).reshape((len(lst), ncol), order="F")
    pick = match.any(axis=1)
    row = np.nonzero(match.T)[1]                   # `find` goes column by column
    reverse = np.argsort(np.asarray(list_order)[row], kind="stable")
    result_order = np.zeros(reverse.size, dtype=np.int64)
    result_order[reverse] = np.arange(reverse.size)
    return pick, result_order


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """MATLAB `values(mask)` with a logical `mask` that may be shorter than
    `values`. `UNITS` and `NUMS` are never filtered by `keyWords`, so with a
    filter MRST indexes their leading entries; this does the same."""
    return values[:mask.size][mask]


def _as_list(x: Any) -> List[str]:
    return [x] if isinstance(x, str) else list(x)


def _is_empty(x: Any) -> bool:
    if isinstance(x, str):
        return x == ""
    return x is None or np.size(x) == 0


def _normalise_prefix(prefix: Any) -> str:
    if isinstance(prefix, (list, tuple)):
        if len(prefix) > 1:
            import warnings
            warnings.warn(f"Case prefix must be single element only\n"
                          f"Using first element '{prefix[0]}'.")
        prefix = prefix[0]
    return str(prefix)


readEclipseSummaryUnFmt = read_eclipse_summary_un_fmt
matchResultFiles = match_result_files

__all__ = ["matchResultFiles", "match_result_files", "readEclipseSummaryUnFmt",
           "read_eclipse_summary_un_fmt"]


if __name__ == "__main__":
    import tempfile

    SPE9 = (Path(__file__).resolve().parents[5]/"mrst-2024a-Geomech"/"modules"
            /"ad-blackoil"/"examples"/"spe9"/"compare"/"SPE9")

    def test_spe9_summary_has_its_wells_and_steps():
        smry, _ = read_eclipse_summary_un_fmt(SPE9)
        assert "PROD13" in smry.WGNAMES and "WBHP" in smry.KEYWORDS
        assert smry.data.shape[1] == smry.ministep.size > 0
        wells = [n for n in smry.getNms("WBHP") if n.startswith(("INJE", "PROD"))]
        assert len(wells) == 26, wells
        print("ok test_spe9_summary_has_its_wells_and_steps")

    def test_get_returns_one_row_per_name_in_search_order():
        smry, _ = read_eclipse_summary_un_fmt(SPE9)
        one = smry.get("PROD18", "WBHP", ":")
        both = smry.get(["PROD18", "PROD13"], "WBHP", np.arange(smry.data.shape[1]))
        assert both.shape == (2, smry.data.shape[1])
        np.testing.assert_array_equal(both[0], one[0])
        np.testing.assert_array_equal(both[1], smry.get("PROD13", "WBHP")[0])
        print("ok test_get_returns_one_row_per_name_in_search_order")

    def test_keyword_filter_and_units():
        smry, _ = read_eclipse_summary_un_fmt(SPE9, ["WBHP", "YEARS"])
        assert set(smry.KEYWORDS) == {"WBHP", "YEARS"}
        years = smry.get(":+:+:+:+", "YEARS", ":")
        assert years.shape[0] == 1 and np.all(np.diff(years[0]) >= 0)
        full, _ = read_eclipse_summary_un_fmt(SPE9)
        _, unit = full.get("PROD13", "WBHP", None, nargout=2)
        assert unit == full.getUnit("PROD13", "WBHP") == "PSIA"
        print("ok test_keyword_filter_and_units")

    def test_cell_spec_is_the_one_based_ijk():
        smry = SimpleNamespace(NUMS=np.array([1, 2, 6]), cartDims=np.array([2, 3, 1]),
                               WGNAMES=["a"], KEYWORDS=["BPR"], nInx=np.zeros(3, int),
                               kInx=np.zeros(3, int), nInxOrder=np.array([0]),
                               kInxOrder=np.array([0]))
        spec, valid = _get_spec(smry, None, "BPR")
        np.testing.assert_array_equal(spec, [[1, 1, 1], [2, 1, 1], [2, 3, 1]])
        assert valid.all()
        print("ok test_cell_spec_is_the_one_based_ijk")

    def test_separate_summary_files_are_matched_in_order():
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("CASE.S0002", "CASE.S0001", "CASE.SMSPEC", "OTHER.S0001"):
                (Path(tmp)/name).touch()
            found = match_result_files(tmp, r"CASE\.S\d{4}")
            assert [Path(f).name for f in found] == ["CASE.S0001", "CASE.S0002"]
        print("ok test_separate_summary_files_are_matched_in_order")

    test_spe9_summary_has_its_wells_and_steps()
    test_get_returns_one_row_per_name_in_search_order()
    test_keyword_filter_and_units()
    test_cell_spec_is_the_one_based_ijk()
    test_separate_summary_files_are_matched_in_order()
    print("\nAll tests passed!")
