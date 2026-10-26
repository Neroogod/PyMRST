"""MRST ``ad-core/utils/getWellOutput.m``.

`get_well_output(wellsols, fldnames, wells, index)` is the port: it returns
`(welldata, wellnames, fldnames)`, `welldata` being `(nsteps, nwells,
nfields)` with entry `index` (1-based, as in MRST) of each field. Wells are
names (matched case-insensitively), a logical mask, or indices (0-based in
the port). A field must be one with at least `index` values (checked
case-insensitively, then read by its exact name, as MATLAB's `y.(fld)`
does); a field that is empty in the first well stays NaN. Well solutions may be dicts or MRST-style structs (namespaces).

`get_well_output_ad`, below, is PyMRST's earlier version, returning
`(welldata, wellnames)` for dict rows; it is kept for its callers. Its notes
follow.

Well time-series extraction from a run's wellSols.

A simulation returns a `wellSols` list - one entry per timestep, each a list of
per-well records (bhp, phase rates, ...). To plot or analyse a run you want the
transpose: a single quantity for chosen wells across all timesteps. This pulls
that out as an `(nsteps, nwells, nfields)` array, selecting fields by name and
wells by name or index, exactly as MRST does.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _well_names(ws: Sequence[Dict]) -> List[str]:
    return [str(w["name"]) for w in ws]


def _resolve_wells(wells, names: List[str]) -> List[int]:
    nw = len(names)
    if wells is None:
        return list(range(nw))
    if isinstance(wells, str):
        wells = [wells]
    if all(isinstance(w, str) for w in wells):
        lower = [n.lower() for n in names]
        return [lower.index(w.lower()) for w in wells]
    subs = [int(w) for w in wells]
    assert min(subs) >= 0 and max(subs) < nw, "well index out of range"
    return subs


def get_well_output_ad(
    wellsols: Sequence[Sequence[Dict]],
    fldnames: Optional[Union[str, Sequence[str]]] = None,
    wells=None,
    index: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """Return `(welldata, wellnames)` for the chosen fields and wells.

    `welldata` is `(nsteps, nwells, nfields)`. `fldnames` defaults to every
    field with at least `index` values per well; `wells` may be names or indices
    (default all). `index` is 1-based into a multi-valued field (default 1).
    """
    assert isinstance(wellsols, (list, tuple)), "wellsols must be a list"
    nsteps = len(wellsols)
    if nsteps == 0:
        return np.empty((0, 0, 0)), []

    ws0 = wellsols[0]
    names = _well_names(ws0)

    def _count(v) -> int:
        return int(np.asarray(v).size)

    valid = [k for k in ws0[0].keys()
             if k != "name" and _count(ws0[0][k]) >= index]
    if fldnames is None:
        fldnames = valid
    elif isinstance(fldnames, str):
        fldnames = [fldnames]

    subs = _resolve_wells(wells, names)
    wellnames = [names[i] for i in subs]

    welldata = np.full((nsteps, len(subs), len(fldnames)), np.nan)
    for fi, fld in enumerate(fldnames):
        if fld not in valid:
            raise ValueError(f"field {fld!r} has no value at index {index}")
        for si in range(nsteps):
            ws = wellsols[si]
            for wj, wi in enumerate(subs):
                welldata[si, wj, fi] = np.asarray(ws[wi][fld]).ravel()[index - 1]
    return welldata, wellnames



def get_well_output(wellsols: Sequence[Sequence[Any]], fldnames: Any = None,
                    wells: Any = None, index: int = 1
                    ) -> Tuple[np.ndarray, List[str], List[str]]:
    """MRST `[welldata, wellnames, fldnames] = getWellOutput(wellsols,
    fldnames, wells, index)`."""
    assert isinstance(wellsols, (list, tuple))
    nsteps = len(wellsols)
    if nsteps == 0:
        return np.zeros((0, 0)), [], []
    ws0 = list(wellsols[0])
    nw = len(ws0)
    wnames = [str(_field(w, "name")) for w in ws0]
    validfields = _field_names(ws0[0])
    datacounts = [_numel(_field(ws0[0], f)) for f in validfields]
    acceptible = [c >= index for c in datacounts]
    validfields = [f for f, ok in zip(validfields, acceptible) if ok]
    datacounts = [c for c, ok in zip(datacounts, acceptible) if ok]
    if fldnames is None:
        fldnames = list(validfields)
    if wells is None:
        wells = np.arange(nw)
    if isinstance(wells, str) or (isinstance(wells, (list, tuple)) and wells
                                  and all(isinstance(w, str) for w in wells)):
        wells = [wells] if isinstance(wells, str) else list(wells)
        lower = [n.lower() for n in wnames]
        subs = np.array([lower.index(w.lower()) for w in wells], dtype=np.int64)
    else:
        wells = np.asarray(wells)
        if wells.dtype == bool:
            assert wells.size == nw
            subs = np.flatnonzero(wells)
        else:
            subs = wells.astype(np.int64).ravel()
            assert subs.min() >= 0 and subs.max() < nw, (
                "Indices for wells outside of valid range")
    wellnames = [wnames[i] for i in subs]
    if isinstance(fldnames, str):
        fldnames = [fldnames]
    fldnames = list(fldnames)
    welldata = np.full((nsteps, subs.size, len(fldnames)), np.nan)
    lower_valid = [f.lower() for f in validfields]
    for i, fld in enumerate(fldnames):
        assert fld.lower() in lower_valid, (
            f"Dimensions of {fld} does not match the given index {index}")
        if datacounts[lower_valid.index(fld.lower())] == 0:
            continue
        for step, ws in enumerate(wellsols):
            ws = list(ws)
            welldata[step, :, i] = [_entry(_field(ws[k], fld), index) for k in subs]
    return welldata, wellnames, fldnames


def _field(row: Any, name: str) -> Any:
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _numel(x: Any) -> int:
    return len(x) if isinstance(x, str) else int(np.size(x))


def _entry(x: Any, index: int) -> float:
    """`x(index)`; a character becomes its code, as when MATLAB stores a
    char in a double array."""
    if isinstance(x, str):
        return float(ord(x[index - 1]))
    return float(np.asarray(x).ravel()[index - 1])


def _field_names(row: Any) -> List[str]:
    return list(row) if isinstance(row, dict) else list(vars(row))


getWellOutput = get_well_output

if __name__ == "__main__":
    def _sols():
        # two wells, three timesteps
        def ws(bhp_i, bhp_p, q_i, q_p):
            return [{"name": "I", "bhp": bhp_i, "qWs": q_i},
                    {"name": "P", "bhp": bhp_p, "qWs": q_p}]
        return [ws(210, 150, 2e-3, -1e-3),
                ws(212, 150, 2.1e-3, -1.1e-3),
                ws(214, 150, 2.2e-3, -1.2e-3)]

    def test_default_pulls_all_fields_all_wells():
        data, names = get_well_output_ad(_sols())
        assert names == ["I", "P"]
        assert data.shape == (3, 2, 2)          # steps, wells, {bhp, qWs}
        print("ok test_default_pulls_all_fields_all_wells")

    def test_single_field_single_well_by_name():
        data, names = get_well_output_ad(_sols(), "bhp", "P")
        assert names == ["P"]
        assert data.shape == (3, 1, 1)
        assert np.allclose(data[:, 0, 0], [150, 150, 150])
        print("ok test_single_field_single_well_by_name")

    def test_field_time_series_matches_input():
        data, _ = get_well_output_ad(_sols(), "qWs", [0])   # injector by index
        assert np.allclose(data[:, 0, 0], [2e-3, 2.1e-3, 2.2e-3])
        print("ok test_field_time_series_matches_input")

    def test_empty_run_returns_empty():
        data, names = get_well_output_ad([])
        assert data.size == 0 and names == []
        print("ok test_empty_run_returns_empty")

    def test_the_port_reads_struct_rows_and_returns_three_outputs():
        from types import SimpleNamespace
        sols = [[SimpleNamespace(name="I", bhp=210.0 + k, qWs=1e-3, flux=np.zeros(0)),
                 SimpleNamespace(name="P", bhp=150.0, qWs=-1e-3, flux=np.zeros(0))]
                for k in range(3)]
        data, names, flds = get_well_output(sols, "bhp", "i")
        assert names == ["I"] and flds == ["bhp"] and data.shape == (3, 1, 1)
        np.testing.assert_array_equal(data[:, 0, 0], [210.0, 211.0, 212.0])
        data, _, every = get_well_output(sols)
        assert every == ["name", "bhp", "qWs"], "empty fields are not valid at index 1"
        assert data[0, 0, 0] == ord("I"), "MATLAB stores the name's character code"
        data, names, _ = get_well_output(sols, "qWs", np.array([False, True]))
        assert names == ["P"] and np.all(data == -1e-3)
        print("ok test_the_port_reads_struct_rows_and_returns_three_outputs")

    test_the_port_reads_struct_rows_and_returns_three_outputs()
    test_default_pulls_all_fields_all_wells()
    test_single_field_single_well_by_name()
    test_field_time_series_matches_input()
    test_empty_run_returns_empty()
    print("\nAll tests passed!")
