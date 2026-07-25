"""Well time-series extraction from a run's wellSols (MRST `getWellOutput`).

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


def get_well_output(
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
        data, names = get_well_output(_sols())
        assert names == ["I", "P"]
        assert data.shape == (3, 2, 2)          # steps, wells, {bhp, qWs}
        print("ok test_default_pulls_all_fields_all_wells")

    def test_single_field_single_well_by_name():
        data, names = get_well_output(_sols(), "bhp", "P")
        assert names == ["P"]
        assert data.shape == (3, 1, 1)
        assert np.allclose(data[:, 0, 0], [150, 150, 150])
        print("ok test_single_field_single_well_by_name")

    def test_field_time_series_matches_input():
        data, _ = get_well_output(_sols(), "qWs", [0])   # injector by index
        assert np.allclose(data[:, 0, 0], [2e-3, 2.1e-3, 2.2e-3])
        print("ok test_field_time_series_matches_input")

    def test_empty_run_returns_empty():
        data, names = get_well_output([])
        assert data.size == 0 and names == []
        print("ok test_empty_run_returns_empty")

    test_default_pulls_all_fields_all_wells()
    test_single_field_single_well_by_name()
    test_field_time_series_matches_input()
    test_empty_run_returns_empty()
    print("\nAll tests passed!")
