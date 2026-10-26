"""MRST ``deckformat/resultinput/convertSummaryToWellSols.m``.

Well solutions from an ECLIPSE summary: per summary time, one record per
well (sorted by name, as MATLAB's `unique` sorts) with `bhp` (WBHP), the
surface rates `qOs`, `qWs`, `qGs` (production negative, injection
positive) and `sign`. Gas production comes from WGPR or else WGOR times the
oil rate, water from WWPR or else the water cut WWCT; WWIR and WGIR are
added for injection. Units are the summary's (from INTEHEAD), or `unit`
('metric', 'field', 'lab', 'pvt-m') when given; a summary passed directly
(not a file prefix) without a unit is read as metric, with MRST's warning.

`fn` is a file prefix or an already-read summary (`read_eclipse_summary_un_fmt`).
MRST prints which rates it could not reconstruct under `mrstVerbose` (off by
default); PyMRST has no such setting, so nothing is printed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

from pymrst.utils.units import ATM, BARSA, CENTI, DAY, FT, HOUR, METER, PSIA, STB

TIME_FIELD: str = ":+:+:+:+"


def convert_summary_to_well_sols(fn: Any, unit: Optional[Any] = None
                                 ) -> Tuple[List[List[SimpleNamespace]], np.ndarray]:
    """MRST `[wellSols, time] = convertSummaryToWellSols(fn, unit)`."""
    smry, spec = _get_summary(fn)
    qOs, qWs, qGs, bhp, wns, time = _extract_quantities(smry, _extract_units(spec, unit))
    return _assign_wellsols(smry, qOs, qWs, qGs, bhp, wns), time


convertSummaryToWellSols = convert_summary_to_well_sols


def _get_summary(fn: Any):
    if isinstance(fn, (str, bytes)) or hasattr(fn, "__fspath__"):
        from pymrst.modules.deckformat.resultinput.read_eclipse_summary_un_fmt import (
            read_eclipse_summary_un_fmt,
        )
        return read_eclipse_summary_un_fmt(fn)
    return fn, None


def _extract_units(spec: Any, unit: Any) -> SimpleNamespace:
    if isinstance(unit, str):
        return _get_units(unit)
    if spec:
        return _get_intehead_unit(spec)
    if unit is not None:
        return unit
    warnings.warn("No unit given, assuming metric")
    return _get_units("metric")


def _extract_quantities(smry: Any, u: SimpleNamespace):
    wns = _get_well_names(smry)
    t = np.atleast_2d(smry.get(TIME_FIELD, "TIME", ":"))
    time = (t*u.t).reshape(-1, order="F")
    nw = len(wns)
    nt = max(time.size, t.shape[1])
    qOs, qWs, qGs, bhp = (np.zeros((nt, nw)) for _ in range(4))
    for k, wn in enumerate(wns):
        akw = set(smry.getKws(wn))
        get = lambda kw: np.asarray(smry.get(wn, kw, ":"), dtype=float).reshape(-1, order="F")
        if "WBHP" in akw:
            bhp[:, k] = get("WBHP")*u.p
        if "WOPR" in akw:
            qOs[:, k] = -get("WOPR")*u.ql
        if "WGPR" in akw:
            qGs[:, k] = -get("WGPR")*u.qg
        elif "WGOR" in akw:
            qGs[:, k] = qOs[:, k]*get("WGOR")
        if "WWPR" in akw:
            qWs[:, k] = -get("WWPR")*u.ql
        elif "WWCT" in akw:
            wcut = get("WWCT")
            with np.errstate(divide="ignore", invalid="ignore"):
                qWs[:, k] = wcut*qOs[:, k]/(1 - wcut)
        if "WWIR" in akw:
            qWs[:, k] = qWs[:, k] + get("WWIR")*u.ql
        if "WGIR" in akw:
            qGs[:, k] = qGs[:, k] + get("WGIR")*u.qg
    return qOs, qWs, qGs, bhp, wns, time


def _get_well_names(smry: Any) -> List[str]:
    wkw = [kw for kw in smry.KEYWORDS if kw.startswith("W")]
    wns: List[str] = []
    for kw in wkw:
        wns.extend(smry.getNms(kw))
    return sorted(set(wns) - {TIME_FIELD})


def _assign_wellsols(smry: Any, qOs, qWs, qGs, bhp, wns) -> List[List[SimpleNamespace]]:
    nt = smry.data.shape[1]
    return [[SimpleNamespace(name=wn, bhp=bhp[kt, kw], qOs=qOs[kt, kw], qWs=qWs[kt, kw],
                             qGs=qGs[kt, kw],
                             sign=np.sign(qWs[kt, kw] + qOs[kt, kw] + qGs[kt, kw]))
             for kw, wn in enumerate(wns)]
            for kt in range(nt)]


def _get_units(unit: str) -> SimpleNamespace:
    key = unit.lower()
    if key == "metric":
        return SimpleNamespace(p=BARSA, ql=METER**3/DAY, qg=METER**3/DAY, t=DAY)
    if key == "field":
        return SimpleNamespace(p=PSIA, ql=STB/DAY, qg=1000*FT**3/DAY, t=DAY)
    if key == "lab":
        return SimpleNamespace(p=ATM, ql=(CENTI*METER)**3/HOUR,
                               qg=(CENTI*METER)**3/HOUR, t=HOUR)
    if key in ("pvt-m", "pvt_m"):
        return SimpleNamespace(p=ATM, ql=METER**3/DAY, qg=METER**3/DAY, t=DAY)
    raise ValueError(f"Unit {unit} not supported")


def _get_intehead_unit(spec: Any) -> SimpleNamespace:
    ustring = ("METRIC", "FIELD", "LAB", "PVT-M")
    return _get_units(ustring[int(np.ravel(np.asarray(spec["INTEHEAD"]))[0]) - 1])


__all__ = ["convert_summary_to_well_sols", "convertSummaryToWellSols"]


if __name__ == "__main__":
    from pathlib import Path

    DATA = Path(__file__).resolve().parents[5] / "mrst-2024a-Geomech" / "examples" / "data"

    def _fake_summary():
        keywords = ["TIME", "WBHP", "WOPR", "WWCT", "WGIR"]
        names = [TIME_FIELD, "P1", "P1", "P1", "I1"]
        data = np.array([[1.0, 2.0], [100.0, 90.0], [10.0, 20.0], [0.5, 0.5], [5.0, 5.0]])

        def rows(nm, kw):
            return [i for i, (n, k) in enumerate(zip(names, keywords))
                    if (nm is None or n == nm) and (kw is None or k == kw)]
        return SimpleNamespace(
            KEYWORDS=keywords, data=data,
            get=lambda nm, kw, ms: data[rows(nm, kw)],
            getKws=lambda nm: [keywords[i] for i in rows(nm, None)],
            getNms=lambda kw: [names[i] for i in rows(None, kw)])

    def test_rates_follow_mrsts_sign_and_unit_conventions():
        ws, time = convert_summary_to_well_sols(_fake_summary(), "metric")
        np.testing.assert_allclose(time, [DAY, 2*DAY])
        assert [w.name for w in ws[0]] == ["I1", "P1"]
        p1 = ws[1][1]
        np.testing.assert_allclose([p1.bhp, p1.qOs, p1.qWs], [90*BARSA, -20/DAY, -20/DAY])
        i1 = ws[0][0]
        np.testing.assert_allclose(i1.qGs, 5/DAY)
        assert i1.sign == 1 and p1.sign == -1
        print("ok test_rates_follow_mrsts_sign_and_unit_conventions")

    def test_a_summary_without_a_unit_is_read_as_metric():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ws, _ = convert_summary_to_well_sols(_fake_summary())
        assert any("assuming metric" in str(w.message) for w in caught)
        np.testing.assert_allclose(ws[0][1].bhp, 100*BARSA)
        print("ok test_a_summary_without_a_unit_is_read_as_metric")

    def test_norne_summary_reads_its_wells():
        prefix = (DATA / "norne_field_model" / "opm-tests" / "norne" / "opm-simulation-reference"
                  / "flow_legacy" / "NORNE_ATW2013")
        if not Path(str(prefix) + ".SMSPEC").exists():
            print("skip test_norne_summary_reads_its_wells")
            return
        ws, time = convert_summary_to_well_sols(prefix, "metric")
        assert len(ws) == time.size and len(ws[0]) > 30
        assert np.all(np.diff(time) > 0)
        print(f"ok test_norne_summary_reads_its_wells ({len(ws[0])} wells, {time.size} times)")

    test_rates_follow_mrsts_sign_and_unit_conventions()
    test_a_summary_without_a_unit_is_read_as_metric()
    test_norne_summary_reads_its_wells()
    print("\nAll tests passed!")
