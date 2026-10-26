"""MRST ``deckformat/params/wells_and_bc/processWells.m``.

`process_wells` builds the wells of one SCHEDULE control, as MRST does. It
numbers the wells by name, fills in defaulted COMPDAT, WCONINJE and WCONPROD
entries, and groups the completions by well. Wells are then added in stages:
injectors first (WCONINJ, WCONINJE, WCONINJH), then producers (WCONHIST,
WCONPROD), then the post-processing keywords (WINJGAS, WPOLYMER, WSOLVENT,
WSURFACT, WTEMP). The stage order is the order of the output, so it fixes
each well's index in `wellSols`. Cells are 0-based; limits are dicts.

`process_wells_ad` is PyMRST's earlier entry point, which delegates to the
compositional module's deck-well helpers (WELSPECS order, its own limits). It
is kept for its callers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from pymrst.params.wells_and_bc.add_well import add_well
from pymrst.utils.rldecode import rldecode
from pymrst.utils.units import ATM, BARSA, FT

DEFAULT_INJECTOR_BHP: float = 6895*BARSA        # about 1e5 psia
DEFAULT_PRODUCER_BHP: float = 1*ATM


def process_wells(G: Any, rock: Any, control: Mapping[str, Any], *,
                  InnerProduct: str = "ip_tpf", Verbose: bool = False,
                  DepthReorder: Any = False, strictParsing: bool = True,
                  createDefaultWell: bool = False, outputDefaulted: bool = False,
                  cellDims: Any = None) -> List[Any]:
    """MRST `processWells`: the wells of one SCHEDULE control record."""
    opt = dict(InnerProduct=InnerProduct, Verbose=Verbose, DepthReorder=DepthReorder,
               strictParsing=strictParsing, createDefaultWell=createDefaultWell,
               outputDefaulted=outputDefaulted, cellDims=cellDims)
    if createDefaultWell:
        return _create_default_well(G, rock)
    well_id = _enumerate_wells(control)
    control = _insert_defaults(control, well_id, int(G.cartDims[2]))
    control, p = _order_completions(control)
    inj = {"WCONINJ": _process_wconinj, "WCONINJE": _process_wconinje,
           "WCONINJH": _process_wconinjh}
    prod = {"WCONHIST": _process_wconhist, "WCONPROD": _process_wconprod}
    post = {"WPOLYMER": _process_wpolymer, "WSURFACT": _process_wsurfact,
            "WSOLVENT": _process_wsolvent, "WTEMP": _process_wtemp,
            "WINJGAS": _process_winjgas}
    W: List[Any] = []
    c2a = _make_cart_to_active(G)
    for stage in (inj, prod, post):
        for kw in sorted(stage):
            if kw in control:
                W = stage[kw](W, control, G, rock, c2a, well_id, p, opt)
    if not W:
        return _create_default_well(G, rock) if createDefaultWell else []
    from pymrst.modules.ad_core.models.facilities.reorder_well_perforations_by_depth import (
        reorder_well_perforations_by_depth,
    )
    if DepthReorder is None:
        W = list(reorder_well_perforations_by_depth(
            W, [str(w.dir)[0].lower() == "z" for w in W]))
    elif DepthReorder:
        W = list(reorder_well_perforations_by_depth(W))
    for w in W:
        w.cstatus = np.logical_and(w.status, np.asarray(w.cstatus, dtype=bool))
        w.status = bool(w.status and np.any(w.cstatus))
    return W


# -- helpers ------------------------------------------------------------------

def _make_cart_to_active(G: Any) -> np.ndarray:
    """Cartesian index to active cell, -1 for an inactive cell."""
    c2a = np.full(int(np.prod(G.cartDims)), -1, dtype=np.int64)
    c2a[np.asarray(G.cells.indexMap, dtype=np.int64).ravel()] = np.arange(G.cells.num)
    return c2a


def _rows(control: Mapping[str, Any], kw: str) -> List[List[Any]]:
    rows = control.get(kw)
    return [] if rows is None else [list(r) for r in rows]


def _enumerate_wells(control: Mapping[str, Any]) -> Callable[[str], int]:
    """Well name to its 0-based row in WELSPECS."""
    names = [row[0] for row in _rows(control, "WELSPECS")]
    assert len(set(names)) == len(names)
    index = {name: i for i, name in enumerate(names)}
    return lambda name: index[name]


def _insert_defaults(control: Mapping[str, Any], well_id, nlayers: int) -> Dict[str, Any]:
    control = dict(control)
    control["COMPDAT"] = _insert_default_compdat(control, well_id, nlayers)
    control["WCONINJE"] = _insert_default_wconinje(control)
    control["WCONPROD"] = _insert_default_wconprod(control)
    control["WELLSTRE"] = control.get("WELLSTRE", [])
    return control


def _insert_default_compdat(control, well_id, nlayers: int) -> List[List[Any]]:
    """COMPDAT with the well name replaced by its id; wells without COMPDAT
    get a full column, and defaulted I/J come from WELSPECS."""
    welspecs = _rows(control, "WELSPECS")
    compdat = _rows(control, "COMPDAT")
    present = np.zeros(len(welspecs), dtype=bool)
    for row in compdat:
        row[0] = well_id(row[0])
        present[row[0]] = True
    template = [-1, -1, 1, nlayers, "OPEN", 0, 0.0, 0.1, -1, 0.0, "Default", "Z", -1]
    for w in np.flatnonzero(~present):
        compdat.append([int(w)] + list(template))
    for row in compdat:
        if row[1] < 1:
            row[1] = welspecs[row[0]][2]
        if row[2] < 1:
            row[2] = welspecs[row[0]][3]
    return compdat


def _insert_default_wconinje(control) -> List[List[Any]]:
    rows = _rows(control, "WCONINJE")
    for row in rows:
        if _isnan(row[6]):
            row[6] = DEFAULT_INJECTOR_BHP
    return rows


def _insert_default_wconprod(control) -> List[List[Any]]:
    rows = _rows(control, "WCONPROD")
    for row in rows:
        if _isnan(row[8]):
            row[8] = DEFAULT_PRODUCER_BHP
    return rows


def _isnan(x: Any) -> bool:
    return isinstance(x, (float, np.floating)) and bool(np.isnan(x))


def _order_completions(control):
    """Completions grouped by well id, in deck order within a well; `p[i]`
    to `p[i+1]` are the rows of well `i`."""
    compdat = control["COMPDAT"]
    if not compdat:
        return control, None
    order = sorted(range(len(compdat)), key=lambda k: (compdat[k][0], k))
    ids = np.array([compdat[k][0] for k in order])
    wells, counts = np.unique(ids, return_counts=True)
    assert np.all(np.diff(wells) == 1)
    p = np.concatenate([[0], np.cumsum(counts)])
    control = dict(control)
    control["COMPDAT"] = [compdat[k] for k in order]
    return control, p


def _active_perf(G, comp, c2a) -> np.ndarray:
    """Active cells of one COMPDAT row, in MATLAB's `ndgrid` order."""
    nx, ny = int(G.cartDims[0]), int(G.cartDims[1])
    k = np.arange(int(comp[3]), int(comp[4]) + 1)
    cart = (int(comp[1]) - 1) + (int(comp[2]) - 1)*nx + (k - 1)*nx*ny
    perf = c2a[cart]
    return perf[perf >= 0]


def _build_well(W, G, rock, c2a, control, i, p, well_type, val, compi, ip, sgn, opt):
    comp = control["COMPDAT"][p[i]:p[i + 1]]
    welspecs = control["WELSPECS"][i]
    name = welspecs[0]
    perf = [_active_perf(G, row, c2a) for row in comp]
    nperf = np.array([x.size for x in perf], dtype=np.int64)
    column = lambda c: np.array([row[c] for row in comp])
    wdir = rldecode(column(12), nperf)
    kh = rldecode(column(9).astype(float), nperf)
    wi = rldecode(column(7).astype(float), nperf)
    wdiam = rldecode(column(8).astype(float), nperf)
    skin = rldecode(column(10).astype(float), nperf)
    open_shut = rldecode(column(5), nperf)
    defaulted_kh = kh <= 0
    defaulted_wi = wi <= 0
    kh[defaulted_kh] = -1
    wi[defaulted_wi] = -1
    wdiam[~(wdiam > 0)] = 1*FT
    assert np.all(wdiam[defaulted_kh & defaulted_wi] > 0), (
        "Well bore diameter must be defined unless productivity index is "
        "explicitly provided.")
    ref_depth = welspecs[4]
    assert isinstance(ref_depth, (int, float, np.number))
    perf = np.concatenate(perf) if perf else np.zeros(0, dtype=np.int64)
    defaulted_depth = bool(np.isnan(ref_depth))
    if defaulted_depth:
        ref_depth = float(np.min(np.asarray(G.cells.centroids)[perf, 2]))
    last = {cell: k for k, cell in enumerate(perf)}
    ia = np.sort(np.fromiter(last.values(), dtype=np.int64))
    cstatus = ~np.isin(open_shut[ia], ["SHUT", "HEAT"])
    if nperf.sum() > 0:
        size_w = len(W)
        W = add_well(W, G, rock, perf[ia], well_type=well_type, val=val,
                     well_dir="".join(str(d) for d in wdir[ia]).lower(),
                     kh=kh[ia], sign=sgn, wi=wi[ia], compi=compi, skin=skin[ia],
                     name=name, radius=wdiam[ia]/2, ref_depth=ref_depth,
                     inner_product=ip, cell_dims_override=opt["cellDims"],
                     heel=[welspecs[2], welspecs[3]])
        if len(W) > size_w:
            W[-1].lims = None
            W[-1].cstatus = cstatus
            # MRST keeps the deck's letters ('Z'); `makeScheduleConsistent`
            # orders a well by depth only when every letter is exactly 'Z'.
            W[-1].dir = "".join(str(d) for d in wdir[ia])
    if opt["outputDefaulted"] and W:
        W[-1].defaulted = {"Kh": defaulted_kh, "WI": defaulted_wi,
                           "refDepth": defaulted_depth}
    return W


def _lims(w) -> Dict[str, float]:
    if not isinstance(w.lims, dict):
        w.lims = {}
    return w.lims


def _injection_compi(phase: str):
    return {"w": [1, 0, 0], "o": [0, 1, 0], "g": [0, 0, 1]}.get(phase[0].lower())


# -- injectors ----------------------------------------------------------------

def _process_wconinj(W, control, G, rock, c2a, well_id, p, opt):
    for row in _rows(control, "WCONINJ"):
        nm, status = row[0], row[2] == "OPEN"
        well_type = row[3].lower()
        columns = {"rate": 4, "resv": 5, "bhp": 8, "thp": 9}
        if well_type not in columns:
            if opt["Verbose"] and status:
                print(f"Control mode '{well_type.upper()}' unsupported for "
                      f"injector '{nm}'.  Ignored.")
            continue
        val = row[columns[well_type]]
        compi = _injection_compi(row[1])
        if compi is None:
            if opt["Verbose"]:
                print(f"Injection phase '{row[1]}' is unknown.  Well ignored.")
            continue
        size_w = len(W)
        W = _build_well(W, G, rock, c2a, control, well_id(nm), p, well_type, val,
                        compi, opt["InnerProduct"], 1, opt)
        if len(W) > size_w:
            lims = _lims(W[-1])
            lims["rate"], lims["bhp"], lims["thp"] = row[4], row[8], row[9]
            W[-1].status = status
        W[-1].vfp_index = row[10]
    return W


def _process_wconinje(W, control, G, rock, c2a, well_id, p, opt):
    for row in _rows(control, "WCONINJE"):
        nm, status = row[0], row[2] == "OPEN"
        well_type = row[3].lower()
        columns = {"rate": 4, "resv": 5, "bhp": 6, "thp": 7}
        if well_type in columns:
            val = row[columns[well_type]]
        else:
            if opt["Verbose"]:
                print(f"Control mode '{well_type.upper()}' unsupported for "
                      f"injector '{nm}'.  Ignored.")
            val = 0
        compi = _injection_compi(row[1])
        if compi is None:
            if opt["Verbose"]:
                print(f"Injection phase '{row[1]}' is unknown.  Well ignored.")
            continue
        size_w = len(W)
        W = _build_well(W, G, rock, c2a, control, well_id(nm), p, well_type, val,
                        compi, opt["InnerProduct"], 1, opt)
        if len(W) > size_w:
            lims = _lims(W[-1])
            lims["rate"], lims["bhp"], lims["thp"] = row[4], row[6], row[7]
            W[-1].status = status
        W[-1].vfp_index = row[8]
    return W


def _process_wconinjh(W, control, G, rock, c2a, well_id, p, opt):
    for row in _rows(control, "WCONINJH"):
        nm, status = row[0], row[2] == "OPEN"
        val = row[3]                    # history-matched: always rate
        compi = _injection_compi(row[1])
        if compi is None:
            if opt["Verbose"]:
                print(f"Injection phase '{row[1]}' is unknown.  Well ignored.")
            continue
        size_w = len(W)
        W = _build_well(W, G, rock, c2a, control, well_id(nm), p, "rate", val,
                        compi, opt["InnerProduct"], 1, opt)
        if len(W) > size_w:
            lims = _lims(W[-1])
            lims["rate"], lims["bhp"] = row[3], DEFAULT_INJECTOR_BHP
            W[-1].status = status
    return W


# -- producers ----------------------------------------------------------------

def _process_wconhist(W, control, G, rock, c2a, well_id, p, opt):
    for row in _rows(control, "WCONHIST"):
        nm, status = row[0], row[1] == "OPEN"
        well_type = row[2].lower()
        if well_type == "orat":
            val, compi = -row[3], [0, 1, 0]
        elif well_type == "wrat":
            val, compi = -row[4], [1, 0, 0]
        elif well_type == "grat":
            val, compi = -row[5], [0, 0, 1]
        elif well_type == "lrat":
            rates = -np.array(row[3:5], dtype=float)
            val = rates.sum()
            compi = list(np.append(rates, 0)/val) if val != 0 else [0.5, 0.5, 0]
        elif well_type == "resv":
            rates = -np.array(row[3:6], dtype=float)
            val = rates.sum()
            compi = list(rates[[1, 0, 2]]/val) if val != 0 else [1/3, 1/3, 1/3]
            well_type = "RESV_HISTORY"
        elif well_type == "bhp":
            val, compi = row[9], [0, 1, 0]
        else:
            if opt["Verbose"] and status:
                print(f"Control mode '{well_type.upper()}' unsupported for "
                      f"producer '{nm}'.  Ignored.")
            val = 0
        size_w = len(W)
        W = _build_well(W, G, rock, c2a, control, well_id(nm), p, well_type, val,
                        compi, opt["InnerProduct"], -1, opt)
        if len(W) > size_w:
            lims = {"orat": -np.inf, "wrat": -np.inf, "grat": -np.inf,
                    "lrat": -np.inf, "resv": -np.inf, "bhp": 1*ATM}
            if well_type == "orat":
                lims["orat"] = -row[3]
            elif well_type == "wrat":
                lims["wrat"] = -row[4]
            elif well_type == "grat":
                lims["grat"] = -row[5]
            elif well_type == "lrat":
                lims["lrat"] = -(row[3] + row[4])
            elif well_type == "bhp":
                lims["bhp"] = row[9]
            W[-1].lims = lims
            W[-1].status = status
    return W


def _process_wconprod(W, control, G, rock, c2a, well_id, p, opt):
    for row in _rows(control, "WCONPROD"):
        nm, status = row[0], row[1] == "OPEN"
        well_type = row[2].lower()
        controls = {"orat": (-row[3], [0, 1, 0]), "wrat": (-row[4], [1, 0, 0]),
                    "grat": (-row[5], [0, 0, 1]), "lrat": (-row[6], [1, 1, 0]),
                    "resv": (-row[7], [1, 1, 1]), "bhp": (row[8], [1, 1, 1]),
                    "thp": (row[9], [1, 1, 1])}
        if well_type in controls:
            val, compi = controls[well_type]
        else:
            if opt["Verbose"] and status:
                print(f"Control mode '{well_type.upper()}' unsupported for "
                      f"producer '{nm}'.  Ignored.")
            val, compi = 0, [0, 1, 0]
        size_w = len(W)
        W = _build_well(W, G, rock, c2a, control, well_id(nm), p, well_type, val,
                        compi, opt["InnerProduct"], -1, opt)
        if len(W) > size_w:
            lims = _lims(W[-1])
            lims["orat"], lims["wrat"] = -row[3], -row[4]
            lims["grat"], lims["lrat"] = -row[5], -row[6]
            lims["bhp"], lims["thp"] = row[8], row[9]
            W[-1].status = status
        W[-1].vfp_index = row[10]
    return W


# -- post-processing ----------------------------------------------------------

def _append_named_value(W, control, kw: str, field: str):
    """`W(:).field(end+1) = 0`, then the deck's value for the named wells."""
    rows = _rows(control, kw)
    if not rows:
        return W
    for w in W:
        current = getattr(w, field, None)
        current = [] if current is None else list(np.atleast_1d(current))
        setattr(w, field, np.array(current + [0.0]))
    for row in rows:
        for w in W:
            if w.name == row[0]:
                getattr(w, field)[-1] = row[1]
    return W


def _process_wpolymer(W, control, *args):
    return _append_named_value(W, control, "WPOLYMER", "cp")


def _process_wsurfact(W, control, *args):
    return _append_named_value(W, control, "WSURFACT", "cs")


def _process_wtemp(W, control, *args):
    return _append_named_value(W, control, "WTEMP", "T")


def _process_wsolvent(W, control, *args):
    rows = _rows(control, "WSOLVENT")
    if not rows:
        return W
    for w in W:
        w.solventFrac = 0
    for row in rows:
        for w in W:
            if w.name == row[0]:
                w.solventFrac = row[1]
    return W


def _process_winjgas(W, control, *args):
    rows = _rows(control, "WINJGAS")
    if not rows or not W:
        return W
    streams = _rows(control, "WELLSTRE")
    for row in rows:
        for w in W:
            if w.name == row[0]:
                assert str(row[1]).lower() == "stream"
                match = [s for s in streams if str(s[0]).lower() == str(row[2]).lower()]
                w.components = np.array(match[0][1:], dtype=float)
    return W


def _create_default_well(G, rock) -> List[Any]:
    from types import SimpleNamespace
    unit_rock = SimpleNamespace(perm=np.ones(G.cells.num),
                                poro=getattr(rock, "poro", None))
    W = add_well([], G, unit_rock, [0], val=0, well_type="rate", sign=1,
                 compi=np.ones(3)/3, ref_depth=float(np.asarray(G.cells.centroids)[0, 2]))
    W[0].lims = {"rate": -np.inf, "bhp": np.inf}
    return W


def process_wells_ad(g, rock, control: Mapping[str, Any], *,
                     compi: Sequence[float] | None = None,
                     gravity: Sequence[float] | None = None,
                     **kwargs) -> list:
    """PyMRST's earlier deck-well builder (legacy): the compositional
    module's `wells_from_deck_completion_records` then
    `apply_deck_well_controls`, in WELSPECS order."""
    from pymrst.modules.compositional.facility import (
        apply_deck_well_controls,
        wells_from_deck_completion_records,
    )

    compi = np.array([1.0, 0.0, 0.0]) if compi is None else np.asarray(compi, float)
    gravity = np.array([0.0, 0.0, 9.80665]) if gravity is None else np.asarray(gravity, float)
    wells = wells_from_deck_completion_records(
        g, rock, control, compi=compi, gravity=gravity, **kwargs)
    return apply_deck_well_controls(wells, control, grid=g)


processWells = process_wells

__all__ = ["processWells", "process_wells", "process_wells_ad"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def _case():
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.2)
        nan = float("nan")
        control = {
            "WELSPECS": [["PROD", "G1", 5, 5, nan, "OIL"],
                         ["INJ", "G1", 1, 1, nan, "WATER"]],
            "COMPDAT": [["PROD", 5, 5, 1, 3, "OPEN", 0, -1.0, 0.2, -1, 0.0, "Default", "Z", -1],
                        ["INJ", 1, 1, 1, 3, "OPEN", 0, -1.0, 0.2, -1, 0.0, "Default", "Z", -1]],
            "WCONPROD": [["PROD", "OPEN", "BHP", 0, 0, 0, 0, 0, 200*BARSA, 0, 0]],
            "WCONINJE": [["INJ", "WATER", "OPEN", "RATE", 100.0, nan, nan, nan, 0]],
        }
        return g, rock, control

    def test_injectors_come_before_producers():
        """MRST adds wells by stage, injectors first, whatever WELSPECS says;
        the order is each well's index in `wellSols`."""
        g, rock, control = _case()
        assert [w.name for w in process_wells(g, rock, control)] == ["INJ", "PROD"]
        print("ok test_injectors_come_before_producers")

    def test_completions_map_to_the_right_cells():
        """Deck indices are one-based; a producer at (5, 5) over layers 1-3 of
        a 5x5x3 grid occupies cells 24, 49 and 74."""
        g, rock, control = _case()
        by_name = {w.name: np.asarray(w.cells).tolist() for w in process_wells(g, rock, control)}
        assert by_name["PROD"] == [24, 49, 74], by_name["PROD"]
        assert by_name["INJ"] == [0, 25, 50], by_name["INJ"]
        print("ok test_completions_map_to_the_right_cells")

    def test_controls_signs_and_limits_follow_the_keywords():
        g, rock, control = _case()
        wells = {w.name: w for w in process_wells(g, rock, control)}
        prod, inj = wells["PROD"], wells["INJ"]
        assert prod.type == "bhp" and np.isclose(prod.val, 200*BARSA) and prod.sign == -1
        assert inj.type == "rate" and np.isclose(inj.val, 100.0) and inj.sign == 1
        assert set(inj.lims) == {"rate", "bhp", "thp"}
        assert inj.lims["bhp"] == DEFAULT_INJECTOR_BHP, "defaulted BHP limit"
        assert set(prod.lims) == {"orat", "wrat", "grat", "lrat", "bhp", "thp"}
        print("ok test_controls_signs_and_limits_follow_the_keywords")

    def test_a_defaulted_reference_depth_is_the_shallowest_perforation():
        g, rock, control = _case()
        for w in process_wells(g, rock, control, outputDefaulted=True):
            top = np.min(np.asarray(g.cells.centroids)[w.cells, 2])
            assert np.isclose(w.refDepth, top) and w.defaulted["refDepth"]
        print("ok test_a_defaulted_reference_depth_is_the_shallowest_perforation")

    def test_a_well_without_compdat_gets_a_full_column():
        g, rock, control = _case()
        control = dict(control, COMPDAT=control["COMPDAT"][:1])
        inj = [w for w in process_wells(g, rock, control) if w.name == "INJ"][0]
        assert np.asarray(inj.cells).tolist() == [0, 25, 50]
        print("ok test_a_well_without_compdat_gets_a_full_column")

    def test_a_shut_well_closes_its_completions():
        g, rock, control = _case()
        control = dict(control)
        control["WCONPROD"] = [["PROD", "SHUT"] + control["WCONPROD"][0][2:]]
        prod = [w for w in process_wells(g, rock, control) if w.name == "PROD"][0]
        assert not prod.status and not np.any(prod.cstatus)
        print("ok test_a_shut_well_closes_its_completions")

    test_injectors_come_before_producers()
    test_completions_map_to_the_right_cells()
    test_controls_signs_and_limits_follow_the_keywords()
    test_a_defaulted_reference_depth_is_the_shallowest_perforation()
    test_a_well_without_compdat_gets_a_full_column()
    test_a_shut_well_closes_its_completions()
    print("\nAll tests passed!")
