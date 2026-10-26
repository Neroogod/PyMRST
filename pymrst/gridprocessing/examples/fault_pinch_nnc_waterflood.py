"""End-to-end faulted corner-point waterflood with pinch and explicit NNCs."""

from dataclasses import dataclass

import numpy as np

from pymrst.gridprocessing import (
    compute_geometry, process_faults, process_grdecl, process_nnc, process_pinch,
)
from pymrst.gridprocessing.testgrids import simple_grdecl
from pymrst.modules.ad_blackoil import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_core import simulate_schedule_ad
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


@dataclass
class FaultPinchResult:
    grid: object
    faults: list
    pressure: np.ndarray
    saturation: np.ndarray
    bhp: np.ndarray
    qws: np.ndarray
    qos: np.ndarray
    iterations: np.ndarray


def _active(grid, cartesian):
    lookup = np.full(int(np.prod(grid.cartDims)), -1, dtype=np.int64)
    lookup[grid.cells.indexMap] = np.arange(grid.cells.num)
    out = lookup[np.asarray(cartesian, dtype=np.int64)]
    return out[out >= 0]


def build_fault_pinch_nnc_waterflood():
    dims = np.array([5, 4, 3])
    deck = simple_grdecl(dims, 0.03, flat=True)
    active = np.ones(tuple(dims), dtype=np.int32)
    active[2, 1, 1] = 0
    active[3, 2, 1] = 0
    deck["ACTNUM"] = active.ravel(order="F")
    deck["PINCH"] = [0.2, "GAP", 0.0, "TOPBOT", "TOP"]

    grid = compute_geometry(process_grdecl(deck))
    geometry = {
        "FAULTS": [
            ["F_X", 2, 2, 1, 4, 1, 3, "X+"],
            ["F_Y", 1, 5, 2, 2, 1, 3, "Y+"],
        ],
        "MULTFLT": [["F_X", 0.25], ["F_Y", 0.55]],
    }
    faults, _ = process_faults(grid, geometry)

    pinch = process_pinch(deck, grid)
    explicit = process_nnc(
        grid, np.array([[1, 1, 1, 5, 4, 3, 4.0e-15]]),
        exclude_existing=True,
    )
    cells, trans = [], []
    if pinch is not None:
        cells.append(pinch["cells"])
        trans.append(np.full(pinch["cells"].shape[0], 2.0e-15))
    if explicit is not None:
        cells.append(explicit["cells"])
        trans.append(explicit["trans"])
    grid.nnc = {"cells": np.vstack(cells), "trans": np.concatenate(trans)}

    cart = np.asarray(grid.cells.indexMap)
    layer = cart // (dims[0] * dims[1])
    perm_md = np.array([160.0, 45.0, 220.0])[layer]
    perm_md *= 1.0 + 0.2*np.sin(cart + 1.0)
    porosity = 0.16 + 0.05 * perm_md / perm_md.max()
    rock = make_rock(grid, (perm_md*MILLI*DARCY)[:, None], porosity[:, None])
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 4.0])*CENTI*POISE, rho=[1000.0, 780.0], n=[2, 2]
    )
    injector = _active(grid, [0, 20, 40])
    producer = _active(grid, [19, 39, 59])
    wells = add_well(None, grid, rock, injector, well_type="rate",
                     val=1.0e-4/DAY, compi=[1.0, 0.0], sign=1, name="INJ",
                     wi=np.full(injector.size, 1.0e-13))
    wells = add_well(wells, grid, rock, producer, well_type="bhp",
                     val=180*BARSA, compi=[0.0, 1.0], sign=-1, name="PROD",
                     wi=np.full(producer.size, 1.0e-13))
    model = TwoPhaseOilWaterModelAD(grid, rock, fluid, wells)

    internal_faces = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    connection = {int(face): ix for ix, face in enumerate(internal_faces)}
    for fault in faults:
        ix = [connection[int(face)] for face in fault.faces if int(face) in connection]
        model.ops.T[np.asarray(ix, dtype=np.int64)] *= fault.multiplier
    return deck, grid, faults, model


def run_fault_pinch_nnc_waterflood():
    _deck, grid, faults, model = build_fault_pinch_nnc_waterflood()
    state = model.initial_state(250*BARSA, 0.15)
    steps = np.full(6, 2*DAY)
    states, iterations = simulate_schedule_ad(state, model, steps, verbose=False)
    return FaultPinchResult(
        grid=grid, faults=faults,
        pressure=np.vstack([s.pressure for s in states]),
        saturation=np.vstack([s.sW for s in states]),
        bhp=np.vstack([s.bhp for s in states]),
        qws=np.vstack([s.qWs for s in states]),
        qos=np.vstack([s.qOs for s in states]),
        iterations=np.asarray(iterations),
    )


if __name__ == "__main__":
    result = run_fault_pinch_nnc_waterflood()
    print(f"{result.grid.cells.num} active cells, {result.grid.nnc['cells'].shape[0]} NNCs")
    print(f"faults: {[(f.name, f.numf, f.multiplier) for f in result.faults]}")
    print(f"final pressure: {result.pressure[-1].min()/BARSA:.2f}--{result.pressure[-1].max()/BARSA:.2f} bar")
    print(f"final Sw: {result.saturation[-1].min():.4f}--{result.saturation[-1].max():.4f}")
