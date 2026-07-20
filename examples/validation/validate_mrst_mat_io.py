"""Validate MRST MATLAB .mat import bridge."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.io import (
    load_mrst_grid_mat,
    load_mrst_mat,
    load_mrst_schedule_mat,
    load_mrst_states_mat,
    load_mrst_well_sols_mat,
    load_mrst_wells_mat,
)


def _mat_grid(g):
    return {
        "cells": {
            "num": g.cells.num,
            "facePos": g.cells.facePos + 1,
            "faces": np.column_stack([g.cells.faces + 1, np.ones(g.cells.faces.size)]),
            "indexMap": g.cells.indexMap + 1,
            "volumes": g.cells.volumes,
            "centroids": g.cells.centroids,
        },
        "faces": {
            "num": g.faces.num,
            "nodePos": g.faces.nodePos + 1,
            "nodes": g.faces.nodes + 1,
            "neighbors": np.where(g.faces.neighbors >= 0, g.faces.neighbors + 1, 0),
            "areas": g.faces.areas,
            "normals": g.faces.normals,
            "centroids": g.faces.centroids,
        },
        "nodes": {
            "num": g.nodes.num,
            "coords": g.nodes.coords,
        },
        "griddim": g.griddim,
        "cartDims": np.asarray(g.cartDims),
        "type": np.array(["tensorGrid", "computeGeometry"], dtype=object),
    }


def main() -> int:
    g = compute_geometry(cart_grid([2, 1, 1], [20.0, 10.0, 5.0]))
    well = {
        "cells": np.array([1, 2]),
        "type": "rate",
        "val": 123.0,
        "r": np.array([0.1, 0.1]),
        "dir": "z",
        "WI": np.array([10.0, 11.0]),
        "dZ": np.array([0.0, 1.0]),
        "name": "I1",
        "compi": np.array([1.0, 0.0, 0.0]),
        "refDepth": 1000.0,
        "sign": 1,
        "status": 1,
        "lims": {"bhp": 300.0},
    }
    state1 = {
        "pressure": np.array([100.0, 110.0]),
        "s": np.array([[0.2, 0.8], [0.25, 0.75]]),
        "wellSol": {"name": "I1", "qWs": 1.0, "bhp": 300.0},
    }
    state2 = {
        "pressure": np.array([120.0, 130.0]),
        "s": np.array([[0.3, 0.7], [0.35, 0.65]]),
        "wellSol": {"name": "I1", "qWs": 1.5, "bhp": 310.0},
    }
    schedule = {
        "step": {"val": np.array([1.0, 2.0]), "control": np.array([1, 1])},
        "control": {"W": well},
    }
    well_sols = np.array([
        {"name": "I1", "qWs": 1.0, "bhp": 300.0},
        {"name": "I1", "qWs": 1.5, "bhp": 310.0},
    ], dtype=object)

    path = Path(tempfile.gettempdir()) / "pymrst_mrst_mat_bridge_validation.mat"
    scipy.io.savemat(path, {
        "G": _mat_grid(g),
        "W": well,
        "states": np.array([state1, state2], dtype=object),
        "wellSols": well_sols,
        "schedule": schedule,
    })

    raw = load_mrst_mat(path)
    grid = load_mrst_grid_mat(path)
    states = load_mrst_states_mat(path)
    wells = load_mrst_wells_mat(path)
    sols = load_mrst_well_sols_mat(path)
    sched = load_mrst_schedule_mat(path)

    checks = [
        ("raw keys", "G" in raw and "states" in raw),
        ("grid cells", grid.cells.num == g.cells.num and np.allclose(grid.cells.volumes, g.cells.volumes)),
        ("state count", len(states) == 2 and np.allclose(states[1]["pressure"], [120.0, 130.0])),
        ("well zero-based cells", len(wells) == 1 and np.array_equal(wells[0].cells, [0, 1])),
        ("well solution count", len(sols) == 2 and sols[0][0]["name"] == "I1"),
        ("schedule W_py", "W_py" in sched["control"][0] and sched["control"][0]["W_py"][0].name == "I1"),
    ]
    failed = 0
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
