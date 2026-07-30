"""Phase 434: compare combineMSwithRegularWells with MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.models.facilities import (
    combine_ms_with_regular_wells,
)

REF = Path(__file__).with_name("ref_p434_combine_wells.mat")


def _strings(array):
    return [str(item[0]) if isinstance(item, np.ndarray) else str(item)
            for item in array.ravel()]


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase434_reference.m first")
        return 1
    ref = loadmat(REF)
    regular = [
        SimpleNamespace(name="R1", cells=np.array([0]), WI=np.array([10.0])),
        SimpleNamespace(name="R2", cells=np.array([1]), WI=np.array([20.0])),
    ]
    ms = [SimpleNamespace(
        name="MS1", cells=np.array([2, 3]),
        nodes=SimpleNamespace(depth=np.array([0.0, 10.0])),
        topo=np.array([[0, 1]]),
    )]
    out = combine_ms_with_regular_wells(regular, ms)
    actual = {
        "names": [w.name for w in out],
        "isMS": [w.isMS for w in out],
        "WI empty": [w.WI is None for w in out],
        "nodes empty": [w.nodes is None for w in out],
        "topo empty": [w.topo is None for w in out],
        "cells": [w.cells + 1 for w in out],
    }
    expected = {
        "names": _strings(ref["names"]),
        "isMS": ref["is_ms"].astype(bool).ravel().tolist(),
        "WI empty": ref["wi_empty"].astype(bool).ravel().tolist(),
        "nodes empty": ref["nodes_empty"].astype(bool).ravel().tolist(),
        "topo empty": ref["topo_empty"].astype(bool).ravel().tolist(),
        "cells": [np.asarray(x).ravel() for x in ref["cells"].ravel()],
    }
    failed = 0
    for name in actual:
        if name == "cells":
            ok = all(np.array_equal(a, b)
                     for a, b in zip(actual[name], expected[name]))
        else:
            ok = actual[name] == expected[name]
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    print("\ncombineMSwithRegularWells matches MRST!"
          if not failed else f"\n{failed} checks FAILED")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
