"""Validate advanced PyMRST ``add_well`` behavior against MATLAB MRST."""

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import add_well, make_rock
from pymrst.utils.compute_well_index import cell_dims
from pymrst.utils.units import DARCY, MILLI


HERE = Path(__file__).parent
RTOL = 2e-12


def _case():
    grid = compute_geometry(cart_grid([2, 2, 2], [20.0, 20.0, 20.0]))
    rock = make_rock(
        grid,
        np.tile(np.array([100.0, 50.0, 10.0])*MILLI*DARCY,
                (grid.cells.num, 1)),
        0.2,
    )
    return grid, rock


def _array(data, name):
    return np.asarray(data[name]).ravel()


def _check(label, actual, expected, *, exact=False):
    actual = np.asarray(actual).ravel()
    expected = np.asarray(expected).ravel()
    ok = (
        np.array_equal(actual, expected)
        if exact else np.allclose(actual, expected, rtol=RTOL, atol=1e-30)
    )
    if not ok:
        difference = np.max(np.abs(actual.astype(float)-expected.astype(float)))
        raise AssertionError(f"{label}: maximum difference {difference:.3e}")
    print(f"  ok   {label}")


def validate_options():
    ref = scipy.io.loadmat(HERE/"ref_params_add_well_options.mat", squeeze_me=True)
    grid, rock = _case()
    mask = np.zeros(grid.cells.num, dtype=bool)
    mask[[0, 2, 4]] = True
    well = add_well(
        None, grid, rock, mask, well_type="rate", val=-2.0, sign=1,
        radius=[0.10, 0.12, 0.14], skin=[0.0, 1.0, 2.0],
        wi=[12.5, -1.0, 7.5], status=False, vfp_index=3,
        cp=np.array([4, 5, 6]), components=[0.2, 0.8],
        heel=np.array([1, 2, 3]), lims={"bhp": 1e7},
    )[0]
    _check("logical completion cells", well.cells, _array(ref, "cells")-1, exact=True)
    for field in ("WI", "r", "rR", "dZ"):
        _check(field, getattr(well, field), _array(ref, field))
    _check("rate-derived sign", [well.sign], [ref["signValue"]], exact=True)
    _check("well status", [well.status], [ref["statusValue"]], exact=True)
    _check("completion status", well.cstatus, _array(ref, "cstatus"), exact=True)
    _check("VFP index", [well.vfp_index], [ref["vfpIndex"]], exact=True)
    _check("completion mapping", well.cp, _array(ref, "cp"), exact=True)
    _check("components", well.components, _array(ref, "components"))
    _check("heel", well.heel, _array(ref, "heel"))
    _check("BHP limit", [well.lims["bhp"]], [ref["limitBhp"]])


def validate_segments():
    ref = scipy.io.loadmat(HERE/"ref_params_add_well_segments.mat", squeeze_me=True)
    grid, rock = _case()
    cells = _array(ref, "cells").astype(np.int64)-1
    dims = np.asarray(ref["dims"], dtype=float)
    segments = np.asarray(ref["segments"], dtype=float)
    well = add_well(
        None, grid, rock, cells, cell_dims_override=dims,
        line_segments=segments,
    )[0]
    _check("slanted trajectory WI", well.WI, _array(ref, "WI"))
    _check("trajectory representative radius", well.rR, _array(ref, "rR"))
    _check("cellDims fixture", cell_dims(grid, cells), dims)
    simple = add_well(
        None, grid, rock, cells, inner_product="ip_simple"
    )[0]
    quasitpf = add_well(
        None, grid, rock, cells, inner_product="ip_quasitpf"
    )[0]
    _check("ip_simple well constant", simple.WI, _array(ref, "WI_simple"))
    _check("ip_quasitpf well constant", quasitpf.WI, _array(ref, "WI_quasitpf"))


def validate_no_repr():
    ref = scipy.io.loadmat(HERE/"ref_params_add_well_no_repr.mat", squeeze_me=True)
    grid, rock = _case()
    well = add_well(None, grid, rock, [0], calc_repr_rad=False)[0]
    _check("WI without representative radius", well.WI, _array(ref, "WI"))
    if well.rR.size or np.asarray(ref["rR"]).size:
        raise AssertionError("calcReprRad=false must produce an empty rR")
    print("  ok   disabled representative radius")


def main():
    required = [
        HERE/"ref_params_add_well_options.mat",
        HERE/"ref_params_add_well_segments.mat",
        HERE/"ref_params_add_well_no_repr.mat",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing MATLAB references: " + ", ".join(missing)
            + ". Run dump_params_add_well_reference.m first."
        )
    validate_options()
    validate_segments()
    validate_no_repr()
    print("\nAdvanced addWell cases match MATLAB MRST.")


if __name__ == "__main__":
    main()
