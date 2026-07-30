import struct

import numpy as np

from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import (
    load_aquifer_eclipse_result,
)
from pymrst.modules.ad_core import State
from pymrst.utils.units import BARSA


def _record(payload):
    marker = struct.pack(">i", len(payload))
    return marker + payload + marker


def _field(name, kind, values):
    if kind == "CHAR":
        values = list(values)
        payload = b"".join(str(value).encode("ascii").ljust(8) for value in values)
        encoded = "CHAR"
    else:
        dtype = {"INTE": ">i4", "REAL": ">f4", "DOUB": ">f8"}[kind]
        array = np.asarray(values, dtype=dtype)
        payload = array.tobytes()
        values = array
        encoded = kind
    header = (name.encode("ascii").ljust(8) +
              struct.pack(">i", len(values)) + encoded.encode("ascii"))
    return _record(header) + _record(payload)


def _write(path, *fields):
    path.write_bytes(b"".join(_field(*field) for field in fields))


def _fixture(tmp_path):
    prefix = tmp_path / "AQCASE"
    _write(prefix.with_suffix(".INIT"),
           ("PORV", "REAL", [1, 1, 0, 1]))
    _write(prefix.with_suffix(".EGRID"),
           ("GRIDHEAD", "INTE", [1, 2, 2, 1]),
           ("ACTNUM", "INTE", [1, 1, 0, 1]),
           ("COORD", "REAL", np.arange(12)),
           ("ZCORN", "REAL", np.arange(32)))
    for step, pressure, sw in (
        (1, [200, 201, 203], [.2, .3, .4]),
        (2, [198, 200, 202], [.25, .35, .45]),
    ):
        _write(tmp_path / f"AQCASE.X{step:04d}",
               ("PRESSURE", "REAL", pressure),
               ("SWAT", "REAL", sw))
    _write(prefix.with_suffix(".SMSPEC"),
           ("WGNAMES", "CHAR", ["FIELD", "AQ1", "AQ1"]),
           ("KEYWORDS", "CHAR", ["TIME", "AAQP", "AAQT"]))
    _write(prefix.with_suffix(".UNSMRY"),
           ("MINISTEP", "INTE", [0]),
           ("PARAMS", "REAL", [0, 250, 0]),
           ("MINISTEP", "INTE", [1]),
           ("PARAMS", "REAL", [20, 245, 1000]))
    return prefix


def test_loadAquiferEclipseResult_reads_grid_restart_and_summary(tmp_path):
    result = load_aquifer_eclipse_result(_fixture(tmp_path))
    assert result.grid.cart_dims == (2, 2, 1)
    assert np.array_equal(result.grid.active, [0, 1, 3])
    assert len(result.states) == 2
    assert np.array_equal(result.states[0].pressure,
                          np.array([200, 201, 203]) * BARSA)
    assert np.allclose(result.states[1].s[:, 1], [.75, .65, .55])
    assert np.array_equal(result.summary.time, [0, 20])
    assert np.array_equal(result.summary.get("AAQP", "AQ1"), [250, 245])
    assert set(result.summary.aquifer_vectors) == {("AQ1", "AAQP"), ("AQ1", "AAQT")}


def test_loadAquiferEclipseResult_compares_pymrst_states_and_aquifer(tmp_path):
    result = load_aquifer_eclipse_result(_fixture(tmp_path))
    states = [
        State(pressure=state.pressure.copy(), sW=state.s[:, 0].copy())
        for state in result.states
    ]
    error = result.compare_states(states)
    assert np.all(error["pressure_relative"] == 0)
    assert np.all(error["water_saturation"] == 0)
    history = {
        "aquifer_p": np.array([[250], [245]]) * BARSA,
        "aquifer_V": np.array([[0], [1000]]),
    }
    aquifer_error = result.compare_aquifer_history(history, name="AQ1")
    assert np.all(aquifer_error["aquifer_pressure_relative"] == 0)
    assert np.all(aquifer_error["aquifer_volume"] == 0)


def test_loadAquiferEclipseResult_reads_unified_restart(tmp_path):
    prefix = _fixture(tmp_path)
    for path in tmp_path.glob("AQCASE.X????"):
        path.unlink()
    _write(prefix.with_suffix(".UNRST"),
           ("SEQNUM", "INTE", [7]),
           ("PRESSURE", "REAL", [200, 201, 203]),
           ("SWAT", "REAL", [.2, .3, .4]),
           ("SEQNUM", "INTE", [9]),
           ("PRESSURE", "REAL", [198, 200, 202]),
           ("SWAT", "REAL", [.25, .35, .45]))
    result = load_aquifer_eclipse_result(prefix)
    assert [state.report_step for state in result.states] == [7, 9]
    assert np.allclose(result.states[1].saturation[:, 0], [.25, .35, .45])
