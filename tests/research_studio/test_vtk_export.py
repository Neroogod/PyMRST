"""VTK/ParaView export (M3.5).

The claim being protected is narrow and checkable: what ParaView reads back is
what the simulation produced. Not close to it - equal. A visualization format
that quietly rounds is worse than none, because the picture still looks
right.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="VTK export needs the plotting extra")

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.research_studio.export import read_vtk_series, write_vtk_series


def _case(dims=(3, 2, 1), steps=3):
    grid = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
    rng = np.random.default_rng(0)
    states = [
        {"pressure": 2.7e7 + rng.random(grid.cells.num)*1e6,
         "sW": rng.random(grid.cells.num)}
        for _ in range(steps)
    ]
    return grid, states


def test_values_read_back_bit_identical(tmp_path):
    """Zero, not a tolerance.

    The values written are the ones the simulation produced, and VTK stores
    float64 verbatim, so there is no reason for any difference. Asserting
    `allclose` here would accept a format that silently narrowed to float32,
    which is exactly the loss this is meant to exclude.
    """
    grid, states = _case()
    collection = write_vtk_series(tmp_path/"out", grid, states,
                                  fields=("pressure", "sW"))
    restored = read_vtk_series(collection)

    assert len(restored) == len(states)
    for (_, mesh), state in zip(restored, states):
        for field in ("pressure", "sW"):
            np.testing.assert_array_equal(
                np.asarray(mesh.cell_data[field], dtype=float), state[field],
            )


def test_the_collection_carries_simulation_time_not_step_number(tmp_path):
    """ParaView indexes a collection by time; the slider is only useful if the
    times are the schedule's, not 0,1,2."""
    grid, states = _case(steps=3)
    times = [0.0, 86400.0, 518400.0]
    collection = write_vtk_series(tmp_path/"out", grid, states, times=times)

    recorded = [float(e.attrib["timestep"])
                for e in ET.parse(collection).getroot().iter("DataSet")]
    assert recorded == times


def test_times_must_correspond_to_states(tmp_path):
    """Off-by-one here is easy - a schedule has one more state than it has
    steps - and would silently mislabel every frame."""
    grid, states = _case(steps=3)
    with pytest.raises(ValueError, match="must correspond"):
        write_vtk_series(tmp_path/"out", grid, states, times=[0.0, 1.0])


def test_the_collection_references_files_relatively(tmp_path):
    """So the directory can be copied or moved and still open."""
    grid, states = _case()
    collection = write_vtk_series(tmp_path/"out", grid, states)
    for entry in ET.parse(collection).getroot().iter("DataSet"):
        assert "/" not in entry.attrib["file"]
        assert (collection.parent/entry.attrib["file"]).exists()

    moved = tmp_path/"moved"
    (tmp_path/"out").rename(moved)
    restored = read_vtk_series(moved/"state.pvd")
    assert len(restored) == len(states)


def test_a_field_that_is_not_cell_data_is_refused(tmp_path):
    """States carry well unknowns and mechanics degrees of freedom whose
    length is not the cell count. Writing one as cell data would produce a
    file ParaView renders as nonsense, so it fails instead."""
    grid, states = _case()
    for state in states:
        state["bhp"] = np.array([3.0e7])
    with pytest.raises(ValueError, match="not cell data"):
        write_vtk_series(tmp_path/"out", grid, states, fields=("bhp",))


def test_wells_are_written_as_their_own_polyline_dataset(tmp_path):
    """ParaView cannot infer a trajectory from cell data.

    Without a separate dataset a well is simply invisible, which is the most
    common complaint about hand-rolled VTK exports.
    """
    grid, states = _case()
    wells = [SimpleNamespace(cells=np.array([0, 1, 2])),
             SimpleNamespace(cells=np.array([3, 4]))]
    write_vtk_series(tmp_path/"out", grid, states, wells=wells)

    trajectories = pv.read(tmp_path/"out"/"wells.vtp")
    assert trajectories.n_points == 5
    assert set(np.asarray(trajectories.point_data["well"])) == {0, 1}


def test_a_well_with_no_perforations_does_not_break_the_export(tmp_path):
    grid, states = _case()
    write_vtk_series(tmp_path/"out", grid, states,
                     wells=[SimpleNamespace(cells=np.array([], dtype=np.int64))])
    assert not (tmp_path/"out"/"wells.vtp").exists()


def test_geometry_is_written_once_per_step_but_built_once(tmp_path):
    """Each step is a standalone `.vtu`, which ParaView needs, but the
    polyhedral cell array is constructed a single time - that construction is
    the expensive part on corner-point geometry."""
    grid, states = _case(dims=(4, 4, 2), steps=4)
    calls = {"n": 0}
    import pymrst.plotting as plotting
    from pymrst.plotting import to_pyvista as real

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    plotting.to_pyvista = counting
    try:
        write_vtk_series(tmp_path/"out", grid, states)
    finally:
        plotting.to_pyvista = real
    assert calls["n"] == 1, f"grid converted {calls['n']} times for {len(states)} steps"


def test_writing_no_states_is_an_error_rather_than_an_empty_case(tmp_path):
    grid, _ = _case()
    with pytest.raises(ValueError, match="no states"):
        write_vtk_series(tmp_path/"out", grid, [])


def test_export_is_additive_and_unimported_by_the_replica_layer():
    root = Path(__file__).resolve().parents[2]/"pymrst"
    offenders = [
        path for path in (root/"modules").rglob("*.py")
        if "research_studio" in path.read_text(errors="ignore")
    ]
    assert not offenders, f"replica modules importing the extension: {offenders}"
