from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.inout import read_sim_data, read_sim_rock, write_sim_data
from pymrst.utils.units import DARCY, MILLI


@pytest.mark.parametrize(
    ("dims", "physical"),
    [([3, 2], [6.0, 2.0]), ([2, 2, 1], [4.0, 2.0, 1.0])],
)
def test_sam_grid_round_trip_preserves_topology_and_geometry(tmp_path, dims, physical):
    grid = compute_geometry(cart_grid(dims, physical))
    case = tmp_path / "sam" / "case"
    write_sim_data(grid, case)
    loaded = read_sim_data(case)
    loaded.check()
    assert loaded.griddim == grid.griddim
    assert loaded.cells.num == grid.cells.num
    assert loaded.faces.num == grid.faces.num
    assert loaded.nodes.num == grid.nodes.num
    assert np.array_equal(loaded.cells.facePos, grid.cells.facePos)
    assert np.array_equal(loaded.cells.faces, grid.cells.faces)
    assert np.array_equal(loaded.faces.nodePos, grid.faces.nodePos)
    assert np.array_equal(loaded.faces.nodes, grid.faces.nodes)
    assert np.array_equal(loaded.faces.neighbors, grid.faces.neighbors)
    assert np.allclose(loaded.nodes.coords, grid.nodes.coords)
    assert np.allclose(loaded.faces.centroids, grid.faces.centroids)
    assert np.allclose(loaded.faces.normals, grid.faces.normals)
    assert np.allclose(loaded.faces.areas, grid.faces.areas)
    assert np.allclose(loaded.cells.centroids, grid.cells.centroids)
    assert np.allclose(loaded.cells.volumes, grid.cells.volumes)


def test_sam_optional_ijk_map_round_trip(tmp_path):
    grid = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    grid.cells.ijkMap = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=int
    )
    case = tmp_path / "mapped"
    write_sim_data(grid, case)
    loaded = read_sim_data(case)
    assert loaded.cartDims == (2, 2, 1)
    assert np.array_equal(loaded.cells.ijkMap, grid.cells.ijkMap)
    assert np.array_equal(loaded.cells.indexMap, np.arange(4))


def test_sam_rock_round_trip_converts_millidarcy_and_keeps_anisotropy(tmp_path):
    rock = Rock(
        perm=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]) * MILLI * DARCY,
        poro=np.array([0.2, 0.3]),
    )
    case = tmp_path / "rock"
    write_sim_data(casename=case, rock=rock)
    loaded = read_sim_rock(case)
    assert np.allclose(loaded.perm, rock.perm)
    assert np.allclose(loaded.poro, rock.poro)


def test_sam_rock_reader_rejects_record_count_mismatch(tmp_path):
    case = tmp_path / "bad"
    (tmp_path / "bad-perm.dat").write_text("2\n10\n20\n")
    (tmp_path / "bad-poro.dat").write_text("1\n0.2\n")
    with pytest.raises(ValueError, match="record counts"):
        read_sim_rock(case)


def test_sam_writer_emits_sources_and_wells(tmp_path):
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    rock = Rock(np.ones((2, 1)) * 10.0 * MILLI * DARCY, np.full(2, 0.25))
    wells = add_well(None, grid, rock, [1], well_type="bhp", val=5.0e6,
                     compi=[1.0, 0.0], name="P1")
    source = SimpleNamespace(cell=np.array([0]), rate=np.array([1.0e-5]),
                             sat=np.array([[1.0, 0.0]]))
    case = tmp_path / "forcing"
    write_sim_data(casename=case, src=source, wells=wells)
    text = (tmp_path / "forcing-wells.dat").read_text()
    assert text.splitlines()[0] == "2"
    assert "SimpleInjection I1" in text
    assert "Pressure P1" in text
