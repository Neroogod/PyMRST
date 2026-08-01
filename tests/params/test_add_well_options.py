import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.compute_well_index import cell_dims, compute_well_index


def fixture():
    grid = compute_geometry(cart_grid([2, 2, 2], [20.0, 20.0, 20.0]))
    rock = make_rock(
        grid,
        np.tile([100e-15, 50e-15, 10e-15], (grid.cells.num, 1)),
        0.2,
    )
    return grid, rock


def test_add_well_accepts_logical_mask_and_empty_mask_is_noop():
    grid, rock = fixture()
    mask = np.zeros(grid.cells.num, dtype=bool)
    mask[[0, 4]] = True
    wells = add_well(None, grid, rock, mask, name="MASK")
    np.testing.assert_array_equal(wells[0].cells, [0, 4])
    assert wells[0].dir == "zz"
    with pytest.warns(RuntimeWarning, match="Empty cell selection"):
        unchanged = add_well(wells, grid, rock, np.zeros(grid.cells.num, bool))
    assert unchanged[0] is wells[0]
    with pytest.raises(ValueError, match="one entry per grid cell"):
        add_well(None, grid, rock, np.ones(2, bool))


def test_add_well_computes_only_negative_well_indices():
    grid, rock = fixture()
    cells = np.array([0, 2, 4])
    supplied = np.array([12.5, -1.0, 7.5])
    well = add_well(
        None, grid, rock, cells, wi=supplied, radius=[0.1, 0.12, 0.14],
        skin=[0.0, 1.0, 2.0],
    )[0]
    expected_middle = compute_well_index(
        grid, rock, [0.12], [2], well_dir="z", skin=[1.0]
    )[0]
    np.testing.assert_allclose(well.WI, [12.5, expected_middle, 7.5])
    np.testing.assert_array_equal(supplied, [12.5, -1.0, 7.5])


def test_add_well_populates_mrst_structure_fields_and_repr_radius():
    grid, rock = fixture()
    well = add_well(
        None,
        grid,
        rock,
        [0, 1],
        well_type="rate",
        val=-2.0,
        sign=1,
        status=False,
        cstatus=[True, False],
        vfp_index=3,
        cp={"region": 2},
        components=[0.2, 0.8],
        heel=[1.0, 2.0, 3.0],
        lims={"bhp": 1e7},
    )[0]
    assert well.sign == -1  # rate target overrides contradictory sign, as MRST
    assert well.status is False
    np.testing.assert_array_equal(well.cstatus, [True, False])
    assert well.vfp_index == 3 and well.cp == {"region": 2}
    np.testing.assert_allclose(well.components, [0.2, 0.8])
    assert well.heel == [1.0, 2.0, 3.0]
    assert well.defaulted == [] and well.lims == {"bhp": 1e7}
    dimensions = cell_dims(grid, [0, 1])
    re = np.sqrt(dimensions[:, 0]*dimensions[:, 1]/np.pi)
    np.testing.assert_allclose(well.rR, np.sqrt(re*0.1))


def test_add_well_line_segments_use_projected_directional_indices():
    grid, rock = fixture()
    cells = np.array([0, 4])
    dimensions = cell_dims(grid, cells)
    segments = np.array([[3.0, 4.0, 5.0], [5.0, 2.0, 1.0]])
    well = add_well(
        None, grid, rock, cells, line_segments=segments,
        cell_dims_override=dimensions,
    )[0]
    expected_squared = np.zeros(cells.size)
    for axis, direction in enumerate("xyz"):
        directional = compute_well_index(
            grid, rock, 0.1, cells, well_dir=direction,
            cell_dims_override=dimensions,
        )
        expected_squared += (segments[:, axis]/dimensions[:, axis]*directional)**2
    np.testing.assert_allclose(well.WI, np.sqrt(expected_squared))


def test_add_well_accepts_full_grid_cell_dimensions():
    grid, rock = fixture()
    all_dimensions = cell_dims(grid, np.arange(grid.cells.num))
    selected = [0, 4]
    from_full = add_well(
        None, grid, rock, selected, cell_dims_override=all_dimensions
    )[0]
    from_subset = add_well(
        None, grid, rock, selected,
        cell_dims_override=all_dimensions[selected],
    )[0]
    np.testing.assert_allclose(from_full.WI, from_subset.WI)


def test_add_well_validates_advanced_option_shapes():
    grid, rock = fixture()
    with pytest.raises(ValueError, match="cstatus"):
        add_well(None, grid, rock, [0, 1], cstatus=[True])
    with pytest.raises(ValueError, match="line_segments"):
        add_well(None, grid, rock, [0, 1], line_segments=np.ones((2, 2)))
    with pytest.raises(ValueError, match="unknown inner product"):
        add_well(None, grid, rock, [0], inner_product="not_an_ip")


def test_add_well_supports_mrst_mimetic_inner_products():
    grid, rock = fixture()
    cells = [0, 1]
    tpf = add_well(None, grid, rock, cells, inner_product="ip_tpf")[0]
    quasi = add_well(None, grid, rock, cells, inner_product="ip_quasitpf")[0]
    simple = add_well(None, grid, rock, cells, inner_product="ip_simple")[0]
    rt = add_well(None, grid, rock, cells, inner_product="ip_rt")[0]
    quasirt = add_well(None, grid, rock, cells, inner_product="ip_quasirt")[0]
    np.testing.assert_allclose(quasi.WI, tpf.WI)
    np.testing.assert_allclose(rt.WI, simple.WI)
    np.testing.assert_allclose(quasirt.WI, simple.WI)
    assert np.all(simple.WI < tpf.WI)
