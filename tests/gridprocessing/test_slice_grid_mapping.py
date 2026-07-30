import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid


def test_slice_grid_returns_mrst_style_mapping_and_preserves_index_map():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    grid.cells.indexMap = np.array([8, 12])
    sliced, gix = slice_grid(
        grid, np.array([0.5, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]),
        return_mapping=True,
    )
    assert sliced.cells.num == 3
    assert np.array_equal(sliced.cells.indexMap, [8, 12, 8])
    assert np.array_equal(gix.parent.cells, [0, 1, 0])
    assert np.array_equal(gix.new.cells, [2, 1, 2])
    assert np.count_nonzero(gix.new.faces == 3) == 1
    assert gix.parent.faces[gix.new.faces == 3].item() == -1
    assert gix.old.cells.tolist() == [False, True]


def test_slice_grid_multiple_cuts_compose_parent_mapping():
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    sliced, gix = slice_grid(
        grid,
        [np.array([0.4, 0.0, 0.0]), np.array([1.6, 0.0, 0.0])],
        normal=[np.array([1.0, 0.0, 0.0])] * 2,
        return_mapping=True,
    )
    sliced.check()
    assert sliced.cells.num == 4
    assert np.array_equal(gix.parent.cells, [0, 1, 0, 1])
    assert np.count_nonzero(gix.new.faces == 3) == 2


def test_slice_grid_returns_lower_dimensional_slice_grid():
    grid = cart_grid([2, 2, 1], [2.0, 2.0, 1.0])
    sliced, gix, cross_section = slice_grid(
        grid,
        np.array([0.5, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        return_mapping=True,
        return_slice=True,
    )
    sliced.check()
    cross_section.check()
    assert cross_section.griddim == 2
    assert cross_section.cells.num == 2
    assert np.array_equal(
        cross_section.cells.parent, np.flatnonzero(gix.new.faces == 3)
    )
    assert cross_section.faces.num == 7
    assert cross_section.nodes.num == 6


def test_slice_grid_multiple_cuts_returns_all_slice_surfaces():
    grid = cart_grid([2, 2, 1], [2.0, 2.0, 1.0])
    sliced, cross_sections = slice_grid(
        grid,
        [np.array([0.4, 0.0, 0.0]), np.array([1.6, 0.0, 0.0])],
        normal=[np.array([1.0, 0.0, 0.0])] * 2,
        return_slice=True,
    )
    sliced.check()
    cross_sections.check()
    assert cross_sections.cells.num == 4
    assert cross_sections.faces.num == 14
    assert cross_sections.nodes.num == 12


def test_slice_grid_empty_cut_returns_no_slice_grid():
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    unchanged, cross_section = slice_grid(
        grid,
        np.array([2.0, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        return_slice=True,
    )
    unchanged.check()
    assert cross_section is None


def test_slice_grid_only_faces_splits_faces_without_splitting_cells():
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    sliced, gix, cross_section = slice_grid(
        grid,
        np.array([0.5, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        only_faces=True,
        return_mapping=True,
        return_slice=True,
    )
    sliced.check()
    assert sliced.cells.num == 1
    assert sliced.faces.num == 10
    assert sliced.nodes.num == 12
    assert np.count_nonzero(gix.new.cells == 2) == 0
    assert np.count_nonzero(gix.new.faces == 2) == 8
    assert np.count_nonzero(gix.new.faces == 3) == 0
    assert np.count_nonzero(gix.old.faces) == 2
    assert cross_section is None


def test_slice_grid_multiple_face_only_cuts_compose_mapping():
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    sliced, gix = slice_grid(
        grid,
        [np.array([0.35, 0.0, 0.0]), np.array([0.0, 0.65, 0.0])],
        normal=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
        only_faces=True,
        return_mapping=True,
    )
    sliced.check()
    assert sliced.cells.num == 1
    assert np.array_equal(gix.parent.cells, [0])
    assert gix.old.cells.tolist() == [True]
    assert np.count_nonzero(gix.new.faces == 3) == 0
    assert np.all(gix.new.nodes[grid.nodes.num:] == 2)


def test_slice_grid_gix_input_matches_one_shot_mapping_composition():
    first_point = np.array([0.4, 0.0, 0.0])
    second_point = np.array([1.6, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])

    staged_grid, first_ix = slice_grid(
        cart_grid([2, 1, 1], [2.0, 1.0, 1.0]),
        first_point, normal=normal, return_mapping=True,
    )
    staged_grid, staged_ix = slice_grid(
        staged_grid, second_point, normal=normal, return_mapping=True,
        gix_input=first_ix,
    )
    combined_grid, combined_ix = slice_grid(
        cart_grid([2, 1, 1], [2.0, 1.0, 1.0]),
        [first_point, second_point], normal=[normal, normal],
        return_mapping=True,
    )

    staged_grid.check()
    combined_grid.check()
    for entity in ("cells", "faces", "nodes"):
        assert np.array_equal(
            getattr(staged_ix.parent, entity), getattr(combined_ix.parent, entity)
        )
        assert np.array_equal(
            getattr(staged_ix.new, entity), getattr(combined_ix.new, entity)
        )
    assert np.array_equal(staged_ix.old.cells, combined_ix.old.cells)
    assert np.array_equal(staged_ix.old.faces, combined_ix.old.faces)
    assert staged_ix.old.cells.tolist() == [False, False]
    assert staged_ix.old.faces.tolist() == [
        True, True, True, False, False, False, False, False, False, False, False
    ]


def test_slice_grid_gix_input_composes_with_a_multi_cut_call():
    normal = np.array([1.0, 0.0, 0.0])
    grid, initial = slice_grid(
        cart_grid([3, 1, 1], [3.0, 1.0, 1.0]),
        np.array([0.4, 0.0, 0.0]), normal=normal, return_mapping=True,
    )
    grid, mapping = slice_grid(
        grid,
        [np.array([1.4, 0.0, 0.0]), np.array([2.4, 0.0, 0.0])],
        normal=[normal, normal], gix_input=initial, return_mapping=True,
    )
    grid.check()
    assert grid.cells.num == 6
    assert np.array_equal(mapping.parent.cells, [0, 1, 2, 0, 1, 2])
    assert mapping.old.cells.tolist() == [False, False, False]
