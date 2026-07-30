from examples.validation.validate_grid_mark_cut_grids import main


def test_mark_cut_grids_matches_mrst():
    assert main() == 0
