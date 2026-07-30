from examples.validation.validate_grid_process_grdecl_omo import main


def test_process_grdecl_omo_matches_mrst():
    assert main() == 0
