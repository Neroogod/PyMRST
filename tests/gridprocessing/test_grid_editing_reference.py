from examples.validation.validate_grid_editing import main


def test_grid_editing_matches_mrst():
    assert main() == 0
