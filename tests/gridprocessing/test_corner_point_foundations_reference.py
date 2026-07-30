from examples.validation.validate_grid_corner_point_foundations import main


def test_corner_point_foundations_match_mrst():
    assert main() == 0
