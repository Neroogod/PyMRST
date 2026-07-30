from examples.validation.validate_grid_nnc_operators import main


def test_nnc_operators_match_mrst():
    assert main() == 0
