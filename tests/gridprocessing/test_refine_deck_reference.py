from examples.validation.validate_grid_refine_deck import main


def test_refine_deck_matches_mrst():
    assert main() == 0
