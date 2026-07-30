from examples.validation.validate_grid_process_pinch import main


def test_process_pinch_matches_mrst():
    assert main() == 0
