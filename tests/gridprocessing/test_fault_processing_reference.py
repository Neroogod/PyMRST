from examples.validation.validate_grid_fault_processing import main


def test_fault_processing_matches_mrst():
    assert main() == 0
