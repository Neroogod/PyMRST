from examples.validation.validate_ad_blackoil_update_state import main


def test_update_state_matches_mrst_reference():
    assert main() == 0
