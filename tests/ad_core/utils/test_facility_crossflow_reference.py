from examples.validation.validate_ad_core_facility_crossflow import main


def test_facility_crossflow_matches_mrst_reference():
    assert main() == 0
