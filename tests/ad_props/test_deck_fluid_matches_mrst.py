"""The deck-to-fluid pipeline against MRST's own, as a test.

`examples/validation/validate_ad_props_deck_fluid.py` is the script form,
which prints every comparison and is what to run when investigating. This
wraps the same comparison so a regression is caught by the test suite rather
than only when somebody remembers to run the validator.

It skips when `ref_ad_props_deck_fluid.mat` is absent, since regenerating it
needs MATLAB. Do not weaken that to a pass: a missing reference means the
check did not happen.
"""

import sys
from pathlib import Path

import pytest

VALIDATION = Path(__file__).resolve().parents[2] / "examples" / "validation"
REFERENCE = VALIDATION / "ref_ad_props_deck_fluid.mat"


@pytest.fixture(scope="module")
def validator():
    if not REFERENCE.exists():
        pytest.skip(f"{REFERENCE.name} absent; regenerate it with "
                    f"dump_ad_props_deck_fluid_reference.m in MATLAB")
    sys.path.insert(0, str(VALIDATION))
    try:
        import validate_ad_props_deck_fluid as module
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def reference(validator):
    from scipy.io import loadmat
    return loadmat(validator.REFERENCE, struct_as_record=True)


@pytest.mark.parametrize("deck", ["SPE1", "SPE3", "SPE9"])
def test_every_fluid_property_matches_mrst(deck, validator, reference, capsys):
    """Reader, unit conversion, regions and assignment, compared end to end."""
    assert deck in reference, f"{deck} missing from the reference file"
    failures = validator.compare(deck, reference[deck])
    capsys.readouterr()          # the comparison prints; the result is enough
    assert failures == [], failures


def test_the_reference_covers_both_keyword_families(reference):
    """SPE1/SPE9 are PVTO+PVDG+SWOF; SPE3 is PVDO+PVTG+SWFN.

    Losing either family from the reference would leave half the assignment
    code unvalidated while the suite still passed.
    """
    def flag(deck, name):
        return bool(reference[deck][name][0, 0].ravel()[0])

    assert flag("SPE1", "live_oil") and not flag("SPE1", "wet_gas")
    assert not flag("SPE3", "live_oil") and flag("SPE3", "wet_gas")
    assert flag("SPE9", "live_oil") and not flag("SPE9", "wet_gas")
