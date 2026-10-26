"""MRST ``incomp/treatLegacyForceOptions.m``: the solvers take wells as
either `'wells'` (legacy) or `'W'`; whichever is given is copied to the
other, and giving both is an error."""

from typing import Any, Dict


def treat_legacy_force_options(opt: Dict[str, Any]) -> Dict[str, Any]:
    """MRST `opt = treatLegacyForceOptions(opt)`, on an option dict."""
    wells = opt.get("wells")
    W = opt.get("W")
    both = "Solver cannot have both legacy 'wells' argument and 'W' argument at the same time"
    if not _is_empty(W):
        assert _is_empty(wells), both
        opt["wells"] = W
    if not _is_empty(wells):
        assert _is_empty(W), both
        opt["W"] = wells
    return opt


def _is_empty(value: Any) -> bool:
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


treatLegacyForceOptions = treat_legacy_force_options
__all__ = ["treat_legacy_force_options", "treatLegacyForceOptions"]


if __name__ == "__main__":
    def test_either_spelling_fills_the_other():
        assert treat_legacy_force_options({"wells": [1], "W": None})["W"] == [1]
        assert treat_legacy_force_options({"wells": [], "W": [2]})["wells"] == [2]
        print("ok test_either_spelling_fills_the_other")

    def test_both_given_is_an_error():
        try:
            treat_legacy_force_options({"wells": [1], "W": [2]})
        except AssertionError:
            print("ok test_both_given_is_an_error")
            return
        raise AssertionError("expected an AssertionError")

    test_either_spelling_fills_the_other()
    test_both_given_is_an_error()
    print("\nAll tests passed!")
