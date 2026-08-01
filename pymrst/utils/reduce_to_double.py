"""Reduce a quantity to plain doubles (MRST ``utils/reduceToDouble.m``).

**MRST's implementation is empty** - `function u = reduceToDouble(u); end` -
so it returns its argument untouched. It is a hook: a place for a backend that
carries something heavier than a double array (a GPU buffer, a distributed
vector) to collapse it, without every call site knowing whether it needs to.

Reproduced as a passthrough, because changing it to strip AD derivatives -
the plausible guess - would silently break every caller that passes an AD
value through and expects to keep differentiating. `value` is the function
that strips derivatives, and it is a different function.
"""


def reduce_to_double(u):
    """Return ``u`` unchanged; see the module docstring."""
    return u


reduceToDouble = reduce_to_double


if __name__ == "__main__":
    import numpy as np
    import scipy.sparse as sp

    from pymrst.utils.adi import ADI

    def test_an_array_passes_through_unchanged() -> None:
        values = np.arange(4.0)
        assert reduce_to_double(values) is values
        print("ok test_an_array_passes_through_unchanged")

    def test_an_ad_value_keeps_its_derivatives() -> None:
        """The behaviour that distinguishes it from `value`."""
        variable = ADI(np.arange(3.0), [sp.csr_array(sp.eye(3, format="csr"))])
        assert reduce_to_double(variable) is variable
        assert reduce_to_double(variable).jac, "derivatives were dropped"
        print("ok test_an_ad_value_keeps_its_derivatives")

    def test_it_differs_from_value() -> None:
        from pymrst.utils.adi import value
        variable = ADI(np.arange(3.0), [sp.csr_array(sp.eye(3, format="csr"))])
        assert not isinstance(value(variable), ADI)
        assert isinstance(reduce_to_double(variable), ADI)
        print("ok test_it_differs_from_value")

    test_an_array_passes_through_unchanged()
    test_an_ad_value_keeps_its_derivatives()
    test_it_differs_from_value()
    print("\nAll tests passed!")
