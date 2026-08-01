"""Element count of a possibly-AD quantity (MRST ``utils/numelValue.m``).

`len()` on an ADI object would be ambiguous - the value has one length, the
Jacobian another - so this asks for the count of the underlying *value*. A
property that must return one entry per cell uses it to size its output without
caring whether it was handed a plain array or an AD variable.
"""

import numpy as np

from pymrst.utils.adi import value


def numel_value(v) -> int:
    """Number of elements in ``v``'s value, AD or not."""
    return int(np.size(value(v)))


numelValue = numel_value


if __name__ == "__main__":
    import scipy.sparse as sp

    from pymrst.utils.adi import ADI

    def test_a_plain_array_counts_its_elements() -> None:
        assert numel_value(np.zeros(7)) == 7
        assert numel_value(np.zeros((2, 3))) == 6
        print("ok test_a_plain_array_counts_its_elements")

    def test_a_scalar_counts_as_one() -> None:
        assert numel_value(3.5) == 1
        print("ok test_a_scalar_counts_as_one")

    def test_an_ad_variable_counts_its_value_not_its_jacobian() -> None:
        """The reason this exists: the two have different sizes."""
        values = np.arange(4.0)
        variable = ADI(values, [sp.csr_array(sp.eye(4, format="csr"))])
        assert numel_value(variable) == 4
        assert variable.jac[0].shape == (4, 4), "the Jacobian is bigger"
        print("ok test_an_ad_variable_counts_its_value_not_its_jacobian")

    def test_it_agrees_for_a_variable_and_its_value() -> None:
        values = np.arange(5.0)
        variable = ADI(values, [sp.csr_array(sp.eye(5, format="csr"))])
        assert numel_value(variable) == numel_value(values)
        print("ok test_it_agrees_for_a_variable_and_its_value")

    def test_an_empty_array_counts_zero() -> None:
        assert numel_value(np.zeros(0)) == 0
        print("ok test_an_empty_array_counts_zero")

    test_a_plain_array_counts_its_elements()
    test_a_scalar_counts_as_one()
    test_an_ad_variable_counts_its_value_not_its_jacobian()
    test_it_agrees_for_a_variable_and_its_value()
    test_an_empty_array_counts_zero()
    print("\nAll tests passed!")
