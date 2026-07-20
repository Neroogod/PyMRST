"""Value extraction, ported from MRST `value`."""

import numpy as np

from pymrst.utils.adi.adi import ADI
from pymrst.utils.adi.adi2 import ADI2


def value(x) -> np.ndarray:
    """Return the underlying value of an ADI, or the input as an array."""
    return x.val if isinstance(x, (ADI, ADI2)) or x.__class__.__name__ == "GenericAD" else np.asarray(x)


if __name__ == "__main__":

    def test_value() -> None:
        adi = ADI(np.array([1.0, 2.0]), [])
        assert np.array_equal(value(adi), [1.0, 2.0])
        assert np.array_equal(value([3.0, 4.0]), [3.0, 4.0])
        print("ok test_value")

    test_value()
    print("\nAll tests passed!")
