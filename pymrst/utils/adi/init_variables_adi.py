"""Primary-variable initialization, ported from MRST `initVariablesADI`."""

from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi.adi import ADI

Numeric = Union[float, int, np.ndarray]


def init_variables_adi(*values: Numeric) -> Tuple[ADI, ...]:
    """Declare primary variables with identity Jacobians in their own blocks."""
    vals = [np.atleast_1d(np.asarray(v, dtype=np.float64)).ravel() for v in values]
    sizes = [v.size for v in vals]
    out = []
    for i, val in enumerate(vals):
        jac = [
            sp.eye_array(val.size, format="csr") if i == j else sp.csr_array((val.size, m))
            for j, m in enumerate(sizes)
        ]
        out.append(ADI(val, jac))
    return tuple(out)


if __name__ == "__main__":

    def test_init_variables_adi() -> None:
        p, b, _ = init_variables_adi(np.array([1.0, 2.0, 3.0]), 5.0, 0.0)
        assert len(p) == 3
        assert len(b) == 1
        assert np.allclose(p.jac[0].toarray(), np.eye(3))
        assert p.jac[1].shape == (3, 1)
        assert p.jac[1].nnz == 0
        assert np.allclose(b.jac[1].toarray(), [[1.0]])
        print("ok test_init_variables_adi")

    test_init_variables_adi()
    print("\nAll tests passed!")

