"""Validate ``getMultiDimInterpolator``, ``bc2ADbc``, and ``HandleStruct``
against MRST.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    HandleStruct,
    bc_to_ad_bc,
    get_multi_dim_interpolator,
)
from pymrst.utils.adi import ADI


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_interpolator_bc_handle.mat",
        squeeze_me=True,
    )
    x = np.array([2.0, 1.0, 0.0])
    y = np.array([0.0, 3.0])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    interpolate, _, epsilons = get_multi_dim_interpolator(
        [x, y], 2.0 * xx + 3.0 * yy
    )
    numeric = interpolate(np.array([0.5, 1.5]), np.array([1.0, 2.0]))
    x_ad = ADI(np.array([0.5, 1.5]), [sp.eye(2, format="csr")])
    ad_value = interpolate(x_ad, np.array([1.0, 2.0]))

    grid = SimpleNamespace(
        faces=SimpleNamespace(neighbors=np.array([[-1, 0], [0, 1], [1, -1]])),
        cells=SimpleNamespace(num=2),
    )
    boundary = SimpleNamespace(face=np.array([0, 2]), value=np.array([10, 20]))
    converted = bc_to_ad_bc(grid, boundary)

    wrapped = HandleStruct({
        "ad": ADI(np.array([1.0, 2.0]), [sp.eye(2, format="csr")]),
        "parts": [np.array([3.0]), np.array([4.0, 5.0])],
    })
    reduced = wrapped.reduce()
    checks = {
        "epsilons": (epsilons, ref["epsilons"]),
        "numeric interpolation": (numeric, ref["numeric_value"]),
        "AD values": (ad_value.val, ref["ad_value"]),
        "AD Jacobian": (ad_value.jac[0].toarray(), ref["ad_jacobian"]),
        "cell to BC face": (
            converted.cell2bcface.toarray(),
            ref["cell2bcface"],
        ),
        "BC face to cell": (
            converted.bcface2cell.toarray(),
            ref["bcface2cell"],
        ),
        "reduced AD": (reduced["ad"], ref["reduced_ad"]),
        "reduced parts": (reduced["parts"], ref["reduced_parts"]),
        "property evaluated": (
            wrapped.struct_prop_evaluated("ad"),
            ref["evaluated_ad"],
        ),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
