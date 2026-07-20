"""Phase 263: getFlashTable utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.utils import (  # noqa: E402
    getFlashTable,
    get_flash_table,
    standaloneFlash,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mixture() -> Mixture:
    return Mixture(
        names=("C1", "C3", "C10"),
        Tcrit=np.array([190.56, 369.83, 617.7]),
        Pcrit=np.array([4.5992e6, 4.248e6, 2.11e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4, 6.17e-4]),
        acentric=np.array([0.0114, 0.1523, 0.4884]),
        molarMass=np.array([0.01604, 0.04410, 0.14229]),
    )


def _fix(vec: np.ndarray, nT: int, np_: int) -> np.ndarray:
    return np.asarray(vec).reshape((nT, np_), order="F").T


def main() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "Peng-Robinson")
    p_range = np.array([9.0e6, 1.2e7, 1.5e7])
    T_range = np.array([335.0, 365.0])
    z = np.array([0.52, 0.28, 0.20])

    L, x, y, Z_L, Z_V, p, T = getFlashTable(p_range, T_range, z, eos)
    Pmesh, Tmesh = np.meshgrid(p_range, T_range)
    flat_p = Pmesh.reshape(-1, order="F")
    flat_T = Tmesh.reshape(-1, order="F")
    flat_z = np.repeat(z.reshape(1, -1), flat_p.size, axis=0)
    direct = standaloneFlash(flat_p, flat_T, flat_z, eos)

    z_list = [np.repeat(z[i], flat_p.size) for i in range(z.size)]
    L_alias, x_alias, y_alias, ZL_alias, ZV_alias, p_alias, T_alias = get_flash_table(
        p_range, T_range, z_list, eos
    )

    print("\nPhase 263: getFlashTable utility parity")
    failures = 0
    failures += _check("tables have MRST mesh orientation",
                       L.shape == (p_range.size, T_range.size)
                       and np.allclose(p[:, 0], p_range)
                       and np.allclose(T[0, :], T_range))
    failures += _check("liquid fraction is one minus flash vapor fraction",
                       np.allclose(L, _fix(1.0 - direct.V, T_range.size, p_range.size)))
    failures += _check("compressibility-factor tables match direct flash",
                       np.allclose(Z_L, _fix(direct.Z_L, T_range.size, p_range.size))
                       and np.allclose(Z_V, _fix(direct.Z_V, T_range.size, p_range.size)))
    failures += _check("component liquid/vapor tables match direct flash",
                       len(x) == mix.n and len(y) == mix.n
                       and all(np.allclose(x[i], _fix(direct.x[:, i], T_range.size, p_range.size)) for i in range(mix.n))
                       and all(np.allclose(y[i], _fix(direct.y[:, i], T_range.size, p_range.size)) for i in range(mix.n)))
    failures += _check("component-list input follows camelCase result",
                       np.allclose(L_alias, L) and np.allclose(ZL_alias, Z_L) and np.allclose(ZV_alias, Z_V)
                       and np.allclose(p_alias, p) and np.allclose(T_alias, T)
                       and all(np.allclose(x_alias[i], x[i]) for i in range(mix.n))
                       and all(np.allclose(y_alias[i], y[i]) for i in range(mix.n)))

    if failures:
        print(f"\nPhase 263 failed ({failures} checks).")
        return 1
    print("\ngetFlashTable now builds MRST-oriented flash tables from native PyMRST flash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
