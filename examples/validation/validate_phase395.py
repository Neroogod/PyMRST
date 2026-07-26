"""Phase 395: fully implicit AD Jacobian through NTPFA convex weights."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.nfvm import NTPFA
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi, value


def _case():
    g = cart_grid([5, 4], [5.0, 4.0])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y = coords.T.copy()
    coords[:, 0] += 0.30*np.sin(np.pi*x/5.0)*np.sin(np.pi*y/4.0)
    coords[:, 1] += 0.12*np.sin(2*np.pi*x/5.0)*np.sin(np.pi*y/4.0)
    g.nodes.coords = coords
    g = compute_geometry(g)
    rock = make_rock(
        g, np.tile([2.0, 0.4, 1.0], (g.cells.num, 1)), 0.2
    )
    rng = np.random.default_rng(4)
    pressure = (
        100.0 + 10.0*rng.random(g.cells.num)
        + np.asarray(g.cells.centroids) @ np.array([3.0, -2.0])
    )
    return g, rock, pressure


def main():
    g, rock, pressure = _case()
    implicit = NTPFA.from_grid(g, rock)
    lagged = NTPFA.from_grid(g, rock, lagged=True)

    (pressure_ad,) = init_variables_adi(pressure)
    implicit_ad = implicit.unscaled_gradient(pressure_ad)
    lagged_ad = lagged.unscaled_gradient(pressure_ad)
    jacobian = implicit_ad.jac[0].toarray()

    step = 1e-5
    finite_difference = np.empty_like(jacobian)
    for cell in range(g.cells.num):
        plus = pressure.copy()
        minus = pressure.copy()
        plus[cell] += step
        minus[cell] -= step
        finite_difference[:, cell] = (
            implicit.unscaled_gradient(plus)
            - implicit.unscaled_gradient(minus)
        )/(2.0*step)

    absolute_error = float(np.max(np.abs(jacobian - finite_difference)))
    relative_error = float(
        np.linalg.norm(jacobian - finite_difference)
        / np.linalg.norm(finite_difference)
    )
    value_parity = float(np.max(np.abs(value(implicit_ad) - value(lagged_ad))))
    derivative_difference = float(np.linalg.norm(
        jacobian - lagged_ad.jac[0].toarray()
    ))

    print("Phase 395: fully implicit NTPFA AD")
    print(f"  Jacobian max finite-difference error: {absolute_error:.3e}")
    print(f"  Jacobian relative error: {relative_error:.3e}")
    print(f"  implicit/lagged value parity: {value_parity:.3e}")
    print(f"  implicit/lagged Jacobian difference: {derivative_difference:.3e}")

    failed = (
        absolute_error > 1e-7
        or relative_error > 1e-7
        or value_parity > 1e-10
        or derivative_difference < 1e-4
    )
    print("\nPhase 395 passed." if not failed else "\nPhase 395 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
