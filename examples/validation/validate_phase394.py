"""Phase 394: NFVM Hybrid, gravity setup, and model installation helpers."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.nfvm import (
    AvgMPFA,
    NTPFA,
    set_avg_mpfa_discretization,
    set_hybrid_discretization,
    set_ntpfa_discretization,
)
from pymrst.modules.nfvm.solvers.set_avg_mpfa_discretization import (
    setAvgMPFADiscretization,
)
from pymrst.modules.nfvm.solvers.set_hybrid_discretization import (
    setHybridDiscretization,
)
from pymrst.modules.nfvm.solvers.set_ntpfa_discretization import (
    setNTPFADiscretization,
)
from pymrst.params.rock import make_rock
from pymrst.utils.adi import init_variables_adi, value


def _grid():
    g = cart_grid([7, 6], [7.0, 6.0])
    coords = np.asarray(g.nodes.coords, dtype=float)
    x, y = coords.T.copy()
    coords[:, 0] += 0.2*np.sin(np.pi*x/7.0)*np.sin(np.pi*y/6.0)
    g.nodes.coords = coords
    return compute_geometry(g)


def _model(g, rock, gravity=9.81):
    return SimpleNamespace(
        G=g,
        rock=rock,
        operators=setup_operators_tpfa(g, rock),
        gravity_magnitude=gravity,
        gdz=np.zeros(np.count_nonzero(np.all(g.faces.neighbors >= 0, axis=1))),
    )


def main():
    g = _grid()
    rock = make_rock(
        g, np.tile([2.0, 0.3, 1.0], (g.cells.num, 1))*1e-13, 0.2
    )
    pressure = 3.0*g.cells.centroids[:, 0] + 2.0*g.cells.centroids[:, 1] + 5.0

    nt_model = _model(g, rock)
    setNTPFADiscretization(nt_model)
    nt_direct = NTPFA.from_grid(g, rock)
    nt_expected = nt_direct.unscaled_gradient(pressure)/nt_model.operators.T
    nt_install_error = float(np.max(np.abs(
        nt_model.operators.grad(pressure) - nt_expected
    )))
    gdz_expected = (
        nt_model.operators.grad(g.cells.centroids[:, 1])
        * nt_model.gravity_magnitude
    )
    gdz_error = float(np.max(np.abs(nt_model.gdz - gdz_expected)))

    avg_model = _model(g, rock)
    setAvgMPFADiscretization(avg_model)
    avg_direct = AvgMPFA.from_grid(g, rock)
    avg_expected = avg_direct.unscaled_gradient(pressure)/avg_model.operators.T
    avg_install_error = float(np.max(np.abs(
        avg_model.operators.grad(pressure) - avg_expected
    )))

    target = _model(g, rock)
    internal_faces = np.flatnonzero(np.all(g.faces.neighbors >= 0, axis=1))
    choose_nt = np.asarray(g.faces.centroids)[:, 0] < 3.5
    setHybridDiscretization(
        target,
        [avg_model, nt_model],
        [np.ones(g.faces.num, dtype=bool), choose_nt],
    )
    hybrid = target.operators.grad(pressure)
    expected = avg_model.operators.grad(pressure).copy()
    mask = choose_nt[internal_faces]
    expected[mask] = nt_model.operators.grad(pressure)[mask]
    hybrid_error = float(np.max(np.abs(hybrid - expected)))

    # Snake-case aliases must install the same objects.
    alias_model = _model(g, rock)
    set_ntpfa_discretization(alias_model)
    alias_error = float(np.max(np.abs(
        alias_model.operators.grad(pressure) - nt_expected
    )))

    lagged_model = _model(g, rock)
    setNTPFADiscretization(lagged_model, lagged=True)
    (pressure_ad,) = init_variables_adi(pressure)
    lagged_ad = lagged_model.operators.grad(pressure_ad)
    lagged_ad_error = float(np.max(np.abs(
        value(lagged_ad) - lagged_model.operators.grad(pressure)
    )))
    avg_ad = avg_model.operators.grad(pressure_ad)
    avg_ad_error = float(np.max(np.abs(
        value(avg_ad) - avg_model.operators.grad(pressure)
    )))
    jacobian_ok = (
        lagged_ad.jac[0].shape == (internal_faces.size, g.cells.num)
        and avg_ad.jac[0].shape == (internal_faces.size, g.cells.num)
    )

    print("Phase 394: NFVM model integration")
    print(f"  NTPFA installation parity: {nt_install_error:.3e}")
    print(f"  AvgMPFA installation parity: {avg_install_error:.3e}")
    print(f"  nonlinear gravity-difference parity: {gdz_error:.3e}")
    print(f"  Hybrid face-block selection: {hybrid_error:.3e}")
    print(f"  snake/MRST setter parity: {alias_error:.3e}")
    print(f"  lagged NTPFA AD value parity: {lagged_ad_error:.3e}")
    print(f"  AvgMPFA AD value parity: {avg_ad_error:.3e}")
    print(f"  AD Jacobian dimensions: {jacobian_ok}")

    failed = max(
        nt_install_error,
        avg_install_error,
        gdz_error,
        hybrid_error,
        alias_error,
        lagged_ad_error,
        avg_ad_error,
    ) > 1e-10
    failed = failed or not jacobian_ok
    print("\nPhase 394 passed." if not failed else "\nPhase 394 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
