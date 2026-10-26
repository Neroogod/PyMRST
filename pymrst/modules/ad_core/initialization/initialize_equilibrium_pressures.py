"""MRST ``ad-core/initialization/initializeEquilibriumPressures.m``.

Hydrostatic phase pressures in a region's cells: the reference phase's
pressure is integrated from the datum (`dp/dz = g*rho(p, z)`, MATLAB's
`ode45` with AbsTol 1e-10 and RelTol 5e-8, up and down over the cells and
contacts), and each other phase's from its contact, where it equals the
reference pressure plus the signed capillary pressure at the contact. The
pressures are read off the solutions with `deval`. A zero gravity warns.

`initialize_equilibrium_pressures_ad` (in `equilibrium.py`) is PyMRST's
earlier version, which used SciPy's RK45.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np

from pymrst.utils.ode45 import deval, ode45


def initialize_equilibrium_pressures(model: Any, region: Any) -> np.ndarray:
    """MRST `p = initializeEquilibriumPressures(model, region)`: an `(nc, nph)`
    array of phase pressures."""
    nph = int(np.sum(model.get_active_phases()))
    cells = region.cells
    if isinstance(cells, str):
        cells = np.arange(model.G.cells.num)
    cells = np.asarray(cells, dtype=np.int64)
    nc = cells.size
    z = np.asarray(model.G.cells.centroids, dtype=float)[cells, 2]
    norm_g = float(np.linalg.norm(model.gravity))
    if norm_g == 0:
        warnings.warn("Gravity is zero. Initialization results may not be what you expect.")
    contacts = np.atleast_1d(np.asarray(region.contacts, dtype=float))
    zmax = max(np.max(z), np.max(contacts)) if contacts.size else np.max(z)
    zmin = min(np.min(z), np.min(contacts)) if contacts.size else np.min(z)
    p = np.zeros((nc, nph))
    ix = region.reference_index
    p[:, ix], po_fn = _solve_pressure(region.datum_pressure, region.datum_depth,
                                      region.rho[ix], norm_g, z, zmin, zmax)
    contact_no = 0
    for ix in range(nph):
        if ix == region.reference_index:
            continue
        contact = contacts[contact_no]
        p_ref = po_fn(contact) + region.pc_sign[ix]*np.ravel(region.contacts_pc)[contact_no]
        p[:, ix], _ = _solve_pressure(float(p_ref), contact, region.rho[ix], norm_g, z,
                                      zmin, zmax)
        contact_no += 1
    return p


def _integrate_pressure_drop(dp: Callable, P0: float, Z0: float, Z_min: float, Z_max: float):
    opts = dict(AbsTol=1.0e-10, RelTol=5.0e-8)
    sol_u = sol_d = None
    if not Z0 < Z_min and abs(Z_min - Z0) > 0:
        sol_u = ode45(dp, [Z0, Z_min], P0, **opts)
    if not Z0 > Z_max and abs(Z_max - Z0) > 0:
        sol_d = ode45(dp, [Z0, Z_max], P0, **opts)
    return sol_u, sol_d


def _solve_pressure(P0: float, Z0: float, rho: Callable, g: float, Z: np.ndarray,
                    Z_min: float, Z_max: float) -> Tuple[np.ndarray, Callable]:
    dp = lambda z, p: g*np.asarray(rho(p, z), dtype=float)
    sol_u, sol_d = _integrate_pressure_drop(dp, float(P0), float(Z0), Z_min, Z_max)
    p = _evaluate_pressure(Z, Z0, P0, sol_u, sol_d)
    return p, (lambda z: _evaluate_pressure(np.atleast_1d(z), Z0, P0, sol_u, sol_d)[0])


def _evaluate_pressure(Z: np.ndarray, Z0: float, P0: float, sol_u: Optional[Any],
                       sol_d: Optional[Any]) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    p = np.zeros(Z.shape)
    down = Z > Z0
    if np.any(down):
        p[down] = deval(sol_d, Z[down])[0]
    above = Z < Z0
    if np.any(above):
        p[above] = deval(sol_u, Z[above])[0]
    p[Z == Z0] = P0
    return p


initializeEquilibriumPressures = initialize_equilibrium_pressures

__all__ = ["initializeEquilibriumPressures", "initialize_equilibrium_pressures"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _model(z):
        return SimpleNamespace(gravity=np.array([0, 0, 9.81]),
                               get_active_phases=lambda: np.array([True, True, False]),
                               G=SimpleNamespace(cells=SimpleNamespace(
                                   num=z.size, centroids=np.column_stack([0*z, 0*z, z]))))

    def test_incompressible_columns_are_linear():
        z = np.array([1000.0, 1010.0, 1030.0])
        rho = [lambda p, z: 1000.0 + 0*p, lambda p, z: 800.0 + 0*p]
        region = SimpleNamespace(cells=np.arange(3), contacts=np.array([1020.0]),
                                 reference_index=1, datum_pressure=1e7, datum_depth=1000.0,
                                 rho=rho, pc_sign=np.array([-1.0, 1.0]),
                                 contacts_pc=np.array([0.0]))
        p = initialize_equilibrium_pressures(_model(z), region)
        np.testing.assert_allclose(p[:, 1], 1e7 + 9.81*800*(z - 1000), rtol=1e-12)
        po_contact = 1e7 + 9.81*800*20
        np.testing.assert_allclose(p[:, 0], po_contact + 9.81*1000*(z - 1020), rtol=1e-12)
        print("ok test_incompressible_columns_are_linear")

    test_incompressible_columns_are_linear()
    print("\nAll tests passed!")
