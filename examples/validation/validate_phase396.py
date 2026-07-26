"""Phase 396: fv-unsat hydraulic catalog and van Genuchten-Mualem curves."""

import numpy as np

from pymrst.modules.fv_unsat import (
    getHydraulicProperties,
    vGM_saturation,
    vGM_theta,
)


def _derivative(function, points, step):
    return (function(points + step) - function(points - step))/(2.0*step)


def main():
    sand = getHydraulicProperties("sand")
    catalog_error = max(
        abs(sand.theta_s - 0.43),
        abs(sand.theta_r - 0.045),
        abs(sand.K_s - 29.70*0.01/3600.0),
        abs(sand.alpha - 14.5),
        abs(sand.n - 2.68),
    )

    phys_theta = {
        "flow": {
            "alpha": sand.alpha,
            "theta_r": sand.theta_r,
            "theta_s": sand.theta_s,
            "n": sand.n,
            "m": sand.m,
        }
    }
    theta, krw, capacity = vGM_theta(phys_theta)
    psi = -np.geomspace(1e-3, 10.0, 80)
    capacity_fd = _derivative(theta, psi, 1e-7)
    capacity_error = float(np.max(np.abs(capacity(psi) - capacity_fd)))
    endpoint_error = max(
        abs(float(theta(np.array([1.0]))[0]) - sand.theta_s),
        abs(float(krw(np.array([1.0]))[0]) - 1.0),
        abs(float(capacity(np.array([1.0]))[0])),
    )
    theta_monotone = bool(np.all(np.diff(theta(np.sort(psi))) >= 0))
    bounds_ok = bool(
        np.all((theta(psi) >= sand.theta_r) & (theta(psi) <= sand.theta_s))
        and np.all((krw(psi) >= 0) & (krw(psi) <= 1))
    )

    rho, gravity, poro = 1000.0, 9.80665, sand.theta_s
    phys_sat = {
        "flow": {
            "alpha": sand.alpha,
            "rho": rho,
            "g": gravity,
            "poro": poro,
            "theta_r": sand.theta_r,
            "n": sand.n,
            "m": sand.m,
        }
    }
    saturation, krw_p, saturation_capacity = vGM_saturation(phys_sat)
    pressure = psi*rho*gravity
    sat_capacity_fd = _derivative(saturation, pressure, 1e-3)
    sat_capacity_error = float(np.max(np.abs(
        saturation_capacity(pressure) - sat_capacity_fd
    )))
    curve_parity = max(
        float(np.max(np.abs(
            saturation(pressure) - theta(psi)/poro
        ))),
        float(np.max(np.abs(krw_p(pressure) - krw(psi)))),
    )

    print("Phase 396: fv-unsat constitutive laws")
    print(f"  sand catalog/SI conversion error: {catalog_error:.3e}")
    print(f"  theta capacity finite-difference error: {capacity_error:.3e}")
    print(f"  saturation capacity FD error: {sat_capacity_error:.3e}")
    print(f"  theta/saturation formulation parity: {curve_parity:.3e}")
    print(f"  saturated endpoint error: {endpoint_error:.3e}")
    print(f"  monotonic and bounded: {theta_monotone and bounds_ok}")

    failed = (
        catalog_error > 1e-14
        or capacity_error > 1e-7
        or sat_capacity_error > 1e-10
        or curve_parity > 1e-12
        or endpoint_error > 1e-14
        or not theta_monotone
        or not bounds_ok
    )
    print("\nPhase 396 passed." if not failed else "\nPhase 396 FAILED.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
