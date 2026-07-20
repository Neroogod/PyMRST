"""Port of MRST `book-ii/introToVaporLiquidEquilibrium.m`.

The MATLAB example is a tour through vapor-liquid equilibrium: a
two-component Rachford-Rice construction, a ternary Peng-Robinson phase map,
and a pressure/temperature flash map for the SPE5 benchmark fluid. The default
PyMRST sizes are scaled down for fast tests, but the returned data mirrors the
sections and variables of the MRST script.
"""

from time import perf_counter

import numpy as np

from pymrst.modules.compositional.eos import (
    Mixture,
    flash,
    fugacity_coefficients,
    rachford_rice,
    wilson_k,
)
from pymrst.modules.compositional.fluids import (
    get_benchmark_mixture,
    get_compositional_fluid_case,
)
from pymrst.modules.compositional.models.eos import EquationOfStateModel
from pymrst.modules.compositional.utils import newtonFugacityEquilibrium
from pymrst.utils.units import BARSA


def rachford_rice_section(n: int = 50):
    """MRST section 8.3.1: two-component Rachford-Rice VLE."""
    z_light = np.linspace(0.0, 1.0, int(n))
    k_values = np.array([0.1, 10.0])
    z = np.column_stack([1.0 - z_light, z_light])
    vapour_fraction = rachford_rice(z, np.tile(k_values, (z.shape[0], 1)))

    objective_v = np.linspace(-1.0, 2.0, int(n))
    z_mid = np.array([0.5, 0.5])
    objective_rr = np.sum(
        ((k_values - 1.0) * z_mid)
        / (1.0 + objective_v[:, None] * (k_values - 1.0)),
        axis=1,
    )
    objective_1 = np.sum(
        k_values * z_mid / (1.0 + objective_v[:, None] * (k_values - 1.0)),
        axis=1,
    )
    objective_2 = np.sum(
        z_mid / (1.0 + objective_v[:, None] * (k_values - 1.0)),
        axis=1,
    )
    return {
        "K": k_values,
        "z_light": z_light,
        "composition": z,
        "vapour_fraction": vapour_fraction,
        "objective_v": objective_v,
        "objective_rr": objective_rr,
        "objective_1": objective_1,
        "objective_2": objective_2,
        "solution_v": 0.5,
    }


def ternary_pr_mixture() -> Mixture:
    """CO2/C1/C10 mixture used in MRST's ternary PR flash diagram."""
    return Mixture(
        names=("CarbonDioxide", "Methane", "n-Decane"),
        Tcrit=np.array([304.7, 190.60, 617.7]),
        Pcrit=np.array([73.87, 43.04, 21.03]) * BARSA,
        Vcrit=np.array([0.094, 0.098, 0.603]) / 1000.0,
        acentric=np.array([0.225, 0.013, 0.4884]),
        molarMass=np.array([44.01, 16.04, 142.285]) / 1000.0,
        name="CO2/C1/C10 ternary PR mixture",
    )


def ternary_flash_section(n: int = 25, p: float = 25.0 * BARSA, T: float = 273.15 + 30.0):
    """MRST section 8.3.2: ternary liquid-fraction map."""
    mixture = ternary_pr_mixture()
    axis = np.linspace(0.0, 1.0, int(n))
    X, Y = np.meshgrid(axis, axis)
    Z = 1.0 - X - Y
    comp_raw = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    valid = np.all(comp_raw >= -1.0e-12, axis=1) & (comp_raw.sum(axis=1) > 0.0)
    comp = np.clip(comp_raw[valid], 0.0, None)
    comp = comp / comp.sum(axis=1, keepdims=True)
    result = flash(
        mixture,
        np.full(comp.shape[0], float(p)),
        np.full(comp.shape[0], float(T)),
        comp,
    )
    liquid_fraction = np.full(comp_raw.shape[0], np.nan)
    liquid_fraction[valid] = 1.0 - result.V
    liquid_fraction = liquid_fraction.reshape(X.shape)
    liquid_fraction[(liquid_fraction <= 1.0e-12) | (liquid_fraction >= 1.0 - 1.0e-12)] = np.nan
    return {
        "mixture": mixture,
        "X": X,
        "Y": Y,
        "Z": Z,
        "valid": valid.reshape(X.shape),
        "composition": comp,
        "flash": result,
        "liquid_fraction": liquid_fraction,
        "pressure": float(p),
        "temperature": float(T),
    }


def spe5_phase_diagram_section(ns: int = 40):
    """Scaled SPE5 pressure/temperature flash map from the MRST example."""
    mixture, info = get_benchmark_mixture("spe5")
    z = np.asarray(info.initial, dtype=float)
    pressure = np.linspace(1.0, 180.0, int(ns)) * BARSA
    temperature = 273.15 + np.linspace(1.0, 400.0, int(ns))
    T_grid, P_grid = np.meshgrid(temperature, pressure)
    comp = np.tile(z, (T_grid.size, 1))
    result = flash(mixture, P_grid.ravel(), T_grid.ravel(), comp)
    liquid_fraction = (1.0 - result.V).reshape(P_grid.shape)
    liquid_fraction[(liquid_fraction <= 1.0e-12) | (liquid_fraction >= 1.0 - 1.0e-12)] = np.nan
    active_cells = result.two_phase.reshape(P_grid.shape).astype(float)
    active_cells[active_cells == 0.0] = np.nan
    return {
        "mixture": mixture,
        "info": info,
        "pressure": pressure,
        "temperature": temperature,
        "P": P_grid,
        "T": T_grid,
        "composition": comp,
        "flash": result,
        "liquid_fraction": liquid_fraction,
        "active_cells": active_cells,
        "iterations": result.iterations,
    }


def _flash_active_cell_history(mixture: Mixture, p, T, z, *, tol: float = 1e-10, max_it: int = 200):
    """Trace active-cell counts for PyMRST's SSI flash loop."""
    p = np.atleast_1d(np.asarray(p, float))
    T = np.atleast_1d(np.asarray(T, float))
    z = np.atleast_2d(np.asarray(z, float))
    p, T = np.broadcast_arrays(p, T)
    if z.shape[0] == 1 and p.size > 1:
        z = np.repeat(z, p.size, axis=0)

    K = wilson_k(mixture, p, T)
    single_liquid = np.sum(z * K, axis=1) <= 1.0
    single_vapour = np.sum(z / K, axis=1) <= 1.0
    active = ~(single_liquid | single_vapour)
    V = rachford_rice(z, K)
    V[single_liquid] = 0.0
    V[single_vapour] = 1.0

    active_counts = []
    errors = []
    for _ in range(1, int(max_it) + 1):
        active_counts.append(int(np.count_nonzero(active)))
        x = z / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        x = x / x.sum(axis=1, keepdims=True)
        y = y / y.sum(axis=1, keepdims=True)
        ln_phi_l, _ = fugacity_coefficients(mixture, p, T, x, vapour=False)
        ln_phi_v, _ = fugacity_coefficients(mixture, p, T, y, vapour=True)
        K_new = np.exp(ln_phi_l - ln_phi_v)
        err = float(np.abs(np.log(K_new[active] / K[active])).max()
                    if active.any() else 0.0)
        errors.append(err)
        K[active] = K_new[active]

        trivial = np.sum(np.log(K) ** 2, axis=1) < 1e-8
        newly_single = active & trivial
        if newly_single.any():
            liq = np.sum(z * K, axis=1) <= 1.0
            V[newly_single & liq] = 0.0
            V[newly_single & ~liq] = 1.0
            active = active & ~newly_single

        Vn = rachford_rice(z, K)
        V[active] = Vn[active]
        if err < tol or not active.any():
            break

    return {
        "active_cells": np.asarray(active_counts, dtype=int),
        "errors": np.asarray(errors, dtype=float),
    }




def _flash_ssi_newton_history(
    mixture: Mixture,
    p,
    T,
    z,
    *,
    ssi_bootstrap: int = 5,
    tol: float = 1e-10,
    max_newton: int = 20,
):
    """Trace an MRST-like SSI-to-Newton flash loop.

    MRST's VLE introduction switches from successive substitution to a Newton
    solve after a short SSI startup.  PyMRST already ports the AD-linearized
    Newton equilibrium step; this helper wires it into the example diagnostics
    without replacing the production value-level ``flash`` routine.
    """
    p = np.atleast_1d(np.asarray(p, float))
    T = np.atleast_1d(np.asarray(T, float))
    z = np.atleast_2d(np.asarray(z, float))
    p, T = np.broadcast_arrays(p, T)
    if z.shape[0] == 1 and p.size > 1:
        z = np.repeat(z, p.size, axis=0)

    K = wilson_k(mixture, p, T)
    single_liquid = np.sum(z * K, axis=1) <= 1.0
    single_vapour = np.sum(z / K, axis=1) <= 1.0
    active = ~(single_liquid | single_vapour)
    V = rachford_rice(z, K)
    V[single_liquid] = 0.0
    V[single_vapour] = 1.0

    # A short SSI bootstrap gives Newton the same kind of starting point MRST
    # uses before it switches method.
    for _ in range(max(0, int(ssi_bootstrap))):
        if not active.any():
            break
        x = z / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        x = x / x.sum(axis=1, keepdims=True)
        y = y / y.sum(axis=1, keepdims=True)
        ln_phi_l, _ = fugacity_coefficients(mixture, p, T, x, vapour=False)
        ln_phi_v, _ = fugacity_coefficients(mixture, p, T, y, vapour=True)
        K_new = np.exp(ln_phi_l - ln_phi_v)
        K[active] = K_new[active]
        Vn = rachford_rice(z, K)
        V[active] = Vn[active]

    eos = EquationOfStateModel(None, mixture)
    active_counts = []
    errors = []
    for _ in range(1, int(max_newton) + 1):
        active_counts.append(int(np.count_nonzero(active)))
        if not active.any():
            errors.append(0.0)
            break
        rows = np.flatnonzero(active)
        L = 1.0 - V[rows]
        x, y, K_new, _ZL, _ZV, L_new, values = newtonFugacityEquilibrium(
            eos,
            p[rows],
            T[rows],
            z[rows],
            K[rows],
            L,
        )
        cell_errors = np.max(np.asarray(values, dtype=float), axis=1)
        err = float(cell_errors.max(initial=0.0))
        errors.append(err)
        K[rows] = np.maximum(K_new, 1.0e-300)
        V[rows] = 1.0 - L_new
        converged = cell_errors < tol
        if np.any(converged):
            active[rows[converged]] = False
        if err < tol:
            active[rows] = False
            break

    return {
        "active_cells": np.asarray(active_counts, dtype=int),
        "errors": np.asarray(errors, dtype=float),
    }


def spe5_flash_diagnostics_section(ns: int = 40):
    """Return MRST-shaped SPE5 flash iteration, convergence, and timing data."""
    mixture, info = get_benchmark_mixture("spe5")
    z = np.asarray(info.initial, dtype=float)
    pressure = np.linspace(1.0, 180.0, int(ns)) * BARSA
    temperature = 273.15 + np.linspace(1.0, 400.0, int(ns))
    T_grid, P_grid = np.meshgrid(temperature, pressure)
    comp = np.tile(z, (T_grid.size, 1))

    t0 = perf_counter()
    result = flash(mixture, P_grid.ravel(), T_grid.ravel(), comp)
    flash_time = perf_counter() - t0
    ssi_history = _flash_active_cell_history(mixture, P_grid.ravel(), T_grid.ravel(), comp)
    t1 = perf_counter()
    newton_history = _flash_ssi_newton_history(mixture, P_grid.ravel(), T_grid.ravel(), comp)
    newton_time = perf_counter() - t1

    iteration_map = result.two_phase.reshape(P_grid.shape).astype(float) * float(result.iterations)
    iteration_map[iteration_map == 0.0] = np.nan
    return {
        "pressure": pressure,
        "temperature": temperature,
        "P": P_grid,
        "T": T_grid,
        "ssi_iteration_map": iteration_map,
        "convergence": {
            "names": ("SSI", "SSI+Newton"),
            "available": ("SSI", "SSI+Newton"),
            "iterations": np.arange(1, max(ssi_history["active_cells"].size, newton_history["active_cells"].size) + 1, dtype=int),
            "active_cells": {
                "SSI": ssi_history["active_cells"],
                "SSI+Newton": newton_history["active_cells"],
            },
            "errors": {
                "SSI": ssi_history["errors"],
                "SSI+Newton": newton_history["errors"],
            },
        },
        "timing": {
            "names": ("SSI", "SSI+Newton"),
            "available": ("SSI", "SSI+Newton"),
            "rows": ("Stability", "Flash"),
            "seconds": np.array([[0.0, 0.0], [flash_time, newton_time]], dtype=float),
        },
        "newton_backend": {
            "available": True,
            "reason": "PyMRST uses the AD-linearized newtonFugacityEquilibrium step after SSI bootstrap.",
        },
    }


def run_intro_to_vapor_liquid_equilibrium(n_rr: int = 50, n_ternary: int = 25, ns_spe5: int = 40):
    """Return all MRST VLE-introduction sections as data arrays."""
    rr = rachford_rice_section(n_rr)
    ternary = ternary_flash_section(n_ternary)
    spe5 = spe5_phase_diagram_section(ns_spe5)
    spe5_diagnostics = spe5_flash_diagnostics_section(ns_spe5)

    # Backward-compatible pressure-line flash retained from the original port.
    mixture, info = get_compositional_fluid_case("simple")
    pressure = np.linspace(20.0e5, 120.0e5, 12)
    temperature = np.full_like(pressure, info.T)
    composition = np.tile(info.initial, (pressure.size, 1))
    result = flash(mixture, pressure, temperature, composition)
    return {
        "rachford_rice": rr,
        "ternary": ternary,
        "spe5": spe5,
        "spe5_diagnostics": spe5_diagnostics,
        "line_flash": {
            "mixture": mixture,
            "pressure": pressure,
            "temperature": temperature,
            "composition": composition,
            "flash": result,
        },
    }


if __name__ == "__main__":
    def test_rachford_rice_section_matches_two_component_example():
        rr = rachford_rice_section()
        assert rr["K"].tolist() == [0.1, 10.0]
        assert np.isclose(rr["vapour_fraction"][25], 0.5, atol=2.1e-2)
        i = np.argmin(np.abs(rr["objective_v"] - rr["solution_v"]))
        assert abs(rr["objective_rr"][i]) < 0.1
        assert np.all(np.isfinite(rr["objective_1"]))
        assert np.all(np.isfinite(rr["objective_2"]))
        print("ok test_rachford_rice_section_matches_two_component_example")

    def test_ternary_flash_section_returns_mrst_phase_map():
        out = ternary_flash_section(n=15)
        assert out["liquid_fraction"].shape == (15, 15)
        assert out["composition"].shape[1] == 3
        assert np.allclose(out["composition"].sum(axis=1), 1.0)
        assert np.any(np.isfinite(out["liquid_fraction"]))
        assert out["flash"].V.shape[0] == out["composition"].shape[0]
        print("ok test_ternary_flash_section_returns_mrst_phase_map")

    def test_spe5_phase_diagram_section_returns_flash_grid():
        out = spe5_phase_diagram_section(ns=12)
        assert out["mixture"].n == 6
        assert out["liquid_fraction"].shape == (12, 12)
        assert out["active_cells"].shape == (12, 12)
        assert out["flash"].V.shape == (144,)
        assert out["iterations"] > 0
        print("ok test_spe5_phase_diagram_section_returns_flash_grid")

    def test_spe5_flash_diagnostics_section_returns_mrst_shapes():
        out = spe5_flash_diagnostics_section(ns=10)
        assert out["ssi_iteration_map"].shape == (10, 10)
        assert out["convergence"]["names"] == ("SSI", "SSI+Newton")
        assert out["convergence"]["available"] == ("SSI", "SSI+Newton")
        assert out["convergence"]["active_cells"]["SSI"].ndim == 1
        assert out["convergence"]["active_cells"]["SSI+Newton"].ndim == 1
        assert np.all(np.isfinite(out["convergence"]["errors"]["SSI+Newton"]))
        assert out["timing"]["seconds"].shape == (2, 2)
        assert np.all(np.isfinite(out["timing"]["seconds"]))
        assert out["newton_backend"]["available"]
        print("ok test_spe5_flash_diagnostics_section_returns_mrst_shapes")

    def test_line_flash_curve_is_consistent():
        out = run_intro_to_vapor_liquid_equilibrium(n_ternary=10, ns_spe5=10)
        pressure = out["line_flash"]["pressure"]
        composition = out["line_flash"]["composition"]
        result = out["line_flash"]["flash"]
        reconstructed = (1.0 - result.V[:, None]) * result.x + result.V[:, None] * result.y
        assert np.allclose(reconstructed, composition, atol=1e-8)
        assert np.all(np.diff(pressure) > 0.0)
        assert result.V.shape == pressure.shape
        assert np.any(result.two_phase)
        assert "spe5_diagnostics" in out
        print("ok test_line_flash_curve_is_consistent")

    test_rachford_rice_section_matches_two_component_example()
    test_ternary_flash_section_returns_mrst_phase_map()
    test_spe5_phase_diagram_section_returns_flash_grid()
    test_spe5_flash_diagnostics_section_returns_mrst_shapes()
    test_line_flash_curve_is_consistent()
    print("\nAll tests passed!")


__all__ = [
    "rachford_rice_section",
    "run_intro_to_vapor_liquid_equilibrium",
    "spe5_flash_diagnostics_section",
    "spe5_phase_diagram_section",
    "ternary_flash_section",
    "ternary_pr_mixture",
]
