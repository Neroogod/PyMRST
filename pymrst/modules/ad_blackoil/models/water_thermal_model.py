"""MRST ``ad-blackoil/models/WaterThermalModel.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.geothermal.geothermal_model import (
    GeothermalModel,
    GeothermalState,
    ThermalFluid,
    ThermalRock,
)
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import BoundaryCondition


class WaterThermalModel(GeothermalModel):
    """Single-phase water with coupled pressure-temperature equations.

    MRST's class is an experimental thermal extension of ``WaterModel``. In
    PyMRST the thermal mass/energy discretization is shared with the geothermal
    module, while this class provides the ad-blackoil layout and naming.
    """

    thermal = True

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: ThermalFluid | None = None,
        thermal_rock: ThermalRock = ThermalRock(),
        bc: BoundaryCondition | None = None,
        bcT=None,
        tol: float = 1e-8,
        max_iterations: int = 30,
    ):
        super().__init__(
            g,
            rock,
            ThermalFluid() if fluid is None else fluid,
            thermal_rock=thermal_rock,
            bc=bc,
            bc_T=bcT,
            tol=tol,
            max_iterations=max_iterations,
        )
        self.G = self.g
        self.operators = self.ops

    def get_valid_driving_forces(self):
        """MRST-style driving force names."""
        return {"bc": None, "src": None, "W": None, "bcT": None}

    def get_equations(self, state0: GeothermalState, state: GeothermalState,
                      dt: float):
        from pymrst.modules.ad_blackoil.utils.equations_water_thermal import (
            equations_water_thermal,
        )

        equations, _ = equations_water_thermal(
            state0, state, self, dt, res_only=True,
        )
        return equations

    def get_surface_densities(self):
        return np.array([self.fluid.rho(np.array([0.0]), np.array([293.15]))[0]])


__all__ = ["WaterThermalModel", "GeothermalState", "ThermalFluid", "ThermalRock"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI

    def test_water_thermal_model_runs():
        g = compute_geometry(cart_grid([4, 1, 1], [20.0, 5.0, 5.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        model = WaterThermalModel(g, rock, max_iterations=12)
        st0 = model.initial_state(200 * BARSA, 330.0)
        st, ok, _ = model.step(st0, DAY)
        assert ok
        assert np.allclose(st.pressure, st0.pressure)
        assert np.allclose(st.T, st0.T)
        print("ok test_water_thermal_model_runs")

    test_water_thermal_model_runs()
