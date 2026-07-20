"""Rock thermal energy (MRST ``RockThermalEnergy.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalRock


def rock_thermal_energy(thermal_rock: ThermalRock, rock_volume, temperature):
    """Rock heat content ``V_r * rho_r * Cp_r * T``."""
    return rock_volume * thermal_rock.rho_r * thermal_rock.Cp_r * temperature


__all__ = ["ThermalRock", "rock_thermal_energy"]
