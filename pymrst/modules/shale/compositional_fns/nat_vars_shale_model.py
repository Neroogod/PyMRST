"""Natural-variable shale model (`NatVarsShaleModel`)."""

import numpy as np

from pymrst.modules.compositional.natural_variables import NaturalVariablesCompositionalModel
from pymrst.modules.shale.data import ShaleRock
from pymrst.utils.units import DARCY, MILLI
from pymrst.modules.shale.compositional_fns.equations_shale_natural_vars import (
    add_sorbed_accumulation,
    apply_gangi_velocity,
    equations_shale_natural_vars,
    non_darcy_velocity,
    sorbed_accumulation,
)


class NatVarsShaleModel(NaturalVariablesCompositionalModel):
    """Natural-variable compositional model carrying shale mechanisms."""

    def __init__(
        self, *args, shale_rock: ShaleRock, grain_volume=None,
        diffusion=None, tortuosity=None, gangi_function=None,
        non_darcy_coefficient=None, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shale_rock = shale_rock
        self.shale_grain_volume = None if grain_volume is None else np.asarray(
            grain_volume, dtype=float
        ).reshape(-1)
        self.shale_diffusion = diffusion
        self.shale_tortuosity = tortuosity
        self.shale_gangi_function = gangi_function
        self.shale_non_darcy_coefficient = non_darcy_coefficient

    def _mechanism_active(self, *names):
        return any(name in self.shale_rock.shale_mechanisms for name in names)

    def _cell_values(self, values, name):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 1:
            return np.full(self.g.cells.num, array.item())
        if array.size != self.g.cells.num:
            raise ValueError(f"{name} must be scalar or have one value per cell")
        return array

    def _gangi_multiplier(self, pressure):
        function = self.shale_gangi_function
        if function is None:
            function = getattr(self, "KGangiFn", None)
        if function is None:
            raise ValueError("Gangi is active but no gangi_function was supplied")
        multiplier = function(pressure)
        is_matrix = np.asarray(self.shale_rock.is_matrix, dtype=bool)
        if is_matrix.size != self.g.cells.num:
            raise ValueError("is_matrix must have one value per global cell")
        # MRST applies Gangi only to matrix cells; fracture cells retain unity.
        out = pressure * 0.0 + multiplier
        out[~is_matrix] = 1.0
        return out

    def _non_darcy_factor(self, density, mobility):
        coefficient = self.shale_non_darcy_coefficient
        if coefficient is None:
            perm = np.asarray(self.shale_rock.perm, dtype=float)
            perm = perm if perm.ndim == 1 else perm[:, 0]
            perm_md = perm / (MILLI * DARCY)
            if np.any(perm_md <= 0):
                raise ValueError("non-Darcy flow requires positive permeability")
            coefficient = perm * (3.2808 * 1.485e9 / perm_md**1.021)
        coefficient = self._cell_values(coefficient, "non_darcy_coefficient")
        if self._mechanism_active("nonDarcyinFracs", "non_darcy_in_fractures"):
            coefficient = coefficient * (~np.asarray(self.shale_rock.is_matrix, dtype=bool))
        return density * mobility * coefficient

    def _modify_natural_velocity(
        self, phase, velocity, pressure, upwind, density, mobility
    ):
        if self._mechanism_active(
            "nonDarcyinFracs", "non_darcy_in_fractures",
            "nonDarcyEverywhere", "non_darcy_everywhere",
        ):
            cell_factor = self._non_darcy_factor(density, mobility)
            face_factor = self.ops.upw(upwind, cell_factor)
            # Preserve the signs used by the reference equation owner.
            positive_root = phase == "oil" and self._mechanism_active(
                "nonDarcyinFracs", "non_darcy_in_fractures"
            )
            velocity = non_darcy_velocity(
                velocity, face_factor, positive_root=positive_root
            )
        if self._mechanism_active("Gangi", "gangi"):
            cell_multiplier = self._gangi_multiplier(pressure)
            velocity = apply_gangi_velocity(
                velocity, self.ops.upw(upwind, cell_multiplier)
            )
        return velocity

    def _diffusion_parameters(self):
        diffusion = self.shale_diffusion
        if diffusion is None:
            diffusion = getattr(self.shale_rock, "Di", None)
        tortuosity = self.shale_tortuosity
        if tortuosity is None:
            tortuosity = getattr(self.shale_rock, "tau", None)
        if diffusion is None or tortuosity is None:
            raise ValueError(
                "diffusion is active but diffusion coefficients and tortuosity "
                "were not supplied"
            )
        diffusion = np.asarray(diffusion, dtype=float).reshape(-1)
        if diffusion.size != self.mix.n:
            raise ValueError("diffusion must have one coefficient per component")
        if float(tortuosity) <= 0:
            raise ValueError("tortuosity must be positive")
        return diffusion, float(tortuosity)

    def _modify_natural_component_flux(
        self, component, flux, vapor_mass_fractions, gas_density,
        gas_saturation, gas_upwind,
    ):
        if not self._mechanism_active("diffusion"):
            return flux
        diffusion, tortuosity = self._diffusion_parameters()
        porosity = np.asarray(self.shale_rock.poro, dtype=float)
        gas_capacity = self.ops.upw(
            gas_upwind, gas_saturation * gas_density * porosity
        )
        gradient = self.ops.grad(vapor_mass_fractions[component] * gas_density)
        return flux - (diffusion[component] / tortuosity) * gas_capacity * gradient

    def _grain_volume(self):
        grain_volume = self.shale_grain_volume
        if grain_volume is None:
            grain_volume = getattr(self.ops, "grain_volume", None)
        if grain_volume is None:
            raise ValueError(
                "sorption is active but no grain_volume was supplied to "
                "NatVarsShaleModel or its operators"
            )
        if np.asarray(grain_volume).size != self.g.cells.num:
            raise ValueError("grain_volume must have one value per global cell")
        return grain_volume

    def assemble_natural(
        self, st0, p, x_prim, s_o, y_prim, s_g, qo, qg, bhp, dt,
        s_w=None, qw=None, bc=None, src=None, flag=None,
    ):
        assembled = super().assemble_natural(
            st0, p, x_prim, s_o, y_prim, s_g, qo, qg, bhp, dt,
            s_w=s_w, qw=qw, bc=bc, src=src, flag=flag,
        )
        eqs, well_eqs, ctrl, fugacity_eqs, closure = assembled
        if "sorption" in self.shale_rock.shale_mechanisms:
            if not hasattr(self.mix, "isotherm"):
                raise TypeError(
                    "sorption requires a SorbedCompositionalFluid-compatible mixture"
                )
            flags = np.zeros(self.g.cells.num, dtype=np.int8) if flag is None else np.asarray(flag)
            pure_liquid = flags == 1
            pure_vapour = flags == 2
            two_phase = flags == 0
            _, y = self._component_list_natural(
                x_prim, y_prim, two_phase, pure_liquid, pure_vapour, self.mix.n
            )
            y0 = [st0.y[:, i] for i in range(self.mix.n)]
            correction = sorbed_accumulation(
                st0.pressure, y0, p, y, self.mix.isotherm,
                self._grain_volume(), dt,
            )
            eqs = add_sorbed_accumulation(eqs, correction, water=self.water)
        return eqs, well_eqs, ctrl, fugacity_eqs, closure

    def getEquations(self, state0, state, dt, drivingForces=None, *args, **kwargs):
        return equations_shale_natural_vars(
            state0, state, self, dt, drivingForces, *args, **kwargs
        )

    get_equations = getEquations


__all__ = ["NatVarsShaleModel"]
