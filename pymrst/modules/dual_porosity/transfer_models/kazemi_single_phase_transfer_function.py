"""Single-phase Kazemi transfer (MRST ``KazemiSinglePhaseTransferFunction.m``)."""

from dataclasses import dataclass

from pymrst.modules.ad_props.simple_adi_fluid import ADFluid
from pymrst.modules.dual_porosity.shape_factors import ShapeFactor
from pymrst.params.rock import Rock


@dataclass
class KazemiSinglePhaseTransfer:
    """Single-phase matrix-fracture transfer."""

    shape_factor: ShapeFactor

    def transfer(self, rock_matrix: Rock, fluid_matrix: ADFluid, p_fracture, p_matrix):
        """T_w per bulk volume, positive when the fracture feeds the matrix."""
        sigma = self.shape_factor.sigma(rock_matrix)
        b_wm = fluid_matrix.b_w(p_matrix)
        mu_wm = fluid_matrix.mu[0]
        return (sigma * b_wm / mu_wm) * (p_fracture - p_matrix)


__all__ = ["KazemiSinglePhaseTransfer"]
