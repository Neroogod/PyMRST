"""Two-phase Kazemi transfer (MRST ``KazemiTwoPhaseTransferFunction.m``)."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from pymrst.modules.ad_props.init_simple_adi_fluid import ADFluid
from pymrst.modules.dual_porosity.shape_factors import ShapeFactor
from pymrst.modules.dual_porosity.transfer import _upstream
from pymrst.params.rock import Rock
from pymrst.utils.adi import value


@dataclass
class KazemiTwoPhaseTransfer:
    """Two-phase oil-water matrix-fracture transfer."""

    shape_factor: ShapeFactor
    pc_ow_fracture: Optional[Callable] = None
    pc_ow_matrix: Optional[Callable] = None

    def transfer(
        self,
        rock_matrix: Rock,
        fluid_fracture: ADFluid,
        fluid_matrix: ADFluid,
        po_fracture,
        sw_fracture,
        po_matrix,
        sw_matrix,
    ) -> Tuple:
        """Return ``(T_w, T_o)`` per bulk volume, positive fracture -> matrix."""
        so_f = 1.0 - sw_fracture
        so_m = 1.0 - sw_matrix

        pc_f = 0.0 if self.pc_ow_fracture is None else self.pc_ow_fracture(sw_fracture)
        pc_m = 0.0 if self.pc_ow_matrix is None else self.pc_ow_matrix(sw_matrix)
        pw_fracture = po_fracture - pc_f
        pw_matrix = po_matrix - pc_m

        kr_w_f = fluid_fracture.kr_w(sw_fracture)
        kr_o_f = fluid_fracture.kr_o(so_f)
        kr_w_m = fluid_matrix.kr_w(sw_matrix)
        kr_o_m = fluid_matrix.kr_o(so_m)

        w_from_fracture = value(pw_matrix - pw_fracture) <= 0
        o_from_fracture = value(po_matrix - po_fracture) <= 0
        kr_wt = _upstream(w_from_fracture, kr_w_f, kr_w_m)
        kr_ot = _upstream(o_from_fracture, kr_o_f, kr_o_m)

        mu_wm, mu_om = fluid_matrix.mu[0], fluid_matrix.mu[1]
        b_wm = fluid_fracture.b_w(pw_matrix)
        b_om = fluid_fracture.b_o(po_matrix)

        sigma = self.shape_factor.sigma(rock_matrix)
        t_w = (sigma * b_wm * kr_wt / mu_wm) * (pw_fracture - pw_matrix)
        t_o = (sigma * b_om * kr_ot / mu_om) * (po_fracture - po_matrix)
        return t_w, t_o


__all__ = ["KazemiTwoPhaseTransfer"]
