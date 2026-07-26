"""Arithmetic MPFA/NFVM blend, corresponding to MRST ``AvgMPFA.m``."""

from dataclasses import dataclass

from pymrst.modules.nfvm.solvers.ntpfa import NTPFA
from pymrst.modules.nfvm.solvers.utils.avg_trans_ntpfa import avg_trans_ntpfa
from pymrst.utils.adi import ad_matmul


@dataclass
class AvgMPFA(NTPFA):
    """Fixed arithmetic average of the two one-sided consistent fluxes."""

    T: tuple = ()

    @classmethod
    def from_grid(
        cls, g, rock, *, interp_face=None, os_flux=None, ratio=None, lagged=False
    ):
        base = NTPFA.from_grid(
            g,
            rock,
            interp_face=interp_face,
            os_flux=os_flux,
            ratio=ratio,
            lagged=lagged,
        )
        return cls(
            base.G,
            base.interpFace,
            base.OSflux,
            base.lagged,
            avg_trans_ntpfa(g, base.OSflux),
        )

    def unscaled_gradient(self, pressure):
        return 0.5*(
            ad_matmul(self.T[0], pressure) - ad_matmul(self.T[1], pressure)
        )
