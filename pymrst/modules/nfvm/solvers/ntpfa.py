"""NTPFA discretization object, corresponding to MRST ``NTPFA.m``."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.nfvm.solvers.utils.compute_flux import compute_flux
from pymrst.modules.nfvm.solvers.utils.correct_hap import correct_hap
from pymrst.modules.nfvm.solvers.utils.find_hap import find_hap
from pymrst.modules.nfvm.solvers.utils.find_os_flux import find_os_flux
from pymrst.modules.nfvm.solvers.utils.trans_ntpfa import trans_ntpfa
from pymrst.utils.adi import value


@dataclass
class NTPFA:
    """Nonlinear two-point flux approximation on a processed grid."""

    G: object
    interpFace: object
    OSflux: np.ndarray
    lagged: bool = False

    @classmethod
    def from_grid(
        cls, g, rock, *, interp_face=None, os_flux=None, ratio=None, lagged=False
    ):
        hap = find_hap(g, rock) if interp_face is None else interp_face
        hap = correct_hap(g, hap, ratio=ratio)
        stencils = find_os_flux(g, rock, hap) if os_flux is None else os_flux
        return cls(g, hap, stencils, lagged)

    def unscaled_gradient(self, pressure):
        is_ad = hasattr(pressure, "jac")
        weights_pressure = value(pressure) if self.lagged else pressure
        T = trans_ntpfa(self.G, weights_pressure, self.OSflux)
        if is_ad:
            internal = np.all(self.G.faces.neighbors >= 0, axis=1)
            c1, c2 = self.G.faces.neighbors[internal].T
            flux = T[0][internal]*pressure[c1] - T[1][internal]*pressure[c2]
            return -flux
        flux = compute_flux(self.G, pressure, T)
        return -flux[np.all(self.G.faces.neighbors >= 0, axis=1)]

    def gradient(self, pressure, scale=1.0):
        return self.unscaled_gradient(pressure)*scale
