"""MICP fluid and reaction parameters (MRST ad-micp).

Microbially induced calcite precipitation after Landa-Marban et al.
(2021), Int. J. Greenh. Gas Control 106, 103256. Water is the only
mobile phase; five components are transported/reacted:

    m  suspended microbes      [kg/m^3 water]
    o  oxygen                  [kg/m^3 water]
    u  urea                    [kg/m^3 water]
    b  biofilm                 [volume fraction of bulk]
    c  calcite                 [volume fraction of bulk]

Porosity is reduced by biofilm and calcite, poro = poro_0 - b - c, and
permeability follows the power law of fluid.K with a critical porosity
below which it collapses to kmin.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class MICPFluid:
    """Fluid, transport, and bio-reaction parameters (defaults: 1D case)."""

    # Water
    muw: float = 2.535e-4        # viscosity, Pa s
    rhoWS: float = 1045.0        # density, kg/m^3

    # Phase densities of the solid products
    rho_b: float = 35.0          # biofilm, kg/m^3
    rho_c: float = 2710.0        # calcite, kg/m^3

    # Transport
    diffm: float = 2.1e-9        # diffusion, microbes, m^2/s
    diffo: float = 2.32e-9       # diffusion, oxygen, m^2/s
    diffu: float = 1.38e-9       # diffusion, urea, m^2/s
    alphaL: float = 1e-3         # longitudinal dispersivity, m
    alphaT: float = 4e-4         # transverse dispersivity, m

    # Bio-chemical rates
    k_str: float = 2.6e-10       # detachment rate, m/(Pa s)
    k_o: float = 2e-5            # half-velocity, oxygen, kg/m^3
    k_u: float = 21.3            # half-velocity, urea, kg/m^3
    mu: float = 4.17e-5          # max specific growth rate, 1/s
    mu_u: float = 0.0161         # max urease utilization rate, 1/s
    k_a: float = 8.51e-7         # microbial attachment rate, 1/s
    k_d: float = 3.18e-7         # microbial death rate, 1/s
    Y: float = 0.5               # yield growth coefficient, -
    Yuc: float = 1.67            # yield coefficient calcite/urea, -
    F: float = 0.5               # oxygen consumption factor, -

    # Clogging / permeability law
    eta: float = 3.0             # fitting exponent, -
    crit: float = 0.1            # critical porosity, -
    kmin: float = 1e-20          # minimum permeability, m^2
    ptol: float = 1e-4           # porosity tolerance (clogging stop)

    # Upper bounds used when clipping the Newton update
    Comax: float = 0.04          # max injected oxygen, kg/m^3
    Cumax: float = 60.0          # max injected urea, kg/m^3
    Cmmax: float = 105.0         # max microbial concentration, kg/m^3

    # Reference rock (set by the model)
    K0: Union[float, np.ndarray] = 1e-12    # initial permeability, m^2
    poro0: Union[float, np.ndarray] = 0.2   # initial porosity, -

    def permeability(self, poro: np.ndarray) -> np.ndarray:
        """K(poro) clogging law (fluid.K in the MRST examples)."""
        k0 = np.asarray(self.K0, dtype=np.float64)
        above = poro > self.crit
        num = k0 * ((poro - self.crit)
                    / (np.asarray(self.poro0) - self.crit)) ** self.eta
        k_above = (num + self.kmin) * k0 / (k0 + self.kmin)
        return np.where(above, k_above, self.kmin)


if __name__ == "__main__":
    def test_permeability_law():
        """K equals K0 at the initial porosity and collapses below crit."""
        f = MICPFluid(K0=1e-12, poro0=0.2)
        k = f.permeability(np.array([0.2]))
        assert np.isclose(k[0], 1e-12, rtol=1e-9)
        assert np.isclose(f.permeability(np.array([0.05]))[0], f.kmin)
        assert np.isclose(f.permeability(np.array([0.1]))[0], f.kmin)
        print("ok test_permeability_law")

    def test_permeability_monotone():
        """K decreases monotonically as pores clog."""
        f = MICPFluid(K0=1e-12, poro0=0.2)
        poro = np.linspace(0.101, 0.2, 50)
        k = f.permeability(poro)
        assert np.all(np.diff(k) > 0)          # increasing with porosity
        assert k[0] < 1e-14 and k[-1] <= 1e-12
        print("ok test_permeability_monotone")

    def test_permeability_heterogeneous_k0():
        """Per-cell K0 broadcasts correctly."""
        f = MICPFluid(K0=np.array([1e-12, 1e-13]), poro0=0.2)
        k = f.permeability(np.array([0.2, 0.2]))
        assert np.allclose(k, [1e-12, 1e-13], rtol=1e-9)
        print("ok test_permeability_heterogeneous_k0")

    test_permeability_law()
    test_permeability_monotone()
    test_permeability_heterogeneous_k0()
    print("\nAll tests passed!")
