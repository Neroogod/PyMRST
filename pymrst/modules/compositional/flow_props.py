"""Flash result -> flow properties (MRST `computeTwoPhaseFlowProps`).

`eos.py` answers *what phases exist and what is in them* (compositions and
compressibility factors); `properties.py` turns a phase into a density and a
viscosity. Neither is yet in the shape a flow model consumes. A compositional
conservation equation is written per **component**, and its accumulation term is

    pv/dt * ( rhoL sL xM_i - rhoL0 sL0 xM0_i + rhoV sV yM_i - rhoV0 sV0 yM0_i )

so before any flux or well is assembled the model needs, for every cell:

- **mass fractions** `xM_i`, `yM_i` - the mass (not mole) fraction of component
  `i` in the liquid and the vapour. The flash speaks in moles; the conservation
  equation is written in mass, so the conversion has to happen somewhere, and
  MRST does it here (`getMassFraction`).
- **phase saturations** `sL`, `sV` - the *volume* fractions of the two
  hydrocarbon phases. The flash gives a *molar* split `L`; turning that into a
  volume split is what `computeSaturations` does, and it is the one place the
  compressibility factors re-enter after the flash: with no volume shift the
  phase volumes are simply `L Z_L` and `(1-L) Z_V` (a molar volume is `Z R T/p`,
  and the `R T/p` cancels between the phases).
- **densities and viscosities** - already ported in Phase 50, called here so
  the whole `(p, T, z) -> flow props` pipeline lives in one place.

This module is the direct analogue of MRST's non-ADI
`computeTwoPhaseFlowProps`: the flash is evaluated on values, and the outer
forward-mode AD of the flow model differentiates the assembled residual. The
molar liquid fraction is `L = 1 - V` where `V` is `eos.flash`'s vapour fraction.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from pymrst.modules.compositional.eos import FlashResult, Mixture, flash
from pymrst.modules.compositional.properties import (
    mass_density,
    molar_density,
    viscosity_lbc,
)


def mass_fraction(mix: Mixture, comp: Any) -> Any:
    """Mole fractions -> mass fractions (MRST `getMassFraction`).

    `w_i = M_i x_i / sum_j M_j x_j`. Accepts an ``(n_cells, n_comp)`` array or a
    list of ``n_comp`` columns (AD-transparent), and returns the same shape.
    """
    mm = np.asarray(mix.molarMass, float)
    if isinstance(comp, (list, tuple)):
        mass = [comp[i] * mm[i] for i in range(len(comp))]
        total = mass[0]
        for i in range(1, len(mass)):
            total = total + mass[i]
        return [m / total for m in mass]
    comp = np.asarray(comp, float)
    mass = comp * mm[None, :]
    return mass / mass.sum(axis=1, keepdims=True)


def mole_fraction(mix: Mixture, comp: Any) -> Any:
    """Mass fractions -> mole fractions (MRST `getMoleFraction`).

    The inverse of :func:`mass_fraction`; used when a well or boundary condition
    is specified on a mass basis and the EOS needs moles.
    """
    mm = np.asarray(mix.molarMass, float)
    if isinstance(comp, (list, tuple)):
        moles = [comp[i] / mm[i] for i in range(len(comp))]
        total = moles[0]
        for i in range(1, len(moles)):
            total = total + moles[i]
        return [m / total for m in moles]
    comp = np.asarray(comp, float)
    moles = comp / mm[None, :]
    return moles / moles.sum(axis=1, keepdims=True)


def phase_saturations(mix: Mixture, p: Any, T: Any, L: Any,
                      x: Any, y: Any, Z_L: Any, Z_V: Any,
                      volume_shift: Optional[np.ndarray] = None
                      ) -> Tuple[Any, Any]:
    """Molar liquid fraction `L` -> phase (volume) saturations (MRST
    `computeSaturations`).

    A saturation is a *volume* fraction; `L` is a *molar* fraction. The phase
    volumes are the moles of each phase times its molar volume. With no volume
    shift the molar volume is `Z R T / p`, and since `R T / p` is common to both
    phases it cancels, leaving `volL = L Z_L`, `volV = (1-L) Z_V`. With a shift
    the molar volume is corrected, so the volumes go through the (shifted) molar
    density instead. Either way `sL = volL / (volL + volV)`.
    """
    if volume_shift is None:
        vol_l = L * Z_L
        vol_v = (1.0 - L) * Z_V
    else:
        vol_l = L / molar_density(mix, p, T, x, Z_L, volume_shift)
        vol_v = (1.0 - L) / molar_density(mix, p, T, y, Z_V, volume_shift)
    s_l = vol_l / (vol_l + vol_v)
    return s_l, 1.0 - s_l


@dataclass
class FlowProps:
    """Everything a compositional flow model reads out of a flash."""

    xM: Any          # liquid mass fractions, list of n_comp columns
    yM: Any          # vapour mass fractions
    sL: Any          # liquid saturation (volume fraction)
    sV: Any          # vapour saturation
    rhoL: Any        # liquid mass density [kg/m3]
    rhoV: Any        # vapour mass density
    muL: Any         # liquid viscosity [Pa s]
    muV: Any         # vapour viscosity
    x: np.ndarray    # liquid mole fractions (as flashed)
    y: np.ndarray    # vapour mole fractions
    L: np.ndarray    # molar liquid fraction
    Z_L: np.ndarray
    Z_V: np.ndarray


def flow_props_from_flash(mix: Mixture, p: Any, T: Any,
                          x: np.ndarray, y: np.ndarray, L: Any,
                          Z_L: Any, Z_V: Any,
                          volume_shift: Optional[np.ndarray] = None
                          ) -> FlowProps:
    """Assemble flow properties from an *already computed* flash.

    This is the exact analogue of the non-ADI branch of MRST's
    `computeTwoPhaseFlowProps`: it takes the flash state (`x, y, L, Z_L, Z_V`)
    and derives the mass fractions, saturations, densities and viscosities. It
    is the natural validation seam - feeding MRST's own flash output in isolates
    these formulas from any flash-solver difference.

    `x`, `y` are ``(n_cells, n_comp)`` mole-fraction arrays; the mass fractions
    are returned as a list of columns to match the per-component layout the flow
    model assembles over.
    """
    cols_x = [np.asarray(x, float)[:, i] for i in range(mix.n)]
    cols_y = [np.asarray(y, float)[:, i] for i in range(mix.n)]
    xM = mass_fraction(mix, cols_x)
    yM = mass_fraction(mix, cols_y)

    rho_l = mass_density(mix, p, T, x, Z_L, volume_shift)
    rho_v = mass_density(mix, p, T, y, Z_V, volume_shift)
    s_l, s_v = phase_saturations(mix, p, T, L, x, y, Z_L, Z_V, volume_shift)
    mu_l = viscosity_lbc(mix, p, T, x, Z_L, volume_shift)
    mu_v = viscosity_lbc(mix, p, T, y, Z_V, volume_shift)

    return FlowProps(xM=xM, yM=yM, sL=s_l, sV=s_v,
                     rhoL=rho_l, rhoV=rho_v, muL=mu_l, muV=mu_v,
                     x=np.asarray(x, float), y=np.asarray(y, float),
                     L=np.asarray(L, float),
                     Z_L=np.asarray(Z_L, float), Z_V=np.asarray(Z_V, float))


def two_phase_flow_props(mix: Mixture, p: Any, T: Any, z: np.ndarray,
                         volume_shift: Optional[np.ndarray] = None,
                         flash_result: Optional[FlashResult] = None
                         ) -> FlowProps:
    """`(p, T, z) -> flow properties` (MRST `computeTwoPhaseFlowProps`).

    Flashes the overall composition (unless a `flash_result` is supplied) and
    hands the result to :func:`flow_props_from_flash`. `L = 1 - V`, converting
    `eos.flash`'s vapour fraction to the liquid fraction MRST's saturation
    routine expects.
    """
    fr = flash_result if flash_result is not None else flash(mix, p, T, z)
    liquid_fraction = 1.0 - fr.V
    return flow_props_from_flash(mix, p, T, fr.x, fr.y, liquid_fraction,
                                 fr.Z_L, fr.Z_V, volume_shift)


if __name__ == "__main__":
    from pymrst.modules.compositional.eos import simple_mixture

    def test_mass_fraction_matches_hand_calc():
        """Mass fraction is `M_i x_i` normalised - check against arithmetic and
        confirm the array and column layouts agree."""
        mix = simple_mixture()
        x = np.array([[0.6, 0.3, 0.1]])
        want_num = x[0] * mix.molarMass
        want = want_num / want_num.sum()
        got_arr = mass_fraction(mix, x)
        assert np.allclose(got_arr[0], want, rtol=1e-14), (got_arr, want)
        got_cols = mass_fraction(mix, [x[:, i] for i in range(3)])
        for i in range(3):
            assert np.allclose(got_cols[i], want[i], rtol=1e-14)
        # a pure-methane cell is 100% methane by mass too
        pure = mass_fraction(mix, np.array([[1.0, 0.0, 0.0]]))
        assert np.allclose(pure[0], [1.0, 0.0, 0.0])
        print("ok test_mass_fraction_matches_hand_calc")

    def test_mass_and_mole_fraction_are_inverse():
        """Converting moles->mass->moles must return the original."""
        mix = simple_mixture()
        x = np.array([[0.5, 0.3, 0.2], [0.8, 0.15, 0.05]])
        back = mole_fraction(mix, mass_fraction(mix, x))
        assert np.allclose(back, x, rtol=1e-13), back
        # a heavy-shifted mixture weighs more per mole -> mass frac tilts heavy
        w = mass_fraction(mix, x)
        assert w[0, 2] > x[0, 2], "decane must gain weight fraction"
        print("ok test_mass_and_mole_fraction_are_inverse")

    def test_saturations_from_flash_are_volume_fractions():
        """With no volume shift `sL = L Z_L / (L Z_L + (1-L) Z_V)`, and the
        saturations partition unity."""
        mix = simple_mixture()
        p, T = np.array([50e5]), np.array([350.0])
        z = np.array([[0.5, 0.3, 0.2]])
        fr = flash(mix, p, T, z)
        assert bool(fr.two_phase[0])
        L = 1.0 - fr.V
        s_l, s_v = phase_saturations(mix, p, T, L, fr.x, fr.y, fr.Z_L, fr.Z_V)
        want = L * fr.Z_L / (L * fr.Z_L + (1.0 - L) * fr.Z_V)
        assert np.allclose(s_l, want, rtol=1e-14), (s_l, want)
        assert np.allclose(s_l + s_v, 1.0)
        # the denser liquid occupies less volume than its molar share
        assert s_l[0] < L[0], (s_l, L)
        print(f"ok test_saturations_from_flash_are_volume_fractions "
              f"(L = {float(L[0]):.3f}, sL = {float(s_l[0]):.3f})")

    def test_single_phase_saturation_limits():
        """An all-liquid cell has sL = 1; an all-vapour cell has sL = 0."""
        mix = simple_mixture()
        z = np.array([[0.5, 0.3, 0.2]])
        hi = two_phase_flow_props(mix, np.array([600e5]), np.array([300.0]), z)
        assert hi.sL[0] > 1.0 - 1e-6, hi.sL
        lo = two_phase_flow_props(mix, np.array([1e5]), np.array([500.0]), z)
        assert lo.sV[0] > 1.0 - 1e-6, lo.sV
        print("ok test_single_phase_saturation_limits")

    def test_full_pipeline_is_consistent():
        """The end-to-end props must hang together: mass fractions sum to one,
        saturations sum to one, and the liquid is heavier and thicker."""
        mix = simple_mixture()
        p, T = np.array([60e5]), np.array([380.0])
        z = np.array([[0.6, 0.25, 0.15]])
        fp = two_phase_flow_props(mix, p, T, z)
        sum_xM = fp.xM[0] + fp.xM[1] + fp.xM[2]
        sum_yM = fp.yM[0] + fp.yM[1] + fp.yM[2]
        assert np.allclose(sum_xM, 1.0) and np.allclose(sum_yM, 1.0)
        assert np.allclose(fp.sL + fp.sV, 1.0)
        assert fp.rhoL[0] > fp.rhoV[0] and fp.muL[0] > fp.muV[0]
        print("ok test_full_pipeline_is_consistent")

    def test_volume_shift_changes_saturation():
        """A volume shift corrects the *volumes*, so it must move the saturation
        split even though it leaves the molar flash untouched."""
        mix = simple_mixture()
        p, T = np.array([50e5]), np.array([350.0])
        z = np.array([[0.5, 0.3, 0.2]])
        fr = flash(mix, p, T, z)
        L = 1.0 - fr.V
        shift = np.array([-0.3, -0.2, -0.1])
        s0, _ = phase_saturations(mix, p, T, L, fr.x, fr.y, fr.Z_L, fr.Z_V)
        s1, _ = phase_saturations(mix, p, T, L, fr.x, fr.y, fr.Z_L, fr.Z_V,
                                  volume_shift=shift)
        assert not np.allclose(s0, s1), "the shift did nothing to saturation"
        print(f"ok test_volume_shift_changes_saturation "
              f"(sL {float(s0[0]):.4f} -> {float(s1[0]):.4f})")

    test_mass_fraction_matches_hand_calc()
    test_mass_and_mole_fraction_are_inverse()
    test_saturations_from_flash_are_volume_fractions()
    test_single_phase_saturation_limits()
    test_full_pipeline_is_consistent()
    test_volume_shift_changes_saturation()
    print("\nAll tests passed!")
