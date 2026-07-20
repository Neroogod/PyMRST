"""Peng-Robinson equation of state and two-phase flash (MRST compositional).

Given an overall composition z at pressure p and temperature T, the flash
splits the mixture into a liquid (x) and a vapour (y) in equilibrium:

    K_i = y_i / x_i = phi_i^L / phi_i^V

Solved by successive substitution:

1. Wilson's correlation seeds the K-values
   (estimateEquilibriumWilson):
       K_i = exp(5.37 (1 + w_i)(1 - 1/Tr_i)) / Pr_i
2. Rachford-Rice gives the vapour fraction V for the current K.
3. The PR cubic

       Z^3 - (1-B) Z^2 + (A - 3B^2 - 2B) Z - (AB - B^2 - B^3) = 0

   is solved for each phase, taking the largest root for the vapour and
   the smallest for the liquid, and the fugacity coefficients follow.
4. K is updated from the fugacity ratio and the loop repeats.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

R_GAS = 8.3144598                # J / (mol K)
SQRT2 = np.sqrt(2.0)


# MRST's EquationOfStateModel carries omegaA/omegaB to seven digits rather
# than the 0.45724 / 0.07780 quoted in Peng & Robinson's paper. The covolume
# constant differs by 0.22%, which is enough to move a flash in the third
# decimal, so use MRST's values.
OMEGA_A = 0.4572355
OMEGA_B = 0.0779691


@dataclass
class Mixture:
    """A compositional mixture (CompositionalMixture)."""

    names: Tuple[str, ...]
    Tcrit: np.ndarray            # K
    Pcrit: np.ndarray            # Pa
    Vcrit: np.ndarray            # m3/mol (unused by PR, kept for parity)
    acentric: np.ndarray         # -
    molarMass: np.ndarray        # kg/mol
    bic: Optional[np.ndarray] = None      # binary interaction coefficients
    # "pr" is Peng-Robinson (1976); "prcorr" swaps in the 1978 polynomial for
    # m(w) on heavy components (w > 0.49). MRST calls these eosType 1 and 5.
    eos_type: str = "pr"
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.names)

    def bin_interaction(self) -> np.ndarray:
        return (np.zeros((self.n, self.n)) if self.bic is None
                else np.asarray(self.bic, float))

    @property
    def acentricFactors(self) -> np.ndarray:
        """MRST-compatible alias for the acentric-factor vector."""
        return self.acentric

    @acentricFactors.setter
    def acentricFactors(self, value: np.ndarray) -> None:
        arr = np.asarray(value, float)
        if arr.shape != np.asarray(self.acentric).shape:
            raise ValueError(
                f"acentricFactors must have shape {np.asarray(self.acentric).shape}, "
                f"got {arr.shape}"
            )
        self.acentric = arr

    def get_number_of_components(self) -> int:
        """Return the number of components (MRST `getNumberOfComponents`)."""
        return self.n

    def getNumberOfComponents(self) -> int:
        """MRST camelCase alias for :meth:`get_number_of_components`."""
        return self.get_number_of_components()

    def get_binary_interaction(self) -> np.ndarray:
        """Return binary interaction coefficients (MRST `getBinaryInteraction`)."""
        return self.bin_interaction()

    def getBinaryInteraction(self) -> np.ndarray:
        """MRST camelCase alias for :meth:`get_binary_interaction`."""
        return self.get_binary_interaction()

    def set_binary_interaction(self, input_bic: np.ndarray) -> "Mixture":
        """Set symmetric binary interaction coefficients.

        A vector input follows MRST's lower-triangular ordering.
        """
        bic = np.asarray(input_bic, float)
        if bic.ndim == 1:
            expected = self.n * (self.n - 1) // 2
            if bic.size != expected:
                raise ValueError(f"expected {expected} BIC entries, got {bic.size}")
            mat = np.zeros((self.n, self.n))
            offset = 0
            for i in range(1, self.n):
                count = i
                mat[i, :i] = bic[offset:offset + count]
                offset += count
            bic = mat + mat.T
        if bic.shape != (self.n, self.n):
            raise ValueError(f"BIC must be {(self.n, self.n)}, got {bic.shape}")
        if not np.allclose(bic, bic.T):
            raise ValueError("BIC matrix must be symmetric")
        self.bic = bic
        return self

    def setBinaryInteraction(self, input_bic: np.ndarray) -> "Mixture":
        """MRST camelCase alias for :meth:`set_binary_interaction`."""
        return self.set_binary_interaction(input_bic)


def _pr_m(w: np.ndarray, eos_type: str) -> np.ndarray:
    """The PR alpha-function slope m(w)."""
    m = 0.37464 + 1.54226 * w - 0.26992 * w**2
    if eos_type == "prcorr":
        heavy = w > 0.49
        m = np.where(heavy,
                     0.379642 + 1.48503 * w - 0.164423 * w**2
                     + 0.016666 * w**3,
                     m)
    return m


def _pr_ab(mix: Mixture, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-component PR attraction a_i(T) and covolume b_i."""
    m = _pr_m(mix.acentric, mix.eos_type)
    Tr = T[:, None] / mix.Tcrit[None, :]
    alpha = (1.0 + m[None, :] * (1.0 - np.sqrt(Tr))) ** 2
    ai = OMEGA_A * (R_GAS * mix.Tcrit) ** 2 / mix.Pcrit
    bi = OMEGA_B * R_GAS * mix.Tcrit / mix.Pcrit
    return ai[None, :] * alpha, np.tile(bi, (T.size, 1))


def _cubic_roots(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Real roots of the PR cubic, one row per cell (padded with NaN)."""
    c2 = -(1.0 - B)
    c1 = A - 3.0 * B**2 - 2.0 * B
    c0 = -(A * B - B**2 - B**3)
    out = np.full((A.size, 3), np.nan)
    for i in range(A.size):
        r = np.roots([1.0, c2[i], c1[i], c0[i]])
        r = r[np.abs(r.imag) < 1e-10].real
        out[i, :r.size] = np.sort(r)
    return out


def fugacity_coefficients(mix: Mixture, p: np.ndarray, T: np.ndarray,
                          x: np.ndarray, vapour: bool
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """ln(phi_i) for a phase of composition x, plus its compressibility Z."""
    n = mix.n
    ai, bi = _pr_ab(mix, T)
    kij = mix.bin_interaction()

    # a_ij = (1 - k_ij) sqrt(a_i a_j);  a_mix = sum_i sum_j x_i x_j a_ij
    sqrt_a = np.sqrt(ai)                                   # (nc, n)
    aij = (1.0 - kij)[None, :, :] * sqrt_a[:, :, None] * sqrt_a[:, None, :]
    a_mix = np.einsum("ci,cij,cj->c", x, aij, x)
    b_mix = np.einsum("ci,ci->c", x, bi)

    A = a_mix * p / (R_GAS * T) ** 2
    B = b_mix * p / (R_GAS * T)

    roots = _cubic_roots(A, B)
    Z = np.empty(p.size)
    for i in range(p.size):
        r = roots[i][np.isfinite(roots[i])]
        r = r[r > B[i]]
        if r.size == 0:
            r = roots[i][np.isfinite(roots[i])]
        Z[i] = r.max() if vapour else r.min()

    sum_xa = np.einsum("cij,cj->ci", aij, x)              # (nc, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
        ln_phi = (
            bi / b_mix[:, None] * (Z - 1.0)[:, None]
            - np.log(np.maximum(Z - B, 1e-300))[:, None]
            - (A / (2.0 * SQRT2 * B))[:, None]
            * (2.0 * sum_xa / a_mix[:, None] - bi / b_mix[:, None])
            * term[:, None]
        )
    return ln_phi, Z


def wilson_k(mix: Mixture, p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Wilson's K-value correlation (estimateEquilibriumWilson)."""
    Pr = p[:, None] / mix.Pcrit[None, :]
    Tr = T[:, None] / mix.Tcrit[None, :]
    k = np.exp(5.37 * (1.0 + mix.acentric[None, :]) * (1.0 - 1.0 / Tr)) / Pr
    k[~np.isfinite(k)] = 1e3
    return k


def rachford_rice(z: np.ndarray, K: np.ndarray, tol: float = 1e-12,
                  max_it: int = 100) -> np.ndarray:
    """Vapour molar fraction V solving sum_i z_i (K_i - 1)/(1 + V(K_i-1)) = 0."""
    nc = z.shape[0]
    V = np.zeros(nc)
    for c in range(nc):
        k, zz = K[c], z[c]
        if k.max() <= 1.0 or k.min() >= 1.0:      # single phase for this K
            V[c] = 1.0 if k.min() >= 1.0 else 0.0
            continue
        lo = 1.0 / (1.0 - k.max()) + 1e-12        # Rachford-Rice bracket
        hi = 1.0 / (1.0 - k.min()) - 1e-12
        lo, hi = max(lo, 0.0), min(hi, 1.0)
        f = lambda v: np.sum(zz * (k - 1.0) / (1.0 + v * (k - 1.0)))
        a, b = lo, hi
        fa, fb = f(a), f(b)
        if fa * fb > 0:                            # no root in (0, 1)
            V[c] = 1.0 if fa > 0 else 0.0
            continue
        for _ in range(max_it):                    # bisection: unconditionally
            m = 0.5 * (a + b)                      # stable for this function
            fm = f(m)
            if abs(fm) < tol or (b - a) < tol:
                break
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        V[c] = 0.5 * (a + b)
    return V


@dataclass
class FlashResult:
    """Two-phase equilibrium at (p, T, z)."""

    V: np.ndarray                # vapour molar fraction
    x: np.ndarray                # liquid composition
    y: np.ndarray                # vapour composition
    K: np.ndarray                # equilibrium ratios
    Z_L: np.ndarray
    Z_V: np.ndarray
    two_phase: np.ndarray        # bool: is the cell actually split?
    iterations: int


def flash(mix: Mixture, p: np.ndarray, T: np.ndarray, z: np.ndarray,
          tol: float = 1e-10, max_it: int = 200) -> FlashResult:
    """Two-phase PR flash by successive substitution."""
    p = np.atleast_1d(np.asarray(p, float))
    T = np.atleast_1d(np.asarray(T, float))
    z = np.atleast_2d(np.asarray(z, float))
    p, T = np.broadcast_arrays(p, T)
    if z.shape[0] == 1 and p.size > 1:
        z = np.repeat(z, p.size, axis=0)

    K = wilson_k(mix, p, T)

    # Phase check before flashing. The Rachford-Rice function at V = 0 and
    # V = 1 brackets the two-phase region: if sum(z K) <= 1 the mixture is
    # below its bubble point (all liquid); if sum(z / K) <= 1 it is above
    # its dew point (all vapour). Without this, successive substitution can
    # drift to the trivial solution K = 1 in the single-phase region.
    # (A rigorous tangent-plane stability test is stronger; this Wilson-based
    # check is the standard cheap screen and is exact away from the critical
    # region.)
    single_liquid = np.sum(z * K, axis=1) <= 1.0
    single_vapour = np.sum(z / K, axis=1) <= 1.0
    active = ~(single_liquid | single_vapour)

    V = rachford_rice(z, K)
    V[single_liquid] = 0.0
    V[single_vapour] = 1.0

    it = 0
    for it in range(1, max_it + 1):
        x = z / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        x = x / x.sum(axis=1, keepdims=True)
        y = y / y.sum(axis=1, keepdims=True)

        ln_phi_l, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
        ln_phi_v, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)

        K_new = np.exp(ln_phi_l - ln_phi_v)
        err = float(np.abs(np.log(K_new[active] / K[active])).max()
                    if active.any() else 0.0)
        K[active] = K_new[active]

        # a collapsing K is the trivial solution: the cell is single phase
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

    x = z / (1.0 + V[:, None] * (K - 1.0))
    y = K * x
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    ln_phi_l, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
    ln_phi_v, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)
    two_phase = (V > 1e-8) & (V < 1.0 - 1e-8)

    return FlashResult(V=V, x=x, y=y, K=K, Z_L=Z_L, Z_V=Z_V,
                       two_phase=two_phase, iterations=it)


def simple_mixture() -> Mixture:
    """A light three-component mixture (methane / n-butane / n-decane)."""
    return Mixture(
        names=("Methane", "n-Butane", "n-Decane"),
        Tcrit=np.array([190.564, 425.125, 617.7]),
        Pcrit=np.array([4599200.0, 3796000.0, 2103000.0]),
        Vcrit=np.array([9.86e-5, 2.55e-4, 6.098e-4]),
        acentric=np.array([0.0114, 0.201, 0.4884]),
        molarMass=np.array([0.016043, 0.058124, 0.142285]),
    )


if __name__ == "__main__":
    def test_wilson_k_monotone():
        """Wilson K falls with pressure and rises with temperature."""
        mix = simple_mixture()
        p = np.array([50e5, 150e5])
        T = np.array([300.0, 300.0])
        k = wilson_k(mix, p, T)
        assert np.all(k[1] < k[0]), k
        # methane (light) is far more volatile than decane (heavy)
        assert k[0, 0] > k[0, 1] > k[0, 2]
        print("ok test_wilson_k_monotone")

    def test_rachford_rice_root():
        """Rachford-Rice returns the root of the flash function."""
        z = np.array([[0.5, 0.3, 0.2]])
        K = np.array([[3.0, 0.8, 0.05]])
        V = rachford_rice(z, K)
        f = np.sum(z[0] * (K[0] - 1.0) / (1.0 + V[0] * (K[0] - 1.0)))
        assert abs(f) < 1e-9, f
        assert 0.0 < V[0] < 1.0
        print(f"ok test_rachford_rice_root (V = {V[0]:.4f})")

    def test_flash_two_phase_and_equilibrium():
        """A flashed mixture satisfies the equal-fugacity condition and
        conserves moles."""
        mix = simple_mixture()
        z = np.array([[0.5, 0.3, 0.2]])
        p = np.array([50e5])
        T = np.array([350.0])
        r = flash(mix, p, T, z)
        assert r.two_phase[0], r.V
        # material balance: z = (1-V) x + V y
        recon = (1 - r.V[:, None]) * r.x + r.V[:, None] * r.y
        assert np.allclose(recon, z, atol=1e-9), recon
        # equal fugacity: f_i^L = f_i^V
        lnl, _ = fugacity_coefficients(mix, p, T, r.x, vapour=False)
        lnv, _ = fugacity_coefficients(mix, p, T, r.y, vapour=True)
        fl = np.log(r.x) + lnl
        fv = np.log(r.y) + lnv
        assert np.abs(fl - fv).max() < 1e-8, np.abs(fl - fv).max()
        # vapour is enriched in the light component
        assert r.y[0, 0] > z[0, 0] > r.x[0, 0]
        assert r.Z_V[0] > r.Z_L[0]
        print(f"ok test_flash_two_phase_and_equilibrium "
              f"(V = {r.V[0]:.4f}, {r.iterations} iterations)")

    def test_flash_single_phase_limits():
        """At high pressure the mixture is all liquid; at low pressure, all
        vapour."""
        mix = simple_mixture()
        z = np.array([[0.5, 0.3, 0.2]])
        r_hi = flash(mix, np.array([600e5]), np.array([300.0]), z)
        assert not r_hi.two_phase[0] and r_hi.V[0] < 1e-6, r_hi.V
        r_lo = flash(mix, np.array([1e5]), np.array([500.0]), z)
        assert not r_lo.two_phase[0] and r_lo.V[0] > 1 - 1e-6, r_lo.V
        print("ok test_flash_single_phase_limits")

    def test_flash_vectorized():
        """Many cells flash at once."""
        mix = simple_mixture()
        n = 12
        p = np.linspace(20e5, 120e5, n)
        T = np.full(n, 350.0)
        z = np.tile([0.5, 0.3, 0.2], (n, 1))
        r = flash(mix, p, T, z)
        assert r.V.shape == (n,)
        assert np.all(np.diff(r.V) <= 1e-9), "vapour fraction must fall with p"
        recon = (1 - r.V[:, None]) * r.x + r.V[:, None] * r.y
        assert np.allclose(recon, z, atol=1e-8)
        print("ok test_flash_vectorized")

    test_wilson_k_monotone()
    test_rachford_rice_root()
    test_flash_two_phase_and_equilibrium()
    test_flash_single_phase_limits()
    test_flash_vectorized()
    print("\nAll tests passed!")
