"""An opt-in EOS backend that flashes with Victory's network.

PyMRST already supports alternate EOS backends: every compositional call goes
through `compositional.backend`, which runs Peng--Robinson when `eos_model` is
`None` and delegates otherwise. That seam was built for the tabular K-value
backend; this reuses it. **Passing `eos_model=None` leaves today's behaviour
exactly as it is** - nothing here is imported, patched or consulted unless a
caller asks for it by name.

Two things about this backend are not optional to understand before using it.

**It cannot replace the stability analysis.** Victory's network takes `Gas` and
`2phase` as *inputs* - it is a conditional flash regressor that is told whether
the state splits. So this backend runs **PyMRST's own stability test** and hands
Victory the answer. The stability test is 36% of the flash cost on the bundled
10,000-state sample (0.129 s of 0.361 s), so the ceiling here is roughly a 2x
speed-up, not the 20x a raw Victory-vs-PyMRST timing suggests. Replacing
stability too would need a classifier that does not yet exist.

**It trades eight orders of physical consistency for that speed.** On the same
sample:

    material-balance RMSE    PyMRST 2.2e-11    Victory 6.4e-03
    log-fugacity RMSE        PyMRST 3.0e-10    Victory 3.5e+00

A simulator's Newton loop measures convergence with exactly those residuals, so
a flash that satisfies them to 1e-3 will not drive a tight nonlinear solve. This
is a benchmark and experiment vehicle, not a production flash.

**Domain**: the supplied models were trained for isothermal C1/C3/C14 at 353 K.
Outside that mixture or temperature the predictions are meaningless, and the
constructor refuses rather than extrapolating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from pymrst.modules.compositional.eos import FlashResult, fugacity_coefficients, wilson_k

TEMPERATURE_K = 353.0


@dataclass
class VictoryEOSModel:
    """Peng--Robinson stability, Victory flash, Peng--Robinson properties.

    Satisfies the backend contract in `compositional.backend`: `flash`,
    `property_model` and `evaluate_equilibrium_constants`.
    """

    mixture: Any
    ensemble: Any
    temperature: float = TEMPERATURE_K
    strict_domain: bool = True

    @classmethod
    def load(cls, model: str = "beta2.2", **kw) -> "VictoryEOSModel":
        """Build from the bundled folds; imports Torch only when called."""
        from pymrst.modules.compositional.ML_models.victory_flash import (
            VictoryFlashEnsemble,
            victory_mixture,
        )
        return cls(mixture=victory_mixture(),
                   ensemble=VictoryFlashEnsemble(model), **kw)

    # ------------------------------------------------------------------ flash

    def flash(self, p, T, z) -> FlashResult:
        """Stability by Peng--Robinson, split by Victory, Z by Peng--Robinson."""
        from pymrst.modules.compositional.utils.phase_stability_test import (
            phaseStabilityTest,
        )

        p = np.atleast_1d(np.asarray(p, float)).ravel()
        z = np.atleast_2d(np.asarray(z, float))
        T = np.broadcast_to(np.atleast_1d(np.asarray(T, float)).ravel(), p.shape)
        self._check_temperature(T)

        z = np.maximum(z, 1e-8)
        z /= z.sum(axis=1, keepdims=True)

        # Victory is told the phase state; it cannot determine it.
        stable, _, _ = phaseStabilityTest(self.mixture, z, p, T,
                                          K=wilson_k(self.mixture, p, T))
        two_phase = ~np.asarray(stable, dtype=bool)

        x = z.copy()
        y = z.copy()
        V = np.zeros(p.size)

        if np.any(two_phase):
            flags = np.zeros((int(two_phase.sum()), 2), dtype=np.float32)
            flags[:, 1] = 1.0                      # [Gas, 2phase] -> two-phase
            pred = self.ensemble.predict(
                p[two_phase], z[two_phase], phase_flags=flags,
                temperature=self.temperature, strict_domain=self.strict_domain)
            x[two_phase] = _rows_to_simplex(pred.x)
            y[two_phase] = _rows_to_simplex(pred.y)
            V[two_phase] = np.clip(1.0 - np.asarray(pred.L, float).ravel(),
                                   0.0, 1.0)

        # Compressibilities stay Peng--Robinson: the network predicts neither,
        # and the flow properties need them.
        _, z_l = fugacity_coefficients(self.mixture, p, T, x, vapour=False)
        _, z_v = fugacity_coefficients(self.mixture, p, T, y, vapour=True)
        K = np.divide(y, x, out=np.ones_like(x), where=x > 0.0)

        return FlashResult(V=V, x=x, y=y, K=K, Z_L=z_l, Z_V=z_v,
                           two_phase=two_phase, iterations=0)

    # ------------------------------------------------- rest of the contract

    @property
    def property_model(self):
        """Densities and viscosities, unchanged from Peng--Robinson."""
        return _PengRobinsonProperties(self.mixture, self.temperature)

    def evaluate_equilibrium_constants(self, p, T, z) -> np.ndarray:
        r = self.flash(p, T, z)
        return r.K

    def _check_temperature(self, T) -> None:
        if not self.strict_domain:
            return
        if not np.allclose(T, self.temperature, rtol=0, atol=1e-6):
            raise ValueError(
                f"the bundled Victory models are isothermal at "
                f"{self.temperature} K; got temperatures in "
                f"[{np.min(T)}, {np.max(T)}]. Pass strict_domain=False only if "
                f"you accept extrapolation with no validity claim.")


@dataclass
class _PengRobinsonProperties:
    """The property half of the backend contract, delegating to PyMRST."""

    mixture: Any
    temperature: float

    def _z(self, p, comp, is_liquid):
        p = np.asarray(p, float).ravel()
        T = np.full(p.shape, self.temperature)
        _, zf = fugacity_coefficients(self.mixture, p, T,
                                      _rows_to_simplex(comp),
                                      vapour=not is_liquid)
        return p, T, zf

    def compute_density(self, p, comp, is_liquid: bool):
        from pymrst.modules.compositional.properties import mass_density
        p, T, zf = self._z(p, comp, is_liquid)
        return mass_density(self.mixture, p, T, _rows_to_simplex(comp), zf)

    def compute_molar_density(self, p, comp, is_liquid: bool):
        from pymrst.modules.compositional.properties import molar_density
        p, T, zf = self._z(p, comp, is_liquid)
        return molar_density(self.mixture, p, T, _rows_to_simplex(comp), zf)

    def compute_viscosity(self, p, comp, is_liquid: bool):
        from pymrst.modules.compositional.properties import viscosity_lbc
        p, T, zf = self._z(p, comp, is_liquid)
        return viscosity_lbc(self.mixture, p, T, _rows_to_simplex(comp), zf)


def _rows_to_simplex(a) -> np.ndarray:
    a = np.maximum(np.atleast_2d(np.asarray(a, dtype=float)), 0.0)
    total = a.sum(axis=1, keepdims=True)
    return np.divide(a, total, out=np.full_like(a, 1.0 / a.shape[1]),
                     where=total > 0.0)


__all__ = ["VictoryEOSModel"]


if __name__ == "__main__":
    from pymrst.modules.compositional.backend import backend_flash, backend_flow_props
    from pymrst.modules.compositional.ML_models.victory_flash import victory_mixture

    MIX = victory_mixture()

    def sample(n=200, seed=0):
        rng = np.random.default_rng(seed)
        z = rng.dirichlet(np.ones(3), size=n)
        p = rng.uniform(2e6, 2e7, size=n)
        return p, z, np.full(n, TEMPERATURE_K)

    def model():
        return VictoryEOSModel.load("beta2.2")

    def test_the_default_path_is_untouched() -> None:
        """The whole point: `eos_model=None` must be today's behaviour."""
        from pymrst.modules.compositional.eos import flash
        p, z, T = sample()
        a = backend_flash(MIX, None, p, T, z)
        b = flash(MIX, p, T, z)
        assert np.allclose(a.V, b.V) and np.allclose(a.x, b.x)
        print("ok test_the_default_path_is_untouched")

    def test_it_satisfies_the_backend_contract() -> None:
        p, z, T = sample()
        fp = backend_flow_props(MIX, model(), p, T, z)
        for name in ("sL", "sV", "rhoL", "rhoV", "muL", "muV"):
            v = getattr(fp, name)
            assert np.all(np.isfinite(v)), name
        print("ok test_it_satisfies_the_backend_contract")

    def test_compositions_are_on_the_simplex() -> None:
        p, z, T = sample()
        r = model().flash(p, T, z)
        assert np.allclose(r.x.sum(axis=1), 1.0), "x must be mole fractions"
        assert np.allclose(r.y.sum(axis=1), 1.0), "y must be mole fractions"
        assert np.all((r.V >= 0.0) & (r.V <= 1.0))
        print("ok test_compositions_are_on_the_simplex")

    def test_single_phase_rows_are_left_alone() -> None:
        """Victory only ever sees the rows the stability test flags."""
        p, z, T = sample()
        r = model().flash(p, T, z)
        single = ~r.two_phase
        if single.any():
            assert np.allclose(r.x[single], r.y[single]), "a single phase splits"
            assert np.allclose(r.V[single], 0.0)
        print(f"ok test_single_phase_rows_are_left_alone ({int(single.sum())})")

    def test_the_phase_split_comes_from_pymrst_not_victory() -> None:
        """Stability is PyMRST's; Victory is told the answer."""
        from pymrst.modules.compositional.eos import flash
        p, z, T = sample()
        assert np.array_equal(model().flash(p, T, z).two_phase,
                              flash(MIX, p, T, z).two_phase)
        print("ok test_the_phase_split_comes_from_pymrst_not_victory")

    def test_the_accuracy_cost_is_real_and_bounded() -> None:
        """Stated rather than hidden: the split differs by ~0.07 in V."""
        from pymrst.modules.compositional.eos import flash
        p, z, T = sample(500, seed=3)
        gap = np.abs(model().flash(p, T, z).V - flash(MIX, p, T, z).V)
        assert gap.mean() > 1e-3, "a surrogate that matched exactly would be suspect"
        assert gap.mean() < 0.25, f"mean |dV| = {gap.mean():.3f}"
        print(f"ok test_the_accuracy_cost_is_real_and_bounded ({gap.mean():.3f})")

    def test_an_off_design_temperature_is_refused() -> None:
        """The models are isothermal at 353 K; extrapolation is not silent."""
        p, z, _ = sample(20)
        try:
            model().flash(p, np.full(p.shape, 400.0), z)
        except ValueError as error:
            assert "isothermal" in str(error)
            print("ok test_an_off_design_temperature_is_refused")
            return
        raise AssertionError("an off-design temperature must be refused")

    test_the_default_path_is_untouched()
    test_it_satisfies_the_backend_contract()
    test_compositions_are_on_the_simplex()
    test_single_phase_rows_are_left_alone()
    test_the_phase_split_comes_from_pymrst_not_victory()
    test_the_accuracy_cost_is_real_and_bounded()
    test_an_off_design_temperature_is_refused()
    print("\nAll tests passed!")
