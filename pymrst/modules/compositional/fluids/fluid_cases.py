"""Benchmark compositional mixtures (MRST `getBenchmarkMixture.m` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.eos import simple_mixture

R_GAS: float = 8.3144598
PSIA: float = 6894.757293168361
ATM: float = 101325.0
BARSA: float = 1.0e5
RANKINE_TO_KELVIN: float = 5.0 / 9.0


@dataclass
class FluidCaseInfo:
    """Initial conditions and injection stream for a benchmark mixture."""

    injection: np.ndarray
    initial: np.ndarray
    pressure: float
    T: float

    @property
    def temp(self) -> float:
        """MRST-compatible alias for temperature."""
        return self.T


def _case_info(
    injection: list[float],
    initial: list[float],
    pressure: float,
    temperature: float,
) -> FluidCaseInfo:
    return FluidCaseInfo(
        injection=np.asarray(injection, float),
        initial=np.asarray(initial, float),
        pressure=float(pressure),
        T=float(temperature),
    )


def _critical_volume(z_crit: np.ndarray, t_crit: np.ndarray, p_crit: np.ndarray) -> np.ndarray:
    """Critical molar volume from critical compressibility."""
    return z_crit * R_GAS * t_crit / p_crit


def _simple_case() -> tuple[Mixture, FluidCaseInfo]:
    names = ("Methane", "CarbonDioxide", "n-Decane")
    mix = Mixture(
        names=names,
        Tcrit=np.array([190.564, 304.2, 617.7]),
        Pcrit=np.array([4599200.0, 72.90 * ATM, 2103000.0]),
        Vcrit=np.array([9.86e-5, 9.41e-5, 6.098e-4]),
        acentric=np.array([0.0114, 0.228, 0.4884]),
        molarMass=np.array([0.016043, 0.04401, 0.142285]),
        name="Simple Methane/CO2/Decane mixture",
    )
    info = _case_info(
        injection=[0.1, 0.9, 0.0],
        initial=[0.3, 0.1, 0.6],
        pressure=75.0 * BARSA,
        temperature=273.15 + 150.0,
    )
    return mix, info


def _spe5_case() -> tuple[Mixture, FluidCaseInfo]:
    t_crit = np.array([343.0, 665.7, 913.4, 1111.8, 1270.0, 1380.0]) * RANKINE_TO_KELVIN
    p_crit = np.array([667.8, 616.3, 436.9, 304.0, 200.0, 162.0]) * PSIA
    z_crit = np.array([0.290, 0.277, 0.264, 0.257, 0.245, 0.235])
    mix = Mixture(
        names=("C1", "C3", "C6", "C10", "C15", "C20"),
        Tcrit=t_crit,
        Pcrit=p_crit,
        Vcrit=_critical_volume(z_crit, t_crit, p_crit),
        acentric=np.array([0.0130, 0.1524, 0.3007, 0.4885, 0.6500, 0.8500]),
        molarMass=np.array([16.040, 44.100, 86.180, 142.290, 206.000, 282.000]) / 1000.0,
        name="SPE5 benchmark",
    )
    bic = np.zeros((mix.n, mix.n))
    bic[4, 0] = 0.05
    bic[5, 0] = 0.05
    bic[4, 1] = 0.005
    bic[5, 1] = 0.005
    mix.set_binary_interaction(bic + bic.T)
    info = _case_info(
        injection=[0.77, 0.20, 0.03, 0.0, 0.0, 0.0],
        initial=[0.5, 0.03, 0.07, 0.20, 0.15, 0.05],
        pressure=4000.0 * PSIA,
        temperature=344.26,
    )
    return mix, info


def _lumped_1_case() -> tuple[Mixture, FluidCaseInfo]:
    t_crit = np.array([189.515, 304.200, 387.607, 597.497, 698.515, 875.00])
    p_crit = np.array([45.2012, 72.90, 40.4196, 33.0150, 17.4525, 11.5372]) * ATM
    z_crit = np.array([0.28981, 0.27055, 0.27588, 0.25668, 0.21967, 0.18250])
    mix = Mixture(
        names=("N2/CH4", "CO2", "C2-5", "C6-13", "C14-24", "C25-80"),
        Tcrit=t_crit,
        Pcrit=p_crit,
        Vcrit=_critical_volume(z_crit, t_crit, p_crit),
        acentric=np.array([0.00854, 0.228, 0.16733, 0.38609, 0.80784, 1.23141]),
        molarMass=np.array([16.1594, 44.01, 45.5725, 117.740, 248.827, 481.520]) / 1000.0,
        name="Lumped mixture, SPE 79691, example 5, Mallison et al.",
    )
    lower = np.array([
        [0.11883, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.00070981, 0.15, 0.0, 0.0, 0.0, 0.0],
        [0.00077754, 0.15, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.15, 0.0, 0.0, 0.0, 0.0],
        [0.011, 0.15, 0.0, 0.0, 0.0, 0.0],
        [0.011, 0.15, 0.0, 0.0, 0.0, 0.0],
    ])
    mix.set_binary_interaction(lower + np.tril(lower, -1).T)
    info = _case_info(
        injection=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        initial=[0.463, 0.01640, 0.20520, 0.19108, 0.08113, 0.04319],
        pressure=225.0 * ATM,
        temperature=387.45,
    )
    return mix, info


def get_benchmark_mixture(name: str) -> tuple[Mixture, FluidCaseInfo]:
    """Return a benchmark mixture and its MRST-style case information."""
    key = name.lower().replace("-", "_")
    if key in {"light_three", "simple_mixture"}:
        mix = simple_mixture()
        mix.name = "Light methane/n-butane/n-decane mixture"
        info = _case_info(
            injection=[0.9, 0.05, 0.05],
            initial=[0.5, 0.3, 0.2],
            pressure=50.0 * BARSA,
            temperature=350.0,
        )
        return mix, info
    if key == "simple":
        return _simple_case()
    if key == "spe5":
        return _spe5_case()
    if key == "lumped_1":
        return _lumped_1_case()
    raise ValueError(f"unknown compositional fluid case '{name}'")


def get_compositional_fluid_case(name: str) -> tuple[Mixture, FluidCaseInfo]:
    """MRST-facing alias for :func:`get_benchmark_mixture`."""
    return get_benchmark_mixture(name)


def get_compositional_mixture_case(name: str) -> tuple[Mixture, FluidCaseInfo]:
    """Deprecated MRST-name alias retained for layout compatibility."""
    return get_benchmark_mixture(name)


if __name__ == "__main__":
    def test_spe5_case_matches_mrst_sizes():
        mix, info = get_benchmark_mixture("spe5")
        assert mix.names == ("C1", "C3", "C6", "C10", "C15", "C20")
        assert mix.get_number_of_components() == 6
        assert info.initial.shape == (6,)
        assert np.isclose(info.initial.sum(), 1.0)
        assert np.isclose(info.injection.sum(), 1.0)
        assert np.isclose(mix.get_binary_interaction()[4, 0], 0.05)
        print("ok test_spe5_case_matches_mrst_sizes")

    def test_simple_case_has_mrst_streams():
        mix, info = get_compositional_fluid_case("simple")
        assert mix.names == ("Methane", "CarbonDioxide", "n-Decane")
        assert info.injection[1] == 0.9
        assert info.initial[2] == 0.6
        assert info.temp == info.T
        print("ok test_simple_case_has_mrst_streams")

    def test_lumped_bic_is_symmetric():
        mix, _ = get_benchmark_mixture("lumped_1")
        bic = mix.get_binary_interaction()
        assert np.allclose(bic, bic.T)
        assert bic.shape == (6, 6)
        print("ok test_lumped_bic_is_symmetric")

    test_spe5_case_matches_mrst_sizes()
    test_simple_case_has_mrst_streams()
    test_lumped_bic_is_symmetric()
    print("\nAll tests passed!")
