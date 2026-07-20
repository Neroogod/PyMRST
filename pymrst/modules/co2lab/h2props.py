"""Sampled hydrogen property tables (MRST h2lab: H2props).

Same tabulated format as the CO2 tables (uniform 800x800 grid in
pressure and temperature, bilinearly interpolated), so the SampledProp2D
machinery is reused. This turns the CO2-brine model into an H2 storage
model simply by swapping the gas properties.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from pymrst.modules.co2lab_common.co2props import CO2Props, SampledProp2D

DEFAULT_H2_TABLE_DIR = (
    Path(__file__).parents[4]
    / "mrst-2024a-Geomech/modules/h2lab/utils/h2props/sampled_tables"
)
DEFAULT_H2_STEM = "Hydrogen_100000_400000000_278_524_800_800"


def h2_props(table_dir=None, stem: str = DEFAULT_H2_STEM,
             with_enthalpy: bool = False) -> CO2Props:
    """Load the MRST hydrogen property tables (H2props)."""
    d = Path(table_dir) if table_dir is not None else DEFAULT_H2_TABLE_DIR
    return CO2Props(
        rho=SampledProp2D.from_mat(d / f"{stem}_D.mat"),
        mu=SampledProp2D.from_mat(d / f"{stem}_V.mat"),
        h=SampledProp2D.from_mat(d / f"{stem}_H.mat") if with_enthalpy
        else None,
    )


def h2_brine_fluid(temperature: float, **kwargs):
    """Brine/H2 fluid: co2_brine_fluid with hydrogen property tables."""
    from pymrst.modules.co2lab_common.co2_brine import co2_brine_fluid
    kwargs.setdefault("props", h2_props())
    return co2_brine_fluid(temperature, **kwargs)


if __name__ == "__main__":
    from pymrst.modules.co2lab_common.co2props import co2_props
    from pymrst.utils.units import BARSA

    def test_h2_tables_load():
        """Hydrogen tables have the same span/resolution as the CO2 ones."""
        h2 = h2_props()
        assert h2.rho.vals.shape == (800, 800)
        assert h2.rho.v1_span == (1e5, 4e8)
        print("ok test_h2_tables_load")

    def test_h2_much_lighter_than_co2():
        """At storage conditions H2 is ~10x lighter and ~30x less viscous."""
        h2, co2 = h2_props(), co2_props()
        p = np.array([150 * BARSA])
        t = np.array([273.15 + 45.0])
        rho_h2 = float(h2.rho(p, t)[0])
        rho_co2 = float(co2.rho(p, t)[0])
        mu_h2 = float(h2.mu(p, t)[0])
        assert 5.0 < rho_h2 < 20.0, rho_h2
        assert rho_co2 / rho_h2 > 30.0
        assert 5e-6 < mu_h2 < 2e-5, mu_h2
        print(f"ok test_h2_much_lighter (rho_H2 = {rho_h2:.1f} vs "
              f"rho_CO2 = {rho_co2:.0f} kg/m3)")

    def test_h2_brine_fluid():
        """The H2 fluid is buoyant in brine and usable by the model."""
        f = h2_brine_fluid(temperature=273.15 + 45.0, srw=0.2, src=0.05)
        p = np.array([150 * BARSA])
        rho = float(f.b_g(p)[0] * f.rhoGS)
        assert rho < 0.05 * f.rhoWS, "H2 is far lighter than brine"
        assert np.isclose(f.kr_g(np.array([1.0]))[0], 1.0)
        print(f"ok test_h2_brine_fluid (rho_H2 = {rho:.1f} kg/m3)")

    test_h2_tables_load()
    test_h2_much_lighter_than_co2()
    test_h2_brine_fluid()
    print("\nAll tests passed!")
