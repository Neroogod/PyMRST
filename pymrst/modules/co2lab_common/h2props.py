"""Sampled hydrogen property tables (MRST ``H2props`` layout)."""

from pathlib import Path

from pymrst.modules.co2lab_common.co2props import CO2Props, SampledProp2D

DEFAULT_H2_TABLE_DIR = (
    Path(__file__).parents[4]
    / "mrst-2024a-Geomech/modules/h2lab/utils/h2props/sampled_tables"
)
DEFAULT_H2_STEM = "Hydrogen_100000_400000000_278_524_800_800"


def h2_props(table_dir=None, stem: str = DEFAULT_H2_STEM, with_enthalpy: bool = False) -> CO2Props:
    """Load the MRST hydrogen property tables."""
    d = Path(table_dir) if table_dir is not None else DEFAULT_H2_TABLE_DIR
    return CO2Props(
        rho=SampledProp2D.from_mat(d / f"{stem}_D.mat"),
        mu=SampledProp2D.from_mat(d / f"{stem}_V.mat"),
        h=SampledProp2D.from_mat(d / f"{stem}_H.mat") if with_enthalpy else None,
    )


def h2_brine_fluid(temperature: float, **kwargs):
    """Brine/H2 fluid: CO2-brine model with hydrogen property tables."""
    from pymrst.modules.co2lab_common.co2_brine import co2_brine_fluid

    kwargs.setdefault("props", h2_props())
    return co2_brine_fluid(temperature, **kwargs)


__all__ = ["h2_brine_fluid", "h2_props"]
