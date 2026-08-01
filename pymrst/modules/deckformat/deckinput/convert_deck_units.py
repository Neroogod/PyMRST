"""Convert a deck to SI (MRST ``deckinput/convertDeckUnits.m``).

The readers deliberately return the deck's own numbers. This is where a deck
in FIELD units becomes a deck in SI, which is the only form the rest of MRST
accepts.

The work is per keyword and per *column*: a PVTW record is pressure, a
dimensionless volume factor, a compressibility, a viscosity and another
compressibility, so it needs five different factors. Getting a column wrong
produces a deck that still looks plausible - which is why every column vector
below is transcribed from `convertPROPS` rather than inferred.

Only the PROPS section is converted here. GRID, SOLUTION and SCHEDULE have
their own converters in MRST, and this port grows them alongside their
readers; a deck section that has not been read has nothing to convert.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from pymrst.modules.deckformat.deckinput.read_pvt_tables import MisciblePVT
from pymrst.modules.deckformat.unit_conversion_factors import (
    UnitSystem,
    unit_conversion_factors,
)
from pymrst.utils.units.units import ATM, BARSA, CENTI, DYNE, KELVIN, METER, MOL

# Defaults ECLIPSE applies when RTEMP or STCOND is absent or defaulted.
DEFAULT_RTEMP = 288.706 * KELVIN
DEFAULT_STCOND = (288.71 * KELVIN, 1.01325 * BARSA)


def _columns(u: UnitSystem, keyword: str) -> Optional[Sequence[float]]:
    """The per-column factors for a table or record keyword.

    Transcribed from convertPROPS.m. A `None` return means "not a keyword with
    a fixed column layout" - those are handled separately below.
    """
    press, visc, compr = u.press, u.viscosity, u.compr
    bo = u.liqvol_r / u.liqvol_s          # oil formation volume factor
    bg = u.gasvol_r / u.gasvol_s          # gas formation volume factor
    rs = u.gasvol_s / u.liqvol_s          # solution gas-oil ratio
    rv = u.liqvol_s / u.gasvol_s          # vaporised oil-gas ratio

    return {
        "DENSITY": (u.density,) * 3,
        "SDENSITY": (u.density,) * 3,
        "ROCK": (press, compr, compr, compr, 1.0, 1.0),
        "ROCKTAB": (press,),               # remaining columns dimensionless
        "PVTW": (press, bo, compr, visc, compr),
        "PVCDO": (press, 1.0, compr, visc, compr),
        "PVDO": (press, bo, visc),
        "PVDG": (press, bg, visc),
        "PVDS": (press, bg, visc),
        "PVCO": (press, bo, visc, compr, compr, 1.0),
        "RSCONSTT": (rs, press),
        "SWOF": (1.0, 1.0, 1.0, press),
        "SGOF": (1.0, 1.0, 1.0, press),
        "SGWFN": (1.0, 1.0, 1.0, press),
        "SLGOF": (1.0, 1.0, 1.0, press),
        "SWFN": (1.0, 1.0, press),
        "SGFN": (1.0, 1.0, press),
        "SOF2": (1.0, 1.0),
        "SOF3": (1.0, 1.0, 1.0),
        "SSFN": (1.0, 1.0, 1.0),
        "MSFN": (1.0, 1.0, 1.0),
        "MISC": (1.0, 1.0),
        "PMISC": (press, 1.0),
        "SPECHEAT": (u.temp,) + (u.massheatcapacity,) * 3,
        "SPECROCK": (u.temp, u.volumheatcapacity),
        "SURFST": (u.concentr, u.surf_tension),
        "SURFADS": (u.concentr, 1.0),
        "SURFCAPD": (1.0, 1.0),
        "SURFROCK": (1.0, u.concentr),
        "SURFVISC": (u.concentr, visc),
        "PLYADS": (u.concentr, 1.0),
        "PLYVISC": (u.concentr, 1.0),
        "PLYSHEAR": (u.length / u.time, 1.0),
        "PLYMAX": (u.concentr, u.concentr),
        "PLYROCK": (1.0, 1.0, u.concentr, 1.0, 1.0),
        "PLMIXPAR": (1.0,),
        "TLMIXPAR": (1.0, 1.0),
        "SHRATE": (1.0,),
        "GRAVITY": (1.0, 1.0, 1.0),
        # component vectors
        "PCRIT": (press,),
        "TCRIT": (u.temp,),
        "VCRIT": (u.density / (u.mass * u.mol),),
        "ZCRIT": (1.0,),
        "MW": (u.mass / u.mol,),
        "ACF": (1.0,),
        "SSHIFT": (1.0,),
        "PARACHOR": (((CENTI * METER) ** 3 / MOL)
                     * (DYNE / (CENTI * METER)) ** -0.25,),
    }.get(keyword)


def _scale(array: np.ndarray, factors: Sequence[float]) -> np.ndarray:
    """Multiply each column by its factor, reusing the last for any surplus."""
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        if len(factors) == 1:
            return array * factors[0]
        width = min(array.size, len(factors))
        out = array.astype(float).copy()
        out[:width] *= np.asarray(factors[:width], dtype=float)
        return out
    width = array.shape[1]
    columns = list(factors[:width])
    columns += [1.0] * (width - len(columns))
    return array * np.asarray(columns, dtype=float)


def convert_props(props: Dict[str, Any], u: UnitSystem) -> Dict[str, Any]:
    """Convert the PROPS section from the deck's units to SI."""
    out: Dict[str, Any] = {}
    for keyword, value in props.items():
        if keyword in ("PVTO", "PVTG"):
            key_unit = (u.gasvol_s / u.liqvol_s if keyword == "PVTO"
                        else u.press)
            data_units = ((u.press, u.liqvol_r / u.liqvol_s, u.viscosity)
                          if keyword == "PVTO" else
                          (u.liqvol_s / u.gasvol_s,
                           u.gasvol_r / u.gasvol_s, u.viscosity))
            out[keyword] = [
                MisciblePVT(key=table.key * key_unit, pos=table.pos,
                            data=_scale(table.data, data_units))
                for table in value
            ]
            continue

        if keyword in ("RTEMP", "RTEMPA"):
            values = (np.asarray(value, dtype=float) + u.tempoffset) * u.temp
            out[keyword] = np.where(np.isnan(values), DEFAULT_RTEMP, values)
            continue

        if keyword == "STCOND":
            record = np.asarray(value, dtype=float).reshape(-1, 2).copy()
            record[:, 0] = (record[:, 0] + u.tempoffset) * u.temp
            record[:, 1] = record[:, 1] * u.press
            for column, default in enumerate(DEFAULT_STCOND):
                missing = np.isnan(record[:, column])
                record[missing, column] = default
            out[keyword] = record
            continue

        factors = _columns(u, keyword)
        if factors is None:
            # Dimensionless, or a keyword this port does not convert. MRST's
            # own list of dimensionless PROPS keywords runs to several hundred
            # entries, nearly all endpoint-scaling saturations.
            out[keyword] = value
            continue

        out[keyword] = ([_scale(table, factors) for table in value]
                        if isinstance(value, list) else _scale(value, factors))
    return out


def convert_deck_units(deck: Dict[str, Any],
                       output_unit: str = "SI") -> Dict[str, Any]:
    """Convert ``deck`` from the unit system its RUNSPEC declares."""
    from pymrst.modules.deckformat.deckinput.read_runspec import unit_system

    if "RUNSPEC" not in deck:
        raise ValueError("input does not appear to be a valid deck")

    input_unit = unit_system(deck)
    u = unit_conversion_factors(input_unit, output_unit)

    converted = dict(deck)
    converted.setdefault("PCUNIT", u.press)   # lets Pc be read as a J-function
    converted["PROPS"] = convert_props(deck.get("PROPS", {}), u)

    runspec = dict(deck["RUNSPEC"])
    for name in ("METRIC", "FIELD", "LAB", "PVT-M", "PVT_M", "SI"):
        runspec.pop(name, None)
    runspec[output_unit.upper().replace("-", "_")] = True
    converted["RUNSPEC"] = runspec
    return converted


convertDeckUnits = convert_deck_units
convertPROPS = convert_props


if __name__ == "__main__":
    from pathlib import Path

    from pymrst.utils.units.units import PSIA

    DATA = Path("/Users/Shared/PyMRST_workspace/"
                "mrst-2024a-Geomech/examples/data")

    def _spe1():
        from pymrst.modules.deckformat.deckinput.read_eclipse_deck import (
            read_eclipse_deck,
        )
        path = DATA / "SPE1" / "BENCH_SPE1.DATA"
        return read_eclipse_deck(path) if path.exists() else None

    def test_pressure_columns_are_converted() -> None:
        u = unit_conversion_factors("FIELD")
        props = convert_props({"PVTW": np.array([[4014.7, 1.029, 3.13e-6,
                                                  0.31, 0.0]])}, u)
        assert np.isclose(props["PVTW"][0, 0], 4014.7 * PSIA)
        assert np.isclose(props["PVTW"][0, 1], 1.029), "Bw is dimensionless"
        assert np.isclose(props["PVTW"][0, 2], 3.13e-6 / PSIA)
        print("ok test_pressure_columns_are_converted")

    def test_saturation_columns_stay_dimensionless() -> None:
        u = unit_conversion_factors("FIELD")
        swof = np.array([[0.12, 0.0, 1.0, 5.0], [0.9, 1.0, 0.0, 1.0]])
        out = convert_props({"SWOF": [swof]}, u)["SWOF"][0]
        assert np.allclose(out[:, :3], swof[:, :3]), "saturations must not move"
        assert np.isclose(out[0, 3], 5.0 * PSIA), "only Pc is a pressure"
        print("ok test_saturation_columns_stay_dimensionless")

    def test_pvto_converts_key_and_data_differently() -> None:
        """Rs scales as gas/liquid volume; the rows are pressure, Bo, viscosity."""
        u = unit_conversion_factors("FIELD")
        packed = MisciblePVT(key=np.array([1.0]), pos=np.array([0, 1]),
                             data=np.array([[4014.7, 1.2, 0.5]]))
        out = convert_props({"PVTO": [packed]}, u)["PVTO"][0]
        assert np.isclose(out.key[0], u.gasvol_s / u.liqvol_s)
        assert np.isclose(out.data[0, 0], 4014.7 * PSIA)
        assert np.array_equal(out.pos, packed.pos)
        print("ok test_pvto_converts_key_and_data_differently")

    def test_si_input_is_left_alone() -> None:
        u = unit_conversion_factors("SI")
        props = convert_props({"DENSITY": np.array([[900.0, 1000.0, 0.8]])}, u)
        assert np.allclose(props["DENSITY"], [[900.0, 1000.0, 0.8]])
        print("ok test_si_input_is_left_alone")

    def test_conversion_does_not_mutate_the_input() -> None:
        u = unit_conversion_factors("FIELD")
        original = np.array([[4014.7, 1.029, 3.13e-6, 0.31, 0.0]])
        convert_props({"PVTW": original.copy()}, u)
        convert_props({"PVTW": original}, u)
        assert np.isclose(original[0, 0], 4014.7), original
        print("ok test_conversion_does_not_mutate_the_input")

    def test_unknown_keywords_pass_through() -> None:
        u = unit_conversion_factors("FIELD")
        out = convert_props({"SWL": np.array([0.12, 0.12])}, u)
        assert np.allclose(out["SWL"], [0.12, 0.12])
        print("ok test_unknown_keywords_pass_through")

    def test_deck_level_conversion_updates_the_unit_flag() -> None:
        deck = _spe1()
        if deck is None:
            print("skip test_deck_level_conversion_updates_the_unit_flag")
            return
        converted = convert_deck_units(deck)
        assert converted["RUNSPEC"].get("SI") is True
        assert "FIELD" not in converted["RUNSPEC"]
        print("ok test_deck_level_conversion_updates_the_unit_flag")

    def test_converted_spe1_matches_the_bespoke_reader() -> None:
        """The decisive check: same SI numbers as the trusted SPE1 reader."""
        deck = _spe1()
        if deck is None:
            print("skip test_converted_spe1_matches_the_bespoke_reader")
            return
        from pymrst.modules.deckformat.spe1.read_spe1_deck import read_spe1_deck
        bespoke = read_spe1_deck(DATA / "SPE1" / "BENCH_SPE1.DATA")
        props = convert_deck_units(deck)["PROPS"]

        assert np.allclose(props["SWOF"][0], bespoke["swof"]), "SWOF"
        assert np.allclose(props["SGOF"][0], bespoke["sgof"]), "SGOF"
        assert np.allclose(props["PVDG"][0], bespoke["pvdg"]), "PVDG"

        pvtw = bespoke["pvtw"]
        assert np.allclose(props["PVTW"][0],
                           [pvtw["p_ref"], pvtw["b_ref"], pvtw["c"],
                            pvtw["mu_ref"], pvtw["viscosibility"]]), "PVTW"
        rock = bespoke["rock_cp"]
        assert np.allclose(props["ROCK"][0][:2], [rock["p_ref"], rock["c"]]), \
            "ROCK"

        table = props["PVTO"][0]
        for index, (rs, rows) in enumerate(bespoke["pvto"]):
            assert np.isclose(table.key[index], rs), f"PVTO key {index}"
            block = table.data[table.pos[index]:table.pos[index + 1]]
            assert np.allclose(block, rows), f"PVTO branch {index}"
        print("ok test_converted_spe1_matches_the_bespoke_reader")

    test_pressure_columns_are_converted()
    test_saturation_columns_stay_dimensionless()
    test_pvto_converts_key_and_data_differently()
    test_si_input_is_left_alone()
    test_conversion_does_not_mutate_the_input()
    test_unknown_keywords_pass_through()
    test_deck_level_conversion_updates_the_unit_flag()
    test_converted_spe1_matches_the_bespoke_reader()
    print("\nAll tests passed!")
