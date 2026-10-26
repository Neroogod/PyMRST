"""Convert a deck to SI (MRST ``deckinput/convertDeckUnits.m``).

The readers deliberately return the deck's own numbers. This is where a deck
in FIELD units becomes a deck in SI, which is the only form the rest of MRST
accepts.

The work is per keyword and per *column*: a PVTW record is pressure, a
dimensionless volume factor, a compressibility, a viscosity and another
compressibility, so it needs five different factors. Getting a column wrong
produces a deck that still looks plausible - which is why every column vector
below is transcribed from `convertPROPS` rather than inferred.

GRID, SOLUTION and SCHEDULE have their own converters, added here alongside
their readers. The SCHEDULE one is the fiddly case: a well record's units
depend on *another field in the same record*. `WCONINJE` column five is a rate,
but a surface liquid rate for a water injector and a surface gas rate for a gas
injector, and column two is what says which. Converting the column with a
single factor is right for eight of the Egg model's wells and would be wrong
for a gas flood.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from pymrst.modules.deckformat.deckinput.read_pvt_tables import MisciblePVT
from pymrst.modules.deckformat.unit_conversion_factors import (
    UnitSystem,
    unit_conversion_factors,
)
from pymrst.utils.units.units import (
    ATM, BARSA, CENTI, DYNE, KELVIN, METER, MOL, PSIA,
)

# Defaults ECLIPSE applies when RTEMP or STCOND is absent or defaulted.
DEFAULT_RTEMP = 288.706 * KELVIN
DEFAULT_STCOND = (288.71 * KELVIN, 1.01325 * BARSA)
VALID_UNIT_NAMES = ("METRIC", "FIELD", "LAB", "PVT_M", "SI")


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


# GRID keyword -> the attribute of `UnitSystem` its values are measured in,
# transcribed from convertGRID.m.
_GRID_UNITS = {
    **{name: "perm" for name in
       ("PERMX", "PERMY", "PERMZ", "PERMXY", "PERMXZ", "PERMYX",
        "PERMYZ", "PERMZX", "PERMZY", "PERMXX", "PERMYY", "PERMZZ")},
    **{name: "length" for name in
       ("DX", "DY", "DZ", "DXV", "DYV", "DZV", "TOPS", "DEPTHZ", "DEPTH",
        "COORD", "COORDX", "COORDY", "COORDZ", "ZCORN", "DZMTRX", "DZMTRXV")},
    **{name: "liqvol_r" for name in ("MINPV", "MINPVV", "PORV")},
    **{name: "trans" for name in ("TRANX", "TRANY", "TRANZ")},
    **{name: "invarea" for name in ("SIGMA", "SIGMAV")},
    "THCONR": "rockcond",
    "YMODULE": "ymodule",
}


def convert_grid(grid: Dict[str, Any], u: UnitSystem) -> Dict[str, Any]:
    """Convert the GRID section from the deck's units to SI."""
    out: Dict[str, Any] = {}
    for keyword, value in (grid or {}).items():
        factor = _GRID_UNITS.get(keyword)
        if factor is None or keyword == "cartDims":
            # Dimensionless (ACTNUM, PORO, NTG, the MULT* modifiers) or not a
            # keyword this port converts.
            out[keyword] = value
            continue
        out[keyword] = np.asarray(value, dtype=float)*getattr(u, factor)
    return out


def convert_solution(solution: Dict[str, Any], u: UnitSystem) -> Dict[str, Any]:
    """Convert the SOLUTION section from the deck's units to SI."""
    from pymrst.modules.deckformat.deckinput.read_solution import DEPTH_TABLES

    out: Dict[str, Any] = {}
    for keyword, value in (solution or {}).items():
        if keyword in ("EQUIL", "STREQUIL"):
            # Three depth/pressure pairs, then dimensionless table selectors.
            factors = [u.length, u.press]*3 + [1.0]*5
            out[keyword] = _scale(np.asarray(value, dtype=float), factors)
        elif keyword in ("PRESSURE", "PBUB"):
            out[keyword] = np.asarray(value, dtype=float)*u.press
        elif keyword == "THPRES":
            # From-region, to-region, threshold pressure.
            out[keyword] = _scale(np.asarray(value, dtype=float), (1.0, 1.0, u.press))
        elif keyword == "RS":
            out[keyword] = np.asarray(value, float)*(u.gasvol_s/u.liqvol_s)
        elif keyword == "RV":
            out[keyword] = np.asarray(value, float)*(u.liqvol_s/u.gasvol_s)
        elif keyword == "TEMPI":
            out[keyword] = (np.asarray(value, float) + u.tempoffset)*u.temp
        elif keyword == "AQUFETP":
            # convertSOLUTION: id, datum depth, pressure, initial volume,
            # compressibility, productivity index, PVT table, salt
            # concentration, temperature.
            out[keyword] = _scale(np.asarray(value, dtype=float), (
                1.0, u.length, u.press, u.liqvol_s, 1.0/u.press,
                u.liqvol_s/u.time/u.press, 1.0, u.mass/u.liqvol_s, u.temp))
        elif keyword == "AQUCT":
            out[keyword] = _scale(np.asarray(value, dtype=float), (
                1.0, u.length, u.press, u.perm, 1.0, 1.0/u.press,
                u.length, u.length, 1.0, 1.0, 1.0, 0.0, float("nan")))
        elif keyword == "AQUANCON":
            # Column 9 (1-based) is the influx coefficient - an *area*. The
            # rest are indices, a face string, and a dimensionless multiplier,
            # so this is the one column with a unit and the row cannot be
            # scaled as a block.
            out[keyword] = [
                [(field*u.length**2 if index == 8
                  and isinstance(field, (int, float)) else field)
                 for index, field in enumerate(record)]
                for record in value]
        elif keyword == "DATUM":
            out[keyword] = float(value)*u.length
        elif keyword in DEPTH_TABLES:
            columns = {"RSVD": u.gasvol_s/u.liqvol_s,
                       "RVVD": u.liqvol_s/u.gasvol_s,
                       "RTEMPVD": u.temp}.get(keyword, u.press)
            offset = u.tempoffset if keyword == "RTEMPVD" else 0.0
            out[keyword] = [
                _scale(np.asarray(table, float) + [0.0, offset],
                       (u.length, columns)) for table in value]
        else:
            # SGAS, SWAT, SOIL are saturations; the report controls are text.
            out[keyword] = value
    return out


def convert_schedule(schedule: Any, u: UnitSystem) -> Any:
    """Convert the SCHEDULE section from the deck's units to SI."""
    step = getattr(schedule, "step", None)
    if step is not None:
        step["val"] = list(np.asarray(step["val"], dtype=float)*u.time)
    for index, control in enumerate(getattr(schedule, "control", []) or []):
        schedule.control[index] = convert_control(control, u)
    return schedule


#: `keyword -> {0-based column: factor name or callable}`, from convertDeckUnits.
def _control_columns(u: UnitSystem) -> Dict[str, Dict[int, float]]:
    resv = u.liqvol_r/u.time
    lrat = u.liqvol_s/u.time
    grat = u.gasvol_s/u.time
    return {
        # convertWellSpecs: columns 5 and 7 (1-based) are depths.
        "WELSPECS": {4: u.length, 6: u.length},
        # convertCompDat: 8 transmissibility, 9 diameter, 10 Kh, 14 length.
        "COMPDAT": {7: u.trans, 8: u.length, 9: u.perm*u.length,
                    13: u.length},
        # convertWConProd: 4..8 rates, 9 and 10 pressures.
        "WCONPROD": {3: lrat, 4: lrat, 5: grat, 6: lrat, 7: resv,
                     8: u.press, 9: u.press},
        # convertWconHist: 4..6 rates, 9 and 10 pressures, 11 gas rate.
        "WCONHIST": {3: lrat, 4: lrat, 5: grat, 8: u.press, 9: u.press,
                     10: grat},
        # convertWConInje: 6 reservoir rate, 8 pressure, 11 gas-oil ratio.
        # Columns 5 and 10 depend on the injected phase; see below.
        "WCONINJE": {5: resv, 7: u.press, 10: u.gasvol_s/u.liqvol_s},
        "WCONINJ": {5: resv, 7: u.press},
        "WTEMP": {1: u.temp},
    }


def convert_control(control: Dict[str, Any], u: UnitSystem) -> Dict[str, Any]:
    """Convert one control period's well records."""
    columns = _control_columns(u)
    out = dict(control)
    for keyword, factors in columns.items():
        records = out.get(keyword)
        if not records:
            continue
        out[keyword] = [_convert_record(record, factors)
                        for record in records]
    if out.get("WCONINJE"):
        out["WCONINJE"] = [_convert_wconinje(record, u)
                           for record in out["WCONINJE"]]
    return out


def _convert_record(record: Sequence[Any],
                    factors: Dict[int, float]) -> List[Any]:
    row = list(record)
    for column, factor in factors.items():
        if column < len(row) and isinstance(row[column], (int, float)):
            row[column] = float(row[column])*factor
    return row


def _convert_wconinje(record: List[Any], u: UnitSystem) -> List[Any]:
    """The phase-dependent columns of a `WCONINJE` record.

    Column five is a surface rate whose unit is the *injected phase's*: gas
    volume for a gas injector, liquid volume otherwise. In METRIC both are m³
    and the distinction is invisible; in FIELD they are Mscf and stb, a factor
    of about 178 apart, and nothing downstream would question the result.

    Column seven is the bottom-hole limit, and a defaulted one means "no
    limit" rather than zero - MRST substitutes 100000 psia, which is what
    keeps a rate-controlled injector rate-controlled.
    """
    row = list(record)
    phase = str(row[1]).strip().upper() if len(row) > 1 else ""
    if len(row) > 4 and isinstance(row[4], (int, float)):
        rate = u.gasvol_s/u.time if phase == "GAS" else u.liqvol_s/u.time
        row[4] = float(row[4])*rate
    if len(row) > 9 and isinstance(row[9], (int, float)):
        ratio = (u.gasvol_s/u.liqvol_s if phase == "OIL"
                 else u.liqvol_s/u.gasvol_s)
        row[9] = float(row[9])*ratio
    if len(row) > 6:
        limit = row[6]
        row[6] = (100.0e3*PSIA if not isinstance(limit, (int, float))
                  or np.isnan(limit) else float(limit)*u.press)
    return row


def convert_deck_units(deck: Dict[str, Any],
                       output_unit: str = "SI") -> Dict[str, Any]:
    """Convert ``deck`` from the unit system its RUNSPEC declares."""
    if "RUNSPEC" not in deck:
        raise ValueError("input does not appear to be a valid deck")

    input_unit = _deck_unit_name(deck["RUNSPEC"])
    u = unit_conversion_factors(input_unit, output_unit)

    converted = dict(deck)
    converted.setdefault("PCUNIT", u.press)   # lets Pc be read as a J-function
    converted["PROPS"] = convert_props(deck.get("PROPS", {}), u)
    converted["GRID"] = convert_grid(deck.get("GRID") or {}, u)
    converted["SOLUTION"] = convert_solution(deck.get("SOLUTION") or {}, u)
    if deck.get("SCHEDULE") is not None:
        converted["SCHEDULE"] = convert_schedule(
            copy.deepcopy(deck["SCHEDULE"]), u)

    runspec = dict(deck["RUNSPEC"])
    for name in ("METRIC", "FIELD", "LAB", "PVT-M", "PVT_M", "SI"):
        runspec.pop(name, None)
    runspec[output_unit.upper().replace("-", "_")] = True
    converted["RUNSPEC"] = runspec
    return converted


def _deck_unit_name(runspec: Dict[str, Any]) -> str:
    """MRST's `deckUnitName`: the one unit keyword RUNSPEC carries, SI
    included, so converting an already-converted deck changes nothing."""
    occurrence = [name for name in VALID_UNIT_NAMES
                  if name in runspec or name.replace("_", "-") in runspec]
    if len(occurrence) != 1:
        raise ValueError("Input unit system must be either METRIC, FIELD, LAB, "
                         "PVT_M, or SI.")
    return occurrence[0]


convertDeckUnits = convert_deck_units
convertPROPS = convert_props
convertGRID = convert_grid
convertSOLUTION = convert_solution
convertSCHEDULE = convert_schedule
convertControl = convert_control


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

    def test_an_si_deck_is_not_converted_again() -> None:
        deck = {"RUNSPEC": {"SI": True}, "PROPS": {"PVTW": np.array([[2e7, 1.0, 4e-10,
                                                                      1e-3, 0.0]])}}
        again = convert_deck_units(deck)
        assert np.array_equal(again["PROPS"]["PVTW"], deck["PROPS"]["PVTW"])
        assert again["RUNSPEC"] == {"SI": True}
        print("ok test_an_si_deck_is_not_converted_again")

    def test_the_unit_keyword_must_be_unique() -> None:
        for runspec in ({}, {"METRIC": True, "SI": True}):
            try:
                convert_deck_units({"RUNSPEC": runspec})
            except ValueError:
                continue
            raise AssertionError(f"{runspec} has no single unit system")
        assert _deck_unit_name({"PVT-M": True}) == "PVT_M"
        print("ok test_the_unit_keyword_must_be_unique")

    def test_threshold_pressures_are_converted() -> None:
        u = unit_conversion_factors("METRIC")
        out = convert_solution({"THPRES": np.array([[1.0, 2.0, 0.5]])}, u)
        np.testing.assert_allclose(out["THPRES"], [[1.0, 2.0, 0.5*u.press]])
        print("ok test_threshold_pressures_are_converted")

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

    # -- GRID, SOLUTION and SCHEDULE --------------------------------------
    #
    # Written in FIELD units throughout. METRIC hides most of these: a
    # millidarcy is a millidarcy, and surface gas and surface liquid are both
    # cubic metres - so a converter that confuses them passes every METRIC
    # test and fails only on a deck nobody ran.

    from pymrst.modules.deckformat.deckinput.initialize_deck import Schedule
    from pymrst.modules.deckformat.unit_conversion_factors import (
        unit_conversion_factors,
    )
    from pymrst.utils.units.units import DARCY, DAY, FT, MILLI, STB

    FIELD = unit_conversion_factors("FIELD", "SI")

    def test_grid_permeability_and_lengths_are_converted() -> None:
        grid = convert_grid({"PERMX": np.array([100.0, 200.0]),
                             "TOPS": np.array([8000.0]),
                             "PORO": np.array([0.2, 0.3]),
                             "cartDims": (2, 1, 1)}, FIELD)
        assert np.allclose(grid["PERMX"]/(MILLI*DARCY), [100.0, 200.0])
        assert np.isclose(grid["TOPS"][0]/FT, 8000.0)
        assert np.allclose(grid["PORO"], [0.2, 0.3]), "porosity is a fraction"
        assert grid["cartDims"] == (2, 1, 1)
        print("ok test_grid_permeability_and_lengths_are_converted")

    def test_equil_alternates_depth_and_pressure() -> None:
        """Three depth/pressure pairs, then table selectors. Scaling the row
        with one factor puts the datum pressure in feet."""
        row = np.array([[8000.0, 4000.0, 8500.0, 0.0, 7500.0, 0.0,
                         0.0, 0.0, 0.0, 1.0, 0.0]])
        equil = convert_solution({"EQUIL": row}, FIELD)["EQUIL"][0]
        assert np.isclose(equil[0]/FT, 8000.0)
        assert np.isclose(equil[1]/PSIA, 4000.0)
        assert np.isclose(equil[2]/FT, 8500.0)
        assert equil[9] == 1.0, "the table selectors are dimensionless"
        print("ok test_equil_alternates_depth_and_pressure")

    def test_solution_arrays_take_their_own_units() -> None:
        solution = convert_solution({"PRESSURE": np.array([4000.0]),
                                     "SWAT": np.array([0.12])}, FIELD)
        assert np.isclose(solution["PRESSURE"][0]/PSIA, 4000.0)
        assert np.isclose(solution["SWAT"][0], 0.12), "saturation is a fraction"
        print("ok test_solution_arrays_take_their_own_units")

    def _schedule(**control):
        base = {"WELSPECS": [], "COMPDAT": [], "WCONPROD": [],
                "WCONINJE": [], "WCONHIST": [], "WCONINJ": []}
        base.update(control)
        return Schedule(control=[base],
                        step={"val": [30.0, 30.0], "control": [1, 1]})

    def test_step_lengths_become_seconds() -> None:
        out = convert_schedule(_schedule(), FIELD)
        assert np.allclose(np.asarray(out.step["val"])/DAY, 30.0)
        print("ok test_step_lengths_become_seconds")

    def test_a_gas_injector_rate_is_a_gas_volume() -> None:
        """The claim this module's docstring makes, and the one METRIC cannot
        test. `WCONINJE` column five is a surface rate in the *injected
        phase's* unit: Mscf/day for gas, stb/day for water. In FIELD they are
        about 178 apart, so a single factor is not a small error.
        """
        record = ["I1", "GAS", "OPEN", "RATE", 1000.0, float("inf"),
                  4000.0, float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gas = convert_control(_schedule(WCONINJE=[record]).control[0],
                              FIELD)["WCONINJE"][0]
        water = convert_control(
            _schedule(WCONINJE=[["I1", "WATER"] + record[2:]]).control[0],
            FIELD)["WCONINJE"][0]

        assert not np.isclose(gas[4], water[4]), (gas[4], water[4])
        assert np.isclose(gas[4], 1000.0*FIELD.gasvol_s/FIELD.time)
        assert np.isclose(water[4]*DAY/STB, 1000.0), water[4]*DAY/STB
        print(f"ok test_a_gas_injector_rate_is_a_gas_volume "
              f"({gas[4]/water[4]:.0f}x apart)")

    def test_a_defaulted_injector_pressure_limit_is_not_zero() -> None:
        """A defaulted column seven means "no limit"; MRST substitutes
        100000 psia. Converting the NaN gives a zero bottom-hole limit, and
        the injector then cannot inject at all."""
        record = ["I1", "WATER", "OPEN", "RATE", 1000.0, float("inf"),
                  float("nan"), float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        out = convert_control(_schedule(WCONINJE=[record]).control[0],
                              FIELD)["WCONINJE"][0]
        assert np.isclose(out[6]/PSIA, 100.0e3), out[6]/PSIA
        print("ok test_a_defaulted_injector_pressure_limit_is_not_zero")

    def test_producer_rate_and_pressure_columns_take_their_own_units() -> None:
        record = ["P1", "OPEN", "BHP", 1000.0, 500.0, 2000.0, 1500.0,
                  100.0, 3000.0, 0.0, 0.0, 0.0]
        out = convert_control(_schedule(WCONPROD=[record]).control[0],
                              FIELD)["WCONPROD"][0]
        assert np.isclose(out[3]*DAY/STB, 1000.0), "oil rate"
        assert np.isclose(out[5]/(FIELD.gasvol_s/FIELD.time), 2000.0), "gas"
        assert np.isclose(out[8]/PSIA, 3000.0), "bottom-hole limit"
        print("ok test_producer_rate_and_pressure_columns_take_their_own_units")

    def test_compdat_columns_take_transmissibility_and_length() -> None:
        record = ["P1", -1.0, -1.0, 1.0, 3.0, "OPEN", -1.0, -1.0, 0.5,
                  -1.0, 0.0, "Default", "Z", -1.0]
        out = convert_control(_schedule(COMPDAT=[record]).control[0],
                              FIELD)["COMPDAT"][0]
        assert np.isclose(out[8]/FT, 0.5), "the wellbore diameter"
        assert out[3] == 1.0 and out[4] == 3.0, "layer indices are not lengths"
        assert out[5] == "OPEN", "text columns are untouched"
        print("ok test_compdat_columns_take_transmissibility_and_length")

    def test_the_schedule_conversion_leaves_the_input_alone() -> None:
        schedule = _schedule(WCONPROD=[["P1", "OPEN", "BHP", 1000.0, 500.0,
                                        2000.0, 1500.0, 100.0, 3000.0,
                                        0.0, 0.0, 0.0]])
        deck = {"RUNSPEC": {"FIELD": True}, "SCHEDULE": schedule}
        convert_deck_units(deck)
        assert schedule.step["val"] == [30.0, 30.0]
        assert schedule.control[0]["WCONPROD"][0][8] == 3000.0
        print("ok test_the_schedule_conversion_leaves_the_input_alone")

    test_pressure_columns_are_converted()
    test_saturation_columns_stay_dimensionless()
    test_pvto_converts_key_and_data_differently()
    test_si_input_is_left_alone()
    test_conversion_does_not_mutate_the_input()
    test_unknown_keywords_pass_through()
    test_threshold_pressures_are_converted()
    test_an_si_deck_is_not_converted_again()
    test_the_unit_keyword_must_be_unique()
    test_deck_level_conversion_updates_the_unit_flag()
    test_converted_spe1_matches_the_bespoke_reader()
    test_grid_permeability_and_lengths_are_converted()
    test_equil_alternates_depth_and_pressure()
    test_solution_arrays_take_their_own_units()
    test_step_lengths_become_seconds()
    test_a_gas_injector_rate_is_a_gas_volume()
    test_a_defaulted_injector_pressure_limit_is_not_zero()
    test_producer_rate_and_pressure_columns_take_their_own_units()
    test_compdat_columns_take_transmissibility_and_length()
    test_the_schedule_conversion_leaves_the_input_alone()
    print("\nAll tests passed!")
