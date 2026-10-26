"""MRST ``ad-blackoil/examples/utils/getNorneFieldDeckOPMData.m``.

The Norne deck (read and converted to SI) and the reference results that
come with it: OPM Flow's (`flow`, else `flow_legacy`) well curves from the
summary and states from the restart, with the run's grid and file prefix
(`output.opm.location`, whose EGRID/INIT the example builds its model on),
and ECLIPSE's (`ECL.2014.2`) well curves and, when its restart is there,
states. A reference that cannot be read is reported and left empty, as in
MRST; `use_opm` is passed to `convertRestartToStates`, which does not know
it and says so.

`get_norne_field_deck_opm_data_ad`, below, is PyMRST's earlier version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import (
    AquiferEclipseResult, load_aquifer_eclipse_result_ad,
)
from pymrst.modules.deckformat.norne import (
    NorneFieldDeck, norne_output_prefix, read_norne_field_deck,
)


@dataclass
class NorneReferenceOutputs:
    opm: AquiferEclipseResult
    eclipse: AquiferEclipseResult | None


def get_norne_field_deck_opm_data():
    """MRST `[deck, output] = getNorneFieldDeckOPMData()`."""
    from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import (
        local_dataset_path,
    )
    from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck

    name, case_name = "norne", "NORNE_ATW2013"
    pth = local_dataset_path("norne_field_model", f"opm-data/{name}/{case_name}.DATA")
    opm = pth/"opm-data"
    opm_tests = pth/"opm-tests"
    deck = read_eclipse_deck(opm/name/f"{case_name}.DATA")
    unit = next((u for u in ("METRIC", "FIELD", "SI", "LAB", "PVT_M")
                 if u in deck["RUNSPEC"]), "SI")
    deck = convert_deck_units(deck)
    fldr = opm_tests/name
    output = SimpleNamespace(opm=None, ecl=None)
    try:
        output.opm = _get_output_opm(deck, fldr, case_name, unit)
    except Exception as e:                                  # noqa: BLE001 - MRST's catch
        print(f"Error in parsing of OPM output. Reported error:\n {e}")
    try:
        output.eclipse = _get_output_eclipse(deck, fldr, case_name, unit)
    except Exception as e:                                  # noqa: BLE001 - MRST's catch
        print(f"Error in parsing of Eclipse output. Reported error:\n {e}")
    return deck, output


getNorneFieldDeckOPMData = get_norne_field_deck_opm_data


def _get_output_opm(deck, fldr: Path, case_name: str, unit: str) -> SimpleNamespace:
    from pymrst.modules.deckformat.grid.init_eclipse_grid import init_eclipse_grid
    from pymrst.modules.deckformat.resultinput.convert_restart_to_states import (
        convert_restart_to_states,
    )
    from pymrst.modules.deckformat.resultinput.convert_summary_to_well_sols import (
        convert_summary_to_well_sols,
    )
    G = init_eclipse_grid(deck, split_disconnected=False)
    smry_prefix = fldr/"opm-simulation-reference"/"flow"/case_name
    if not Path(str(smry_prefix) + ".UNSMRY").exists():
        smry_prefix = fldr/"opm-simulation-reference"/"flow_legacy"/case_name
    ws, wstime = convert_summary_to_well_sols(smry_prefix, unit)
    states, _ = convert_restart_to_states(smry_prefix, G, use_opm=True,
                                          consistentWellSols=False)
    return SimpleNamespace(wellSols=ws, states=states, G=G, location=str(smry_prefix),
                           time=wstime)


def _get_output_eclipse(deck, fldr: Path, case_name: str, unit: str) -> SimpleNamespace:
    from pymrst.modules.deckformat.grid.init_eclipse_grid import init_eclipse_grid
    from pymrst.modules.deckformat.resultinput.convert_restart_to_states import (
        convert_restart_to_states,
    )
    from pymrst.modules.deckformat.resultinput.convert_summary_to_well_sols import (
        convert_summary_to_well_sols,
    )
    G = init_eclipse_grid(deck)
    smry_prefix = fldr/"eclipse-simulation"/case_name
    if not Path(str(smry_prefix) + ".SMSPEC").exists():
        smry_prefix = fldr/"ECL.2014.2"/case_name
    ws, _ = convert_summary_to_well_sols(smry_prefix, unit)
    try:
        states, _ = convert_restart_to_states(smry_prefix, G, use_opm=False,
                                              consistentWellSols=False)
    except Exception as e:                                  # noqa: BLE001 - MRST's catch
        print(f"Restartfiles not pressent. Reported error:\n {e}")
        states = []
    return SimpleNamespace(wellSols=ws, states=states, G=G)


def get_norne_field_deck_opm_data_ad(
    root: str | Path | None = None, *, load_eclipse=True,
) -> tuple[NorneFieldDeck, NorneReferenceOutputs]:
    """MRST-compatible complete deck plus OPM and optional Eclipse outputs."""
    deck = read_norne_field_deck(root)
    opm = load_aquifer_eclipse_result_ad(norne_output_prefix(root, simulator="opm"))
    eclipse = None
    if load_eclipse:
        prefix = norne_output_prefix(root, simulator="eclipse")
        # The archived Eclipse result has summary/EGRID/INIT but no restart.
        # Reuse its summary with the OPM states/grid when restart is absent.
        try:
            eclipse = load_aquifer_eclipse_result_ad(prefix)
        except FileNotFoundError:
            from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import _summary
            summary = _summary(prefix)
            if summary is not None:
                eclipse = AquiferEclipseResult([], opm.grid, summary, prefix)
    return deck, NorneReferenceOutputs(opm, eclipse)


__all__ = ["NorneReferenceOutputs", "getNorneFieldDeckOPMData", "get_norne_field_deck_opm_data",
           "get_norne_field_deck_opm_data_ad"]


if __name__ == "__main__":
    import numpy as np

    DECK, OUTPUT = get_norne_field_deck_opm_data()

    def test_the_deck_is_norne_in_si():
        assert np.ravel(DECK["GRID"]["cartDims"]).tolist() == [46, 112, 22]
        assert len(DECK["SCHEDULE"].control) == 247
        first = np.asarray(DECK["SCHEDULE"].step["val"])[0]
        assert first == 691200.0009957701, first    # 8 days and a millisecond, as MRST
        print("ok test_the_deck_is_norne_in_si")

    def test_opm_output_on_the_unsplit_grid():
        assert len(OUTPUT.opm.wellSols) == 248 and len(OUTPUT.opm.states) == 65
        assert OUTPUT.opm.G.cells.num == 44927
        assert Path(OUTPUT.opm.location).parent.name == "flow_legacy"
        print("ok test_opm_output_on_the_unsplit_grid")

    def test_eclipse_output_without_restart_on_the_split_grid():
        assert len(OUTPUT.eclipse.wellSols) == 884 and len(OUTPUT.eclipse.states) == 0
        assert [g.cells.num for g in OUTPUT.eclipse.G] == [44915, 12]
        assert OUTPUT.ecl is None
        print("ok test_eclipse_output_without_restart_on_the_split_grid")

    test_the_deck_is_norne_in_si()
    test_opm_output_on_the_unsplit_grid()
    test_eclipse_output_without_restart_on_the_split_grid()
    print("\nAll tests passed!")
