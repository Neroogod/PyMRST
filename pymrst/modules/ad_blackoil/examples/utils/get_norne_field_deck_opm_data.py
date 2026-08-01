"""Load the complete Norne deck and OPM/ECLIPSE reference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import (
    AquiferEclipseResult, load_aquifer_eclipse_result,
)
from pymrst.modules.deckformat.norne import (
    NorneFieldDeck, norne_output_prefix, read_norne_field_deck,
)


@dataclass
class NorneReferenceOutputs:
    opm: AquiferEclipseResult
    eclipse: AquiferEclipseResult | None


def get_norne_field_deck_opm_data(
    root: str | Path | None = None, *, load_eclipse=True,
) -> tuple[NorneFieldDeck, NorneReferenceOutputs]:
    """MRST-compatible complete deck plus OPM and optional Eclipse outputs."""
    deck = read_norne_field_deck(root)
    opm = load_aquifer_eclipse_result(norne_output_prefix(root, simulator="opm"))
    eclipse = None
    if load_eclipse:
        prefix = norne_output_prefix(root, simulator="eclipse")
        # The archived Eclipse result has summary/EGRID/INIT but no restart.
        # Reuse its summary with the OPM states/grid when restart is absent.
        try:
            eclipse = load_aquifer_eclipse_result(prefix)
        except FileNotFoundError:
            from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import _summary
            summary = _summary(prefix)
            if summary is not None:
                eclipse = AquiferEclipseResult([], opm.grid, summary, prefix)
    return deck, NorneReferenceOutputs(opm, eclipse)


__all__ = ["NorneReferenceOutputs", "get_norne_field_deck_opm_data"]
