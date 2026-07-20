"""Port placeholder for MRST ``getNorneFieldDeckOPMData.m``."""

from __future__ import annotations

from pathlib import Path


def get_norne_field_deck_opm_data(root: str | Path | None = None) -> Path:
    root = Path(root or Path.cwd())
    candidates = list(root.rglob("NORNE_ATW2013.DATA")) if root.exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "NORNE_ATW2013.DATA was not found. PyMRST has geometry helpers for "
        "Norne, but this ad-blackoil example requires the full OPM deck."
    )


__all__ = ["get_norne_field_deck_opm_data"]
