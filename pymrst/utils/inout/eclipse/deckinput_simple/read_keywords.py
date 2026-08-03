"""Keyword-array reader for GRDECL include files."""

import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
    apply_operator_simple,
)

# Per-cell (or per-node) arrays a GRID section may declare. Corner-point decks
# use COORD/ZCORN; block-centred decks such as the Egg model use DX/DY/DZ and
# TOPS instead and carry no corner-point data at all, so a reader that knows
# only the first group cannot read them.
ARRAY_KEYWORDS = {
    # corner-point geometry
    "COORD", "ZCORN",
    # block-centred geometry
    "DX", "DY", "DZ", "DXV", "DYV", "DZV", "TOPS", "DEPTHZ",
    # petrophysics
    "ACTNUM", "PORO", "NTG", "PERMX", "PERMY", "PERMZ",
    "PERMXY", "PERMXZ", "PERMYX", "PERMYZ", "PERMZX", "PERMZY",
    # pore-volume and transmissibility modifiers
    "MULTPV", "MULTX", "MULTY", "MULTZ", "PORV",
}

# Keywords that modify an array already read, rather than declaring one. These
# used to be skipped silently, which meant a deck saying "MULTIPLY PERMX 0.1"
# was read with PERMX unscaled and nothing said so - see apply_operator_simple.
OPERATOR_KEYWORDS = ("ADD", "COPY", "EQUALS", "MAXVALUE", "MINVALUE",
                     "MULTIPLY")


def read_keywords(path, keywords: Optional[Set[str]] = None) -> Dict:
    """Parse grid/petrophysics keywords from a GRDECL-style file."""
    return read_keywords_text(Path(path).read_text(), keywords)


def read_keywords_text(text: str, keywords: Optional[Set[str]] = None) -> Dict:
    """Same as `read_keywords`, on deck text already in memory.

    A GRID section usually arrives split across INCLUDE files, and an operator
    such as `COPY` needs the `SPECGRID` from the parent deck to know the box
    dimensions. Reading each file separately loses that context, so the caller
    splices the text first and parses it once.
    """
    known = ARRAY_KEYWORDS if keywords is None else set(keywords)
    tokens = _tokens_from_text(text)
    out: Dict = {}
    index = 0
    while index < len(tokens):
        keyword = tokens[index].upper()
        index += 1
        if keyword == "SPECGRID":
            out["cartDims"] = tuple(int(float(tokens[index + offset])) for offset in range(3))
            index = _skip_record(tokens, index)
        elif keyword in known:
            out[keyword], index = _read_array(tokens, index)
        elif keyword in OPERATOR_KEYWORDS:
            records, index = _read_records(tokens, index)
            if records:
                dims = out.get("cartDims")
                if dims is None:
                    raise ValueError(
                        f"{keyword} modifies an array over a box of cells, but "
                        f"the deck has no SPECGRID to give the grid dimensions")
                out = apply_operator_simple(out, records, keyword, dims)
    return out


def _read_records(tokens: List[str], index: int):
    """Records of an operator keyword, up to its closing '/'."""
    records: List[List[str]] = []
    current: List[str] = []
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if token != "/":
            current.append(token)
            continue
        if not current:          # the empty record closing the keyword
            break
        records.append(current)
        current = []
    return records, index


def _tokens(path: Path) -> List[str]:
    return _tokens_from_text(path.read_text())


def _tokens_from_text(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("--")[0].strip()
        if not line:
            continue
        # Commas separate values in ECLIPSE, and a slash is its own token.
        fields = line.replace("/", " / ").replace(",", " ").split()
        # Production decks annotate a record after its terminating slash, with
        # no comment marker: Norne writes
        #     'MULTZ' 0.005 6 14 11 18 10 10 /  C-3H
        # where `C-3H` names the fault the line refers to. Carrying that token
        # forward prepends it to the next record, which then has one field too
        # many and is rejected. The main deck tokenizer already drops such
        # text; this reader needs the same rule.
        for position, token in enumerate(fields):
            if token == "/" and position + 1 < len(fields):
                following = fields[position + 1]
                if following[:1].isalpha():
                    fields = fields[:position + 1]
                    break
        out.extend(fields)
    return out


def _skip_record(tokens: List[str], index: int) -> int:
    while index < len(tokens) and tokens[index] != "/":
        index += 1
    return index + 1


def _read_array(tokens: List[str], index: int) -> tuple[np.ndarray, int]:
    values: List[float] = []
    while index < len(tokens) and tokens[index] != "/":
        token = tokens[index]
        if "*" in token:
            count, _, value = token.partition("*")
            values.extend([float(value)] * int(count))
        else:
            values.append(float(token))
        index += 1
    return np.array(values), index + 1


def _mrst_data_dir():
    """Find MRST's example data by searching upward, not by counting levels.

    These files moved from `modules/deckformat/grdecl/` to
    `utils/inout/eclipse/deckinput_simple/`, which changed their depth and
    silently broke a hard-coded `parents[5]`. Searching for the sibling tree
    survives the next move too.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / "mrst-2024a-Geomech" / "examples" / "data"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "could not find mrst-2024a-Geomech/examples/data above "
        + str(pathlib.Path(__file__).resolve()))


if __name__ == "__main__":
    deck_dir = _mrst_data_dir()

    def test_read_keywords() -> None:
        data = read_keywords(deck_dir / "SPE1/SPE1.GRDECL")
        assert data["cartDims"] == (10, 10, 3)
        assert data["COORD"].size == 6 * 11 * 11
        assert data["ZCORN"].size == 8 * 300
        print("ok test_read_keywords")

    test_read_keywords()
    print("\nAll tests passed!")

