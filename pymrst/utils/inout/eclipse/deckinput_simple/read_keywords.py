"""Keyword-array reader for GRDECL include files."""

import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
    apply_operator_simple,
)

ARRAY_KEYWORDS = {"COORD", "ZCORN", "ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ", "NTG"}

# Keywords that modify an array already read, rather than declaring one. These
# used to be skipped silently, which meant a deck saying "MULTIPLY PERMX 0.1"
# was read with PERMX unscaled and nothing said so - see apply_operator_simple.
OPERATOR_KEYWORDS = ("ADD", "COPY", "EQUALS", "MAXVALUE", "MINVALUE",
                     "MULTIPLY")


def read_keywords(path, keywords: Optional[Set[str]] = None) -> Dict:
    """Parse grid/petrophysics keywords from a GRDECL-style file."""
    known = ARRAY_KEYWORDS if keywords is None else set(keywords)
    tokens = _tokens(Path(path))
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
    out: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("--")[0].strip()
        if line:
            out.extend(line.replace("/", " / ").split())
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

