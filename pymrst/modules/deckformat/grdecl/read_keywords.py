"""Keyword-array reader for GRDECL include files."""

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

ARRAY_KEYWORDS = {"COORD", "ZCORN", "ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ", "NTG"}


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
    return out


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


if __name__ == "__main__":
    root = Path(__file__).parents[5]
    deck_dir = root / "mrst-2024a-Geomech/examples/data"

    def test_read_keywords() -> None:
        data = read_keywords(deck_dir / "SPE1/SPE1.GRDECL")
        assert data["cartDims"] == (10, 10, 3)
        assert data["COORD"].size == 6 * 11 * 11
        assert data["ZCORN"].size == 8 * 300
        print("ok test_read_keywords")

    test_read_keywords()
    print("\nAll tests passed!")

