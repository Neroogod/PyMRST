"""What this package will and will not do with a given ECLIPSE file (M2.6).

`keyword_support()` states what the reader understands *in general*. This
answers the question a user actually has, which is narrower and more useful:
**for this file on my disk, what gets used and what gets dropped?**

The distinction matters because `read_eclipse_case` is permissive by design. It
reads what it recognises and carries the rest through as raw arrays that
nothing consumes, so a case built from a file full of unrecognised keywords
loads cleanly and is quietly not the case on disk. That is the right behaviour
for a reader - refusing a whole model over one unknown keyword would make it
useless - but it is only safe if the omissions are visible. This makes them
visible.

Nothing here reads a case or builds a grid. It inspects the record headers
only, so it is cheap and works on files that `read_eclipse_case` would reject.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pymrst.research_studio.eclipse.case import keyword_support
from pymrst.research_studio.eclipse.records import read_records

# Keywords this package knows it does not implement, and what they carry, so a
# report can say *what* was lost rather than only that something was.
_KNOWN_GAPS = {
    "NNCHEAD": "non-neighbour connection header",
    "LGR": "local grid refinement",
    "LGRHEADI": "local grid refinement header",
    "LGRHEADQ": "local grid refinement header",
    "LGRSGONE": "local grid refinement",
    "COARSEN": "grid coarsening",
}

# Keywords this package *can* read, through a call the case reader does not
# make. Distinct from both "used" and "not implemented", and the distinction is
# the honest one: calling these used would claim the imported case contains
# them, and calling them unimplemented would hide that the data is reachable.
_AVAILABLE_SEPARATELY = {
    "NNC1": "non-neighbour connections (cell pairs), via read_nnc",
    "NNC2": "non-neighbour connections (cell pairs), via read_nnc",
    "TRANNNC": "non-neighbour transmissibilities, via read_nnc",
}

# Classifications, worst last so a report reads in increasing order of concern.
USED = "used"
AVAILABLE_SEPARATELY = "available_separately"
READ_BUT_UNUSED = "read_but_unused"
NOT_IMPLEMENTED = "not_implemented"
UNRECOGNISED = "unrecognised"


@dataclass(frozen=True, slots=True)
class KeywordCoverage:
    """What one file contains, classified."""

    path: Path
    counts: dict[str, int]
    classification: dict[str, str]

    def of(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.classification.items() if v == kind))

    @property
    def complete(self) -> bool:
        """True when nothing carrying model data is dropped.

        `available_separately` *is* counted: `read_nnc` can reach those
        connections, but `read_eclipse_case` does not fold them in, so as far
        as the imported case is concerned they are absent - which is the thing
        this property is asked about.

        `read_but_unused` is deliberately excluded. It holds file metadata -
        FILEHEAD, INTEHEAD - which contributes no physics, so counting it would
        mark every valid file incomplete, including the ones this package
        writes itself. A flag that fires on a correct file is the failure the
        transmissibility check already had to be rewritten to avoid: it trains
        the reader to ignore it. Metadata still appears in `describe_gaps`,
        where it is information rather than an alarm.
        """
        return not (self.of(NOT_IMPLEMENTED) or self.of(UNRECOGNISED)
                    or self.of(AVAILABLE_SEPARATELY))

    def __str__(self) -> str:
        parts = [f"{self.path.name}: {len(self.classification)} keyword(s)"]
        for kind in (USED, READ_BUT_UNUSED, AVAILABLE_SEPARATELY,
                     NOT_IMPLEMENTED, UNRECOGNISED):
            names = self.of(kind)
            if names:
                parts.append(f"  {kind}: {', '.join(names)}")
        return "\n".join(parts)


def _supported_sets() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    support = keyword_support()
    used = frozenset(
        name.upper()
        for group in ("grid", "rock", "state", "structural")
        for name in support.get(group, ())
    )
    unused = frozenset(n.upper() for n in support.get("read_but_unused", ()))
    # `not_implemented` in keyword_support is prose ("NNC", "LGR", ...), not
    # keywords, so the concrete record names come from _KNOWN_GAPS instead.
    return used, unused, frozenset(_KNOWN_GAPS)


def classify_keyword(name: str) -> str:
    """Where one keyword falls, independent of any file."""
    used, unused, gaps = _supported_sets()
    key = name.upper()
    if key in used:
        return USED
    if key in _AVAILABLE_SEPARATELY:
        return AVAILABLE_SEPARATELY
    if key in unused:
        return READ_BUT_UNUSED
    if key in gaps:
        return NOT_IMPLEMENTED
    return UNRECOGNISED


def keyword_coverage(path: str | Path) -> KeywordCoverage:
    """Classify every keyword in one ECLIPSE output file."""
    target = Path(path)
    counts: Counter[str] = Counter()
    for name, value_type, _ in read_records(target):
        if value_type == "MESS":
            # A marker such as ENDGRID carries no data; counting it as a
            # dropped keyword would report a gap that does not exist.
            continue
        counts[name.upper()] += 1
    return KeywordCoverage(
        target, dict(counts),
        {name: classify_keyword(name) for name in counts})


def case_coverage(prefix: str | Path,
                  suffixes: Iterable[str] = (".EGRID", ".INIT", ".UNRST"),
                  ) -> dict[str, KeywordCoverage]:
    """Coverage for each file of a case that exists.

    Missing files are omitted rather than reported as empty: a case with no
    INIT has no rock, which `read_eclipse_case` already states in its `source`
    mapping, and repeating it here as "zero keywords covered" would read as a
    failure rather than an absence.
    """
    stem = Path(prefix)
    reports = {}
    for suffix in suffixes:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            reports[suffix.lstrip(".")] = keyword_coverage(candidate)
    if not reports:
        raise FileNotFoundError(
            f"no ECLIPSE output found for {stem}; looked for "
            f"{', '.join(suffixes)}")
    return reports


def describe_gaps(coverage: KeywordCoverage) -> tuple[str, ...]:
    """One line per dropped keyword, saying what is lost.

    Written for a person deciding whether to trust an imported case, so the
    named gaps carry their meaning - "NNC1: non-neighbour connections (cell
    pairs)" rather than a bare keyword the reader would have to look up.
    """
    lines = []
    for name in coverage.of(AVAILABLE_SEPARATELY):
        lines.append(f"{name}: {_AVAILABLE_SEPARATELY[name]} - readable, but "
                     f"read_eclipse_case does not fold these into the case, so "
                     f"the imported model is sealed across them")
    for name in coverage.of(NOT_IMPLEMENTED):
        lines.append(f"{name}: {_KNOWN_GAPS[name]} - not implemented, so these "
                     f"connections are absent from the imported case")
    for name in coverage.of(UNRECOGNISED):
        lines.append(f"{name}: not recognised by this reader; carried through "
                     f"as a raw array and used by nothing")
    for name in coverage.of(READ_BUT_UNUSED):
        lines.append(f"{name}: parsed but not used to build the case")
    return tuple(lines)


__all__ = ["AVAILABLE_SEPARATELY", "KeywordCoverage", "NOT_IMPLEMENTED",
           "READ_BUT_UNUSED",
           "UNRECOGNISED", "USED", "case_coverage", "classify_keyword",
           "describe_gaps", "keyword_coverage"]
