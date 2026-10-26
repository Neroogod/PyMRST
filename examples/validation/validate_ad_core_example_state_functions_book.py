"""Validate MRST exampleStateFunctionsBook, the port in
`ad_core/examples/book_ii/example_state_functions_book.py`, with
`TutorialNumber`, `TutorialProduct` and `TutorialAdd`.

The reference is the script's console output in MRST 2024a, recorded with
`diary` (`dump_example_state_functions_book_reference.m`), without its figure
and TikZ output: the grouping's display, the empty and the filled cache, each
state function's message in the order of evaluation (all on the first `get`,
none on the second, only the missing ones once `XY` is cached), `G`, and the
dependency checks. The port's output must be the same text, line for line.
"""

import contextlib
import difflib
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    from pymrst.modules.ad_core.examples.book_ii.example_state_functions_book import (
        example_state_functions_book,
    )
    ref = (HERE/"ref_example_state_functions_book.txt").read_text().splitlines()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = example_state_functions_book()
    mine = out.getvalue().splitlines()
    strip = lambda lines: [l.rstrip() for l in lines]
    while ref and not ref[-1].strip():
        ref.pop()
    while mine and not mine[-1].strip():
        mine.pop()
    diff = list(difflib.unified_diff(strip(ref), strip(mine), "MRST", "PyMRST", lineterm=""))
    ok = not diff and result.G == 29 and result.ok is False
    print(f"{'ok  ' if not diff else 'FAIL'} console output identical ({len(ref)} lines)")
    print("\n".join(diff[:40]))
    print("\nexampleStateFunctionsBook matches MRST" if ok
          else "\nexampleStateFunctionsBook does NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
