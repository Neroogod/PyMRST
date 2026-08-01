"""Read one deck record as text (MRST ``.../private/readRecordString.m``).

A record runs until a `/`, possibly across several lines. This joins those
lines, strips `--` and `#` comments, cuts the text at the terminator and
removes quote characters, returning the record as a single string.

It is the lowest-level primitive of the whole deck reader - 15 files in MRST
call it - and the three things it does are each easy to get wrong alone:

- **comments are stripped before the terminator is looked for**, so a `/`
  inside a comment does not end the record;
- **only the first `/` counts**; anything after it on the line belongs to the
  next record and is discarded here;
- **quotes are removed entirely**, not just at the ends, which is why
  `'PERMX'` arrives as `PERMX` and a quoted string containing spaces would not
  survive. That is MRST's behaviour and the format does not use such strings.

Takes an iterator of lines rather than a file handle, so it composes with any
source; the iterator is left positioned after the record, which is what makes
successive calls work.
"""

import re
from typing import Iterable, Iterator, Optional

COMMENT = re.compile(r"(--|#).*$")


def read_record_string(lines: Iterator[str]) -> Optional[str]:
    """Return the next record as one string, or ``None`` at end of input."""
    collected = []
    for raw in lines:
        text = COMMENT.sub("", raw)
        collected.append(text)
        if "/" in text:
            joined = " ".join(collected)
            return joined[:joined.index("/")].replace("'", "").replace('"', "")
    if not collected:
        return None
    # Input ended without a terminator: return what there was.
    return " ".join(collected).replace("'", "").replace('"', "")


readRecordString = read_record_string


if __name__ == "__main__":

    def test_a_single_line_record() -> None:
        got = read_record_string(iter(["'PERMX' 0.1 /"]))
        assert got.split() == ["PERMX", "0.1"], got
        print("ok test_a_single_line_record")

    def test_a_record_spanning_lines() -> None:
        got = read_record_string(iter(["'PERMX' 0.1", " 1 10", " 1 10 1 5 /"]))
        assert got.split() == ["PERMX", "0.1", "1", "10", "1", "10", "1", "5"]
        print("ok test_a_record_spanning_lines")

    def test_comments_are_stripped() -> None:
        got = read_record_string(iter(["'PERMX' 0.1 -- scale it /"]))
        assert got is None or "scale" not in got
        print("ok test_comments_are_stripped")

    def test_a_slash_inside_a_comment_does_not_terminate() -> None:
        """Comments go first, so a '/' in one is invisible."""
        lines = iter(["'PERMX' -- units m/s", " 0.1 /"])
        got = read_record_string(lines)
        assert got.split() == ["PERMX", "0.1"], got
        print("ok test_a_slash_inside_a_comment_does_not_terminate")

    def test_a_hash_comment_is_stripped_too() -> None:
        got = read_record_string(iter(["'PERMX' 0.1 # note", "/"]))
        assert got.split() == ["PERMX", "0.1"], got
        print("ok test_a_hash_comment_is_stripped_too")

    def test_only_the_first_slash_counts() -> None:
        got = read_record_string(iter(["'A' 1 / 'B' 2 /"]))
        assert got.split() == ["A", "1"], got
        print("ok test_only_the_first_slash_counts")

    def test_quotes_are_removed_everywhere() -> None:
        got = read_record_string(iter(["'PERMX' 'PERMY' /"]))
        assert "'" not in got and got.split() == ["PERMX", "PERMY"]
        print("ok test_quotes_are_removed_everywhere")

    def test_successive_calls_walk_the_input() -> None:
        """The iterator is left positioned for the next record."""
        lines = iter(["'A' 1 /", "'B' 2 /", "'C' 3 /"])
        assert read_record_string(lines).split() == ["A", "1"]
        assert read_record_string(lines).split() == ["B", "2"]
        assert read_record_string(lines).split() == ["C", "3"]
        print("ok test_successive_calls_walk_the_input")

    def test_end_of_input_gives_none() -> None:
        assert read_record_string(iter([])) is None
        print("ok test_end_of_input_gives_none")

    def test_an_unterminated_record_returns_what_there_was() -> None:
        got = read_record_string(iter(["'PERMX' 0.1"]))
        assert got.split() == ["PERMX", "0.1"], got
        print("ok test_an_unterminated_record_returns_what_there_was")

    def test_an_empty_record() -> None:
        assert read_record_string(iter(["/"])).strip() == ""
        print("ok test_an_empty_record")

    test_a_single_line_record()
    test_a_record_spanning_lines()
    test_comments_are_stripped()
    test_a_slash_inside_a_comment_does_not_terminate()
    test_a_hash_comment_is_stripped_too()
    test_only_the_first_slash_counts()
    test_quotes_are_removed_everywhere()
    test_successive_calls_walk_the_input()
    test_end_of_input_gives_none()
    test_an_unterminated_record_returns_what_there_was()
    test_an_empty_record()
    print("\nAll tests passed!")
