"""ECLIPSE record reading (MRST ``deckinput/readDefaultedRecord.m``).

A deck is a stream of keywords, each followed by `/`-terminated records of
whitespace-separated fields. Two conventions make the format awkward and are
the whole reason this module exists:

- `n*` in a record means "the next n fields take their default", so a field's
  position depends on what came before it;
- an all-default record terminates a multi-record keyword.

Both are handled here so the keyword readers can work with plain lists.

The reader is a cursor over a token stream rather than a file handle, which
makes it testable without touching disk and lets the GRDECL reader share it.
"""

import re
from typing import Any, List, Optional, Sequence, Tuple

DEFAULT = re.compile(r"^(\d+)\*$")
REPEAT = re.compile(r"^(\d+)\*(.+)$")

# MRST's rule, from getEclipseKeyword.m: "An ECLIPSE keyword consists of at
# least one, and at most eight, upper-case English (ASCII) characters at the
# beginning of a line. Any file data from column 9 up to (and including) the
# newline are considered comments and ignored."
KEYWORD = re.compile(r"^([A-Z][A-Z0-9]{0,7})")


class DeckCursor:
    """A position in a deck's token stream."""

    def __init__(self, tokens: Sequence[str], index: int = 0,
                 lines: Optional[Sequence[int]] = None):
        self.tokens = list(tokens)
        self.index = index
        self.lines = list(lines) if lines is not None else None
        # Where the keyword returned by the last next_keyword() sat, so a
        # caller that reads one keyword too many can put it back.
        self.keyword_index = index

    @classmethod
    def from_text(cls, text: str) -> "DeckCursor":
        """Build a cursor over deck text, keeping track of line boundaries."""
        tokens, lines = tokenize_with_lines(text)
        return cls(tokens, lines=lines)

    def __bool__(self) -> bool:
        return self.index < len(self.tokens)

    def peek(self) -> Optional[str]:
        return self.tokens[self.index] if self else None

    def next(self) -> Optional[str]:
        token = self.peek()
        if token is not None:
            self.index += 1
        return token

    def read_record_tokens(self) -> List[str]:
        """Everything up to and including the next ``/``, without the slash."""
        out: List[str] = []
        while self:
            token = self.next()
            if token == "/":
                return out
            if token.endswith("/") and len(token) > 1:
                out.append(token[:-1])
                return out
            out.append(token)
        return out

    def _starts_a_line(self, index: int) -> bool:
        if self.lines is None:
            return True                     # no line information; accept any
        return index == 0 or self.lines[index] != self.lines[index - 1]

    def _skip_rest_of_line(self) -> None:
        """Drop the trailing commentary ECLIPSE allows after a keyword."""
        if self.lines is None or self.index == 0:
            return
        line = self.lines[self.index - 1]
        while self and self.lines[self.index] == line:
            self.index += 1

    def next_keyword(self) -> Optional[str]:
        """Advance to the next keyword, skipping anything that is not one.

        Data lines never match the keyword pattern, so an unrecognised
        keyword's records are stepped over for free. That is what makes it
        safe to ignore a keyword this reader does not know: the alternative,
        guessing how many records it owns, silently swallows the *next*
        keyword's table when the guess is wrong.
        """
        while self:
            index = self.index
            token = self.tokens[index]
            match = KEYWORD.match(token.rstrip("/"))
            if match and self._starts_a_line(index) and token[0].isupper():
                self.keyword_index = index
                self.index = index + 1
                self._skip_rest_of_line()
                return match.group(1)
            self.index += 1
        return None

    def rewind_keyword(self) -> None:
        """Put back the keyword returned by the last ``next_keyword``."""
        self.index = self.keyword_index


def tokenize_with_lines(text: str) -> Tuple[List[str], List[int]]:
    """Tokenize, returning each token's line number alongside it."""
    tokens: List[str] = []
    lines: List[int] = []
    for number, line in enumerate(text.splitlines()):
        line = line.split("--", 1)[0]
        if not line.strip():
            continue
        # ECLIPSE treats a comma as a separator, exactly like whitespace, so a
        # table row may be written `0.1000,  0.0000e+00,  8.0e-01  0`. Matching
        # bare `\S+` glues the comma onto the number and the value then fails to
        # parse. Quoted strings keep their commas because the quoted
        # alternatives are tried first.
        matches = list(re.finditer(r"'[^']*'|\"[^\"]*\"|[^\s,]+", line))
        for position, match in enumerate(matches):
            token = match.group(0)
            # a trailing slash is its own token, however it is written
            if len(token) > 1 and token.endswith("/") and not token.startswith("'"):
                tokens.extend((token[:-1], "/"))
                lines.extend((number, number))
            else:
                tokens.append(token)
                lines.append(number)
            # Free-form text after a record-ending slash is commentary in
            # many production decks (for example ``/ Justert 22/7`` in
            # Norne). Preserve compact numeric records such as ``1 / 2 /``.
            ended = token == "/" or (
                len(token) > 1 and token.endswith("/")
                and not token.startswith(("'", '"'))
            )
            if ended and position + 1 < len(matches):
                following = matches[position + 1].group(0)
                if re.match(r"[A-Za-z]", following):
                    break
    return tokens, lines


def tokenize(text: str) -> List[str]:
    """Split deck text into tokens, dropping comments and keeping quotes intact."""
    return tokenize_with_lines(text)[0]


def strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1].strip()
    return token


def expand_defaults(fields: Sequence[str], template: Sequence[Any]) -> List[Any]:
    """Fill ``template`` from ``fields``, honouring ``n*`` default runs.

    `n*` consumes n slots without supplying a value, so a later field lands
    further along than its position in the record suggests. Getting this wrong
    silently shifts every subsequent value.
    """
    record = list(template)
    if not fields:
        return record

    position = 0
    for field in fields:
        run = DEFAULT.match(field)
        if run:
            position += int(run.group(1))
            continue
        if position < len(record):
            record[position] = strip_quotes(field)
        position += 1
    return record


def read_defaulted_record(cursor: DeckCursor,
                          template: Sequence[Any]) -> List[Any]:
    """Read one record, returning ``template`` with its non-default fields set."""
    return expand_defaults(cursor.read_record_tokens(), template)


def read_defaulted_kw(cursor: DeckCursor, template: Sequence[Any],
                      n_rec: Optional[int] = None) -> List[List[Any]]:
    """Read records until an all-default one, or until ``n_rec`` are read.

    MRST detects the terminator by comparing against the template, so a record
    that happens to restate every default also terminates the keyword. That is
    the format's rule, not an approximation.
    """
    records: List[List[Any]] = []
    while cursor:
        record = read_defaulted_record(cursor, template)
        if list(record) == list(template):
            break
        records.append(record)
        if n_rec is not None and len(records) >= n_rec:
            break
    return records


def read_fixed_num_records(cursor: DeckCursor, template: Sequence[Any],
                           n_rec: int) -> List[List[Any]]:
    """Read exactly ``n_rec`` records (MRST ``readFixedNumRecords.m``).

    Unlike `read_defaulted_kw`, an all-default record does not terminate the
    keyword - it is a legitimate record that happens to take every default.
    PVTW and ROCK are read this way, so a region that defaults everything
    still occupies its slot instead of truncating the table.
    """
    return [read_defaulted_record(cursor, template) for _ in range(n_rec)]


readDefaultedRecord = read_defaulted_record
readDefaultedKW = read_defaulted_kw
readFixedNumRecords = read_fixed_num_records


if __name__ == "__main__":

    def test_tokenize_drops_comments_and_splits_slashes() -> None:
        tokens = tokenize("SWOF -- a comment\n 0.1 0.2 /\n/\n")
        assert tokens == ["SWOF", "0.1", "0.2", "/", "/"], tokens
        print("ok test_tokenize_drops_comments_and_splits_slashes")

    def test_commas_separate_values_like_whitespace() -> None:
        """ECLIPSE allows a comma as a separator, and real decks use it.

        The Egg model writes `0.1000,  0.0000e+00,  8.0e-01  0` in SWOF.
        Matching `\\S+` glued the comma onto the number, so the whole deck
        failed to load on `float('0.1000,')`. Quoted text keeps its commas,
        because the quoted alternatives are tried first.
        """
        assert tokenize("0.1000,  0.0000e+00,  8.0e-01  0") == [
            "0.1000", "0.0000e+00", "8.0e-01", "0"]
        assert tokenize("1,2,3") == ["1", "2", "3"]
        assert tokenize("'A,B' 4") == ["'A,B'", "4"]
        print("ok test_commas_separate_values_like_whitespace")

    def test_tokenize_keeps_quoted_strings_whole() -> None:
        tokens = tokenize("WELSPECS\n 'PROD A' 'G1' 10 10 /\n")
        assert "'PROD A'" in tokens, tokens
        print("ok test_tokenize_keeps_quoted_strings_whole")

    def test_tokenize_drops_free_text_after_record_slash() -> None:
        tokens = tokenize("859.5 1033.0 0.854 / Justert 22/7\n")
        assert tokens == ["859.5", "1033.0", "0.854", "/"], tokens
        print("ok test_tokenize_drops_free_text_after_record_slash")

    def test_record_stops_at_the_slash() -> None:
        cursor = DeckCursor(tokenize("1 2 3 / 4 5 /"))
        assert cursor.read_record_tokens() == ["1", "2", "3"]
        assert cursor.read_record_tokens() == ["4", "5"]
        print("ok test_record_stops_at_the_slash")

    def test_defaults_shift_later_fields() -> None:
        """`2*` skips two slots, so 9 belongs in position 3, not position 1."""
        out = expand_defaults(["2*", "9"], [None] * 4)
        assert out == [None, None, "9", None], out
        print("ok test_defaults_shift_later_fields")

    def test_template_supplies_untouched_fields() -> None:
        out = expand_defaults(["7"], ["a", "b", "c"])
        assert out == ["7", "b", "c"], out
        print("ok test_template_supplies_untouched_fields")

    def test_quotes_are_stripped_from_values() -> None:
        out = expand_defaults(["'WELL 1'"], [None])
        assert out == ["WELL 1"], out
        print("ok test_quotes_are_stripped_from_values")

    def test_multi_record_keyword_stops_on_an_empty_record() -> None:
        cursor = DeckCursor(tokenize("1 /\n2 /\n/\n3 /\n"))
        records = read_defaulted_kw(cursor, [None, None])
        assert records == [["1", None], ["2", None]], records
        # the cursor is left positioned after the terminator
        assert cursor.read_record_tokens() == ["3"]
        print("ok test_multi_record_keyword_stops_on_an_empty_record")

    def test_record_limit_is_respected() -> None:
        cursor = DeckCursor(tokenize("1 /\n2 /\n3 /\n"))
        records = read_defaulted_kw(cursor, [None], n_rec=2)
        assert len(records) == 2, records
        print("ok test_record_limit_is_respected")

    def test_all_default_record_terminates_even_with_stars() -> None:
        cursor = DeckCursor(tokenize("3* /\n"))
        assert read_defaulted_kw(cursor, [None, None, None]) == []
        print("ok test_all_default_record_terminates_even_with_stars")

    def test_keyword_ignores_trailing_decoration() -> None:
        """`PROPS ====` is the keyword PROPS; column 9 on is commentary."""
        cursor = DeckCursor.from_text("PROPS  =========\nDENSITY\n900 /\n")
        assert cursor.next_keyword() == "PROPS", cursor.tokens
        assert cursor.next_keyword() == "DENSITY"
        assert cursor.read_record_tokens() == ["900"]
        print("ok test_keyword_ignores_trailing_decoration")

    def test_unknown_keyword_data_is_skipped_not_consumed() -> None:
        """Numbers are not keywords, so a stray table cannot hide the next one."""
        cursor = DeckCursor.from_text("MYSTERY\n1 2 3 /\n4 5 6 /\nDENSITY\n900 /\n")
        assert cursor.next_keyword() == "MYSTERY"
        assert cursor.next_keyword() == "DENSITY"
        print("ok test_unknown_keyword_data_is_skipped_not_consumed")

    def test_keyword_must_start_a_line() -> None:
        cursor = DeckCursor.from_text("DENSITY\n900 1000 0.8 /\n")
        assert cursor.next_keyword() == "DENSITY"
        cursor.read_record_tokens()
        assert cursor.next_keyword() is None
        print("ok test_keyword_must_start_a_line")

    def test_a_rewound_keyword_is_read_again() -> None:
        cursor = DeckCursor.from_text("DENSITY\n900 /\nREGIONS\n")
        cursor.next_keyword()
        cursor.read_record_tokens()
        assert cursor.next_keyword() == "REGIONS"
        cursor.rewind_keyword()
        assert cursor.next_keyword() == "REGIONS"
        print("ok test_a_rewound_keyword_is_read_again")

    def test_keyword_strips_a_terminating_slash() -> None:
        cursor = DeckCursor.from_text("ENDBOX/\nDENSITY\n900 /\n")
        assert cursor.next_keyword() == "ENDBOX"
        print("ok test_keyword_strips_a_terminating_slash")

    test_keyword_ignores_trailing_decoration()
    test_unknown_keyword_data_is_skipped_not_consumed()
    test_keyword_must_start_a_line()
    test_a_rewound_keyword_is_read_again()
    test_keyword_strips_a_terminating_slash()
    test_tokenize_drops_comments_and_splits_slashes()
    test_tokenize_drops_free_text_after_record_slash()
    test_commas_separate_values_like_whitespace()
    test_tokenize_keeps_quoted_strings_whole()
    test_record_stops_at_the_slash()
    test_defaults_shift_later_fields()
    test_template_supplies_untouched_fields()
    test_quotes_are_stripped_from_values()
    test_multi_record_keyword_stops_on_an_empty_record()
    test_record_limit_is_respected()
    def test_fixed_num_records_does_not_stop_on_an_all_default_record() -> None:
        """PVTW's second region may legitimately take every default."""
        cursor = DeckCursor.from_text("300 1.0 /\n2* /\n900 2.0 /\n")
        records = read_fixed_num_records(cursor, [None, None], 3)
        assert len(records) == 3, records
        assert records[1] == [None, None], records[1]
        assert records[2] == ["900", "2.0"], records[2]
        print("ok test_fixed_num_records_does_not_stop_on_an_all_default_record")

    test_all_default_record_terminates_even_with_stars()
    test_fixed_num_records_does_not_stop_on_an_all_default_record()
    print("\nAll tests passed!")
