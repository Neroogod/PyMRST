"""Read a deck's SCHEDULE section (MRST ``deckinput/readSCHEDULE.m``).

`read_eclipse_deck` used to list SCHEDULE among its skipped sections, so a
deck-driven case had to have its wells and time steps supplied by hand. This
closes that: the section becomes `schedule.control` - one entry per control
period - and `schedule.step`, which says how long each step is and which
period governs it.

The whole of the file is one rule, and it is the reason a naive reader gets
the answer wrong:

    a well keyword opens a new control period; a time step closes it.

Consecutive well keywords therefore belong to *one* period, however many there
are, and a run of `TSTEP`s that follows them all points at that same period.
Opening a period per keyword instead would give the Egg model twelve control
periods where it has one, and every step after the first would run against a
partially specified well set.

The period a keyword opens starts as a copy of the one before it, because
ECLIPSE carries a well's declaration forward until the deck restates it - see
`next_control`. A period that started empty would shut in every well the deck
stopped mentioning.

Values are the deck's own; `convert_deck_units` puts them in SI.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pymrst.modules.deckformat.deckinput.read_well_kw import (
    REPORT_KEYWORDS, SUPPORTED, next_control, read_well_kw,
)
from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
    DeckCursor, read_defaulted_record,
)

#: Keywords that belong to a control period. MRST's list, restricted to the
#: ones `read_well_kw` implements; the rest are reported as unhandled.
WELL_KEYWORDS = tuple(SUPPORTED)

#: Keywords that advance time.
STEP_KEYWORDS = ("TSTEP", "DATES")

#: Section headers, so a SCHEDULE reader handed a mid-deck cursor stops.
_SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS", "SOLUTION",
             "SUMMARY")

_REPEAT = re.compile(r"^(\d+)\*(.*)$")

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JLY": 7, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11,
           "DEC": 12}


def read_schedule(cursor: DeckCursor, deck: Dict[str, Any]) -> Dict[str, Any]:
    """Read SCHEDULE from ``cursor`` into ``deck``, and return the deck."""
    schedule = deck["SCHEDULE"]
    start = deck.get("RUNSPEC", {}).get("START", date(1983, 1, 1))
    unhandled = deck.setdefault("UnhandledKeywords", {}).setdefault(
        "SCHEDULE", [])

    control: Dict[str, Any] = next_control()
    open_period = False        # MRST's `def_ctrl`
    number = 0                 # MRST's `cno`, 1-based once a period opens

    while True:
        keyword = cursor.next_keyword()
        if keyword is None or keyword == "END":
            break
        if keyword in _SECTIONS:
            cursor.rewind_keyword()
            break

        if keyword in WELL_KEYWORDS:
            if not open_period:
                open_period, number = True, number + 1
                control = next_control(control)
            read_well_kw(cursor, control, keyword)

        elif keyword in STEP_KEYWORDS:
            if open_period:
                open_period = False
                schedule.control.append(control)
            values = (_read_tstep(cursor) if keyword == "TSTEP"
                      else _read_dates(cursor, start, schedule.step["val"]))
            schedule.step["val"].extend(values)
            schedule.step["control"].extend([number]*len(values))

        elif keyword == "DRSDT":
            if not open_period:
                open_period, number = True, number + 1
                control = next_control(control)
            record = read_defaulted_record(cursor, ("NaN", "ALL"))
            control["DRSDT"] = [_number(record[0]), str(record[1])]

        elif keyword == "VAPPARS":
            if not open_period:
                open_period, number = True, number + 1
                control = next_control(control)
            record = read_defaulted_record(cursor, ("0", "0"))
            control["VAPPARS"] = [_number(value) for value in record]

        elif keyword in ("ECHO", "NOECHO"):
            continue

        else:
            if keyword not in unhandled:
                unhandled.append(keyword)

    if open_period:
        schedule.control.append(control)
    if not schedule.control:
        # A deck may declare no wells at all; the steps still need a period to
        # point at, and an empty one is what they had implicitly.
        schedule.control.append(control)
        schedule.step["control"] = [max(index, 1)
                                    for index in schedule.step["control"]]
    return deck


readSCHEDULE = read_schedule


def _read_tstep(cursor: DeckCursor) -> List[float]:
    """`TSTEP` values up to the terminating slash, expanding `n*v` runs."""
    values: List[float] = []
    for token in cursor.read_record_tokens():
        repeat = _REPEAT.match(token)
        if repeat and repeat.group(2):
            values.extend([_number(repeat.group(2))]*int(repeat.group(1)))
        else:
            values.append(_number(token))
    return values


def _read_dates(cursor: DeckCursor, start: date,
                elapsed: List[float]) -> List[float]:
    """`DATES` as step lengths in days (MRST ``readDATES``).

    A deck may mix `DATES` and `TSTEP`, so the first interval is measured from
    the start date *plus everything already elapsed* rather than from the start
    date itself.
    """
    dates: List[date] = []
    while cursor:
        record = cursor.read_record_tokens()
        if not record:
            break
        dates.append(_parse_date(record))
    if not dates:
        return []

    previous = _to_ordinal(start) + float(sum(elapsed))
    out: List[float] = []
    for moment in dates:
        current = float(_to_ordinal(moment))
        step = current - previous
        if step <= 0.0:
            raise ValueError(
                f"DATES entry {moment} does not advance time past the "
                "previous report step")
        out.append(step)
        previous = current
    return out


def _parse_date(record: List[str]) -> date:
    """`10 'JUN' 2011` - the day/month-name/year form ECLIPSE writes - with an
    optional time of day, `10 'JUN' 2011 12:00:00.001`, which MATLAB's
    `datenum` reads too (Norne's first report date is a millisecond past
    midnight)."""
    fields = [str(field).strip().strip("'\"") for field in record if field]
    if len(fields) < 3:
        raise ValueError(f"DATES record {record!r} is not a day, month, year")
    day, month, year = fields[0], fields[1].upper()[:3], fields[2]
    if month not in _MONTHS:
        raise ValueError(f"unknown month {fields[1]!r} in DATES")
    moment = date(int(float(year)), _MONTHS[month], int(float(day)))
    if len(fields) > 3 and ":" in fields[3]:
        h, mi, s = (fields[3].split(":") + ["0", "0"])[:3]
        seconds = float(s)
        whole = int(seconds)
        return datetime(moment.year, moment.month, moment.day, int(h), int(mi), whole,
                        round((seconds - whole)*1e6))
    return moment


def _to_ordinal(value: Any) -> float:
    if isinstance(value, datetime):
        # MATLAB's datenum: day + (h + (mi + s/60)/60)/24.
        s = value.second + value.microsecond/1e6
        return value.toordinal() + (value.hour + (value.minute + s/60)/60)/24
    if isinstance(value, date):
        return float(value.toordinal())
    return float(value)


def _number(token: Any) -> float:
    if isinstance(token, (int, float)):
        return float(token)
    text = str(token).strip()
    try:
        return float(re.sub(r"[dD]", "e", text))
    except ValueError:
        return float("nan")


__all__ = ["STEP_KEYWORDS", "WELL_KEYWORDS", "readSCHEDULE", "read_schedule"]


if __name__ == "__main__":
    from pymrst.modules.deckformat.deckinput.initialize_deck import (
        initialize_deck,
    )

    def _read(text: str, start=date(1983, 1, 1)):
        deck = initialize_deck()
        deck["RUNSPEC"]["START"] = start
        cursor = DeckCursor.from_text(text)
        keyword = cursor.next_keyword()
        assert keyword == "SCHEDULE", keyword
        return read_schedule(cursor, deck)["SCHEDULE"]

    WELLS = ("SCHEDULE\n"
             "WELSPECS\n 'P1' 'G' 5 5 1* 'OIL' /\n 'I1' 'G' 1 1 1* 'WATER' /\n/\n"
             "COMPDAT\n 'P1' 2* 1 3 'OPEN' 2* 0.2 /\n 'I1' 2* 1 3 'OPEN' 2* 0.2 /\n/\n"
             "WCONPROD\n 'P1' 'OPEN' 'BHP' 5* 395 /\n/\n"
             "WCONINJE\n 'I1' 'WATER' 'OPEN' 'RATE' 79.5 1* 420 /\n/\n")

    def test_consecutive_well_keywords_make_one_control_period():
        """The rule the whole reader turns on.

        Four well keywords in a row describe one state of the field. Opening a
        period per keyword gives the Egg model twelve periods where it has
        one, and every step but the last runs a half-specified well set.
        """
        schedule = _read(WELLS + "TSTEP\n 10 /\n")
        assert len(schedule.control) == 1, len(schedule.control)
        assert schedule.step["control"] == [1]
        print("ok test_consecutive_well_keywords_make_one_control_period")

    def test_a_time_step_closes_the_period():
        schedule = _read(WELLS + "TSTEP\n 10 20 /\n"
                         "WCONPROD\n 'P1' 'OPEN' 'BHP' 5* 300 /\n/\n"
                         "TSTEP\n 30 /\n")
        assert len(schedule.control) == 2, len(schedule.control)
        assert schedule.step["control"] == [1, 1, 2], schedule.step["control"]
        assert schedule.step["val"] == [10.0, 20.0, 30.0]
        print("ok test_a_time_step_closes_the_period")

    def test_the_second_period_keeps_the_wells_it_was_not_told_about():
        """`WCONPROD` alone in the second period must not un-declare `I1`."""
        schedule = _read(WELLS + "TSTEP\n 10 /\n"
                         "WCONPROD\n 'P1' 'OPEN' 'BHP' 5* 300 /\n/\n"
                         "TSTEP\n 30 /\n")
        second = schedule.control[1]
        assert len(second["WELSPECS"]) == 2, second["WELSPECS"]
        assert len(second["COMPDAT"]) == 2
        assert len(second["WCONINJE"]) == 1, "the injector lost its control"
        assert second["WCONPROD"][0][8] == 300.0
        print("ok test_the_second_period_keeps_the_wells_it_was_not_told_about")

    def test_periods_do_not_share_records():
        schedule = _read(WELLS + "TSTEP\n 10 /\n"
                         "WCONPROD\n 'P1' 'OPEN' 'BHP' 5* 300 /\n/\n"
                         "TSTEP\n 30 /\n")
        assert schedule.control[0]["WCONPROD"][0][8] == 395.0
        print("ok test_periods_do_not_share_records")

    def test_a_repeat_count_expands_into_separate_steps():
        """`4*30` is four thirty-day steps, not one. Read as a single value it
        gives a schedule a quarter the length the deck asked for."""
        schedule = _read(WELLS + "TSTEP\n 4*30 /\n")
        assert schedule.step["val"] == [30.0]*4, schedule.step["val"]
        print("ok test_a_repeat_count_expands_into_separate_steps")

    def test_dates_become_step_lengths():
        schedule = _read(WELLS + "DATES\n 1 FEB 2011 /\n 1 MAR 2011 /\n/\n",
                         start=date(2011, 1, 1))
        assert schedule.step["val"] == [31.0, 28.0], schedule.step["val"]
        print("ok test_dates_become_step_lengths")

    def test_dates_measure_from_the_time_already_elapsed():
        """A deck may mix the two. Measuring from the start date instead makes
        the first `DATES` step repeat every `TSTEP` that preceded it."""
        schedule = _read(WELLS + "TSTEP\n 10 /\n"
                         "DATES\n 1 FEB 2011 /\n/\n", start=date(2011, 1, 1))
        assert schedule.step["val"] == [10.0, 21.0], schedule.step["val"]
        print("ok test_dates_measure_from_the_time_already_elapsed")

    def test_jly_is_july():
        """ECLIPSE writes JLY; `datenum` accepts it and so must this."""
        schedule = _read(WELLS + "DATES\n 1 JLY 2011 /\n/\n",
                         start=date(2011, 6, 1))
        assert schedule.step["val"] == [30.0]
        print("ok test_jly_is_july")

    def test_a_time_of_day_counts_as_datenum_counts_it():
        """Norne: `14 'NOV' 1997 00:00:00.001`. MRST's step lengths, from
        MATLAB 2024a, are 8 and 17 days plus and minus that millisecond as
        datenum differences round it: 691200.00099577 s and
        1468799.99900423 s after convertDeckUnits."""
        schedule = _read(WELLS + "DATES\n 14 'NOV' 1997 00:00:00.001 /\n"
                         " 1 'DEC' 1997 /\n/\n", start=date(1997, 11, 6))
        seconds = [v*86400 for v in schedule.step["val"]]
        assert seconds == [691200.0009957701, 1468799.99900423], seconds
        print("ok test_a_time_of_day_counts_as_datenum_counts_it")

    def test_a_date_that_goes_backwards_is_refused():
        try:
            _read(WELLS + "DATES\n 1 FEB 2011 /\n 1 JAN 2011 /\n/\n",
                  start=date(2011, 1, 1))
        except ValueError as refused:
            assert "advance" in str(refused)
            print("ok test_a_date_that_goes_backwards_is_refused")
            return
        raise AssertionError("time must move forward")

    def test_an_unhandled_keyword_is_recorded_not_fatal():
        deck = initialize_deck()
        cursor = DeckCursor.from_text(
            "SCHEDULE\nTUNING\n0.1 30 /\n/\n1 1 250 /\n" + WELLS[9:]
            + "TSTEP\n 10 /\n")
        cursor.next_keyword()
        deck = read_schedule(cursor, deck)
        assert "TUNING" in deck["UnhandledKeywords"]["SCHEDULE"]
        assert len(deck["SCHEDULE"].control) == 1
        assert deck["SCHEDULE"].step["val"] == [10.0]
        print("ok test_an_unhandled_keyword_is_recorded_not_fatal")

    def test_a_deck_with_no_wells_still_gets_a_period():
        schedule = _read("SCHEDULE\nTSTEP\n 10 20 /\n")
        assert len(schedule.control) == 1
        assert schedule.step["control"] == [1, 1]
        print("ok test_a_deck_with_no_wells_still_gets_a_period")

    def test_a_trailing_period_without_steps_is_kept():
        schedule = _read(WELLS)
        assert len(schedule.control) == 1 and schedule.step["val"] == []
        print("ok test_a_trailing_period_without_steps_is_kept")

    test_consecutive_well_keywords_make_one_control_period()
    test_a_time_step_closes_the_period()
    test_the_second_period_keeps_the_wells_it_was_not_told_about()
    test_periods_do_not_share_records()
    test_a_repeat_count_expands_into_separate_steps()
    test_dates_become_step_lengths()
    test_dates_measure_from_the_time_already_elapsed()
    test_jly_is_july()
    test_a_time_of_day_counts_as_datenum_counts_it()
    test_a_date_that_goes_backwards_is_refused()
    test_an_unhandled_keyword_is_recorded_not_fatal()
    test_a_deck_with_no_wells_still_gets_a_period()
    test_a_trailing_period_without_steps_is_kept()
    print("\nAll tests passed!")
