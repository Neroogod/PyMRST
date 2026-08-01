"""Human-readable durations (MRST ``utils/formatTimeRange.m``).

Turns a number of seconds into "2 Years, 30 Days, 4 Hours". Used in progress
reporting, where "63115200 seconds" tells a reader nothing.

`limit` caps how many units appear, and the last one shown is given to two
decimals rather than truncated - so `limit=1` on eighteen months reads
"1.50 Years" and not "1 Year", which would be wrong by six months.
"""

from pymrst.utils.units.units import DAY, HOUR, MINUTE, YEAR

# Ordered coarse to fine, as MRST has them. Minutes are absent from MRST's
# list, which jumps from hours to seconds; that is reproduced.
UNITS = (("Year", YEAR), ("Day", DAY), ("Hour", HOUR),
         ("Second", 1.0), ("Millisecond", 1.0e-3))


def format_time_range(time: float, limit: int = None) -> str:
    """Describe ``time`` seconds in years, days, hours, seconds, milliseconds."""
    parts = []
    remaining = float(time)
    for index, (name, size) in enumerate(UNITS):
        count = int(remaining // size)
        if count <= 0:
            continue
        last = limit is not None and len(parts) + 1 >= limit
        if last:
            value = remaining / size
            parts.append(f"{value:1.2f} {name}" + ("s" if value != 1 else ""))
            break
        parts.append(f"{count} {name}" + ("s" if count != 1 else ""))
        remaining -= count * size
    return ", ".join(parts)


formatTimeRange = format_time_range


if __name__ == "__main__":

    def test_a_single_unit_is_singular() -> None:
        assert format_time_range(HOUR) == "1 Hour", format_time_range(HOUR)
        print("ok test_a_single_unit_is_singular")

    def test_several_units_are_listed_coarse_to_fine() -> None:
        out = format_time_range(2 * DAY + 3 * HOUR)
        assert out == "2 Days, 3 Hours", out
        print("ok test_several_units_are_listed_coarse_to_fine")

    def test_units_that_do_not_appear_are_omitted() -> None:
        out = format_time_range(YEAR + HOUR)
        assert "Day" not in out, out
        print("ok test_units_that_do_not_appear_are_omitted")

    def test_the_limit_caps_the_number_of_units() -> None:
        out = format_time_range(2 * DAY + 3 * HOUR + 4, limit=2)
        assert out.count(",") == 1, out
        print("ok test_the_limit_caps_the_number_of_units")

    def test_the_last_unit_shown_carries_the_remainder() -> None:
        """`limit=1` on 18 months is 1.50 Years, not 1 Year."""
        out = format_time_range(1.5 * YEAR, limit=1)
        assert out.startswith("1.50 Year"), out
        print(f"ok test_the_last_unit_shown_carries_the_remainder ({out})")

    def test_no_limit_lists_everything_present() -> None:
        out = format_time_range(YEAR + DAY + HOUR + 1)
        assert out.count(",") == 3, out
        print("ok test_no_limit_lists_everything_present")

    def test_zero_is_the_empty_string() -> None:
        assert format_time_range(0.0) == ""
        print("ok test_zero_is_the_empty_string")

    def test_a_sub_second_time_reaches_milliseconds() -> None:
        out = format_time_range(0.005)
        assert "Millisecond" in out, out
        print(f"ok test_a_sub_second_time_reaches_milliseconds ({out})")

    test_a_single_unit_is_singular()
    test_several_units_are_listed_coarse_to_fine()
    test_units_that_do_not_appear_are_omitted()
    test_the_limit_caps_the_number_of_units()
    test_the_last_unit_shown_carries_the_remainder()
    test_no_limit_lists_everything_present()
    test_zero_is_the_empty_string()
    test_a_sub_second_time_reaches_milliseconds()
    print("\nAll tests passed!")
