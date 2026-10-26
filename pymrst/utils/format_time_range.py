"""Human-readable durations (MRST ``utils/formatTimeRange.m``).

Turns a number of seconds into "2 Years, 30 Days, 4 Hours". Used in progress
reporting, where "63115200 seconds" tells a reader nothing.

`limit` reproduces MRST's refinement-depth counter.  The first emitted unit
has depth zero, so a limit of two can display three units; the unit at the
requested depth carries the remainder as a decimal.
"""

import math

from pymrst.utils.units.units import DAY, HOUR, MINUTE, YEAR

# Ordered coarse to fine, as MRST has them. Minutes are absent from MRST's
# list, which jumps from hours to seconds; that is reproduced.
UNITS = (("Year", YEAR), ("Day", DAY), ("Hour", HOUR),
         ("Second", 1.0), ("Millisecond", 1.0e-3))


def format_time_range(time: float, limit: int = None) -> str:
    """Describe ``time`` seconds in years, days, hours, seconds, milliseconds."""
    parts = []
    remaining = float(time)
    depth = 0
    added = False
    for index, (name, size) in enumerate(UNITS):
        # MRST's `floor(time/unit)`; `remaining // size` rounds differently
        # (1.0 // 0.1 is 9). An infinite time reads 'Inf Years', and the
        # NaN remainder it leaves adds nothing more.
        ratio = remaining/size
        if not ratio >= 1:
            continue
        count = math.floor(ratio) if math.isfinite(ratio) else ratio
        depth += int(added)
        last = limit is not None and depth == limit
        if last:
            value = remaining / size
            parts.append(f"{value:1.2f} {name}" + ("s" if value != 1 else ""))
            break
        shown = "Inf" if math.isinf(count) else str(count)
        parts.append(f"{shown} {name}" + ("s" if count != 1 else ""))
        remaining -= count * size
        added = True
    return ", ".join(parts)


formatTimeRange = format_time_range


if __name__ == "__main__":

    def test_infinity_is_inf_years_as_in_matlab() -> None:
        assert format_time_range(float("inf")) == "Inf Years"
        print("ok test_infinity_is_inf_years_as_in_matlab")

    def test_units_are_counted_by_floor_of_the_ratio() -> None:
        """MATLAB prints '300 Milliseconds' for 0.3 s; `0.3 // 1e-3` is 299."""
        assert format_time_range(0.3) == "300 Milliseconds", format_time_range(0.3)
        print("ok test_units_are_counted_by_floor_of_the_ratio")

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
        assert out == "2 Days, 3 Hours, 4.00 Seconds", out
        print("ok test_the_limit_caps_the_number_of_units")

    def test_the_last_unit_shown_carries_the_remainder() -> None:
        """MRST counts refinements after the first displayed unit."""
        out = format_time_range(1.5 * YEAR, limit=1)
        assert out.startswith("1 Year, 182.62 Day"), out
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

    test_infinity_is_inf_years_as_in_matlab()
    test_units_are_counted_by_floor_of_the_ratio()
    test_a_single_unit_is_singular()
    test_several_units_are_listed_coarse_to_fine()
    test_units_that_do_not_appear_are_omitted()
    test_the_limit_caps_the_number_of_units()
    test_the_last_unit_shown_carries_the_remainder()
    test_no_limit_lists_everything_present()
    test_zero_is_the_empty_string()
    test_a_sub_second_time_reaches_milliseconds()
    print("\nAll tests passed!")
