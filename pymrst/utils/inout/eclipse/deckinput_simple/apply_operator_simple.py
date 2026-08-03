"""Apply ECLIPSE box operators (MRST ``.../deckinput_simple/applyOperatorSimple.m``).

A deck may modify an array after declaring it, over a box of cells:

    PERMX
      1000*100 /
    MULTIPLY
      'PERMX'  0.1   1 10  1 10  3 5 /
    /

means "scale PERMX by 0.1 in layers 3-5". Six operators are defined - `ADD`,
`COPY`, `EQUALS`, `MAXVALUE`, `MINVALUE`, `MULTIPLY` - and a reader that ignores
them does not fail, it **silently returns the unmodified array**. That is the
worst kind of wrong: the numbers look reasonable and are off by whatever the
deck asked for.

PyMRST had no operator support at all, in this reader or in `deckformat`, so
the deck above was read with PERMX at 100 rather than 10 and nothing said so.

**The box is 1-based and inclusive**, as ECLIPSE writes it, and is converted
here. Omitted box entries default to the whole array, or to the enclosing
`BOX` if one is active.

MRST's `readOperator` carries the same dead arity check as `readFaults` and
`readMultfelt` - `if(~length(split)==8)`, which parses as `(~length(split)) == 8`
and is never true. Records are checked properly here.
"""

from typing import Dict, Optional, Sequence

import numpy as np

OPERATORS = ("ADD", "COPY", "EQUALS", "MAXVALUE", "MINVALUE", "MULTIPLY")


def box_indices(box: Sequence[int], cart_dims: Sequence[int]) -> np.ndarray:
    """Linear indices of an ECLIPSE box, 1-based inclusive, Fortran order."""
    i1, i2, j1, j2, k1, k2 = (int(v) for v in box)
    nx, ny, nz = (int(v) for v in cart_dims)
    for lo, hi, n, name in ((i1, i2, nx, "I"), (j1, j2, ny, "J"),
                            (k1, k2, nz, "K")):
        if not 1 <= lo <= hi <= n:
            raise ValueError(
                f"box range {name}={lo}:{hi} lies outside 1..{n}; ECLIPSE "
                f"numbers cells from 1 and the range is inclusive")
    i = np.arange(i1 - 1, i2)
    j = np.arange(j1 - 1, j2)
    k = np.arange(k1 - 1, k2)
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    return (ii + nx * (jj + ny * kk)).ravel(order="F")


def apply_operator(sect: Dict, keyword: str, target: str, value,
                   box: Sequence[int], cart_dims: Sequence[int]) -> Dict:
    """Apply one operator record to ``sect`` in place, and return it."""
    keyword = keyword.upper()
    if keyword not in OPERATORS:
        raise ValueError(
            f"{keyword!r} is not a box operator; expected one of {OPERATORS}")

    index = box_indices(box, cart_dims)

    if keyword == "COPY":
        source = str(value).strip().strip("'\"")
        if source not in sect:
            raise ValueError(f"COPY reads {source!r}, which the deck has not set")
        if target not in sect:
            sect[target] = np.full(int(np.prod(cart_dims)), np.nan)
        sect[target] = np.asarray(sect[target], dtype=float).copy()
        sect[target][index] = np.asarray(sect[source], dtype=float)[index]
        return sect

    number = float(value)
    if keyword == "EQUALS":
        if target not in sect:
            sect[target] = np.full(int(np.prod(cart_dims)), np.nan)
        sect[target] = np.asarray(sect[target], dtype=float).copy()
        sect[target][index] = number
        return sect

    if target not in sect:
        raise ValueError(
            f"{keyword} modifies {target!r}, which the deck has not set; only "
            f"EQUALS and COPY may create an array")
    array = np.asarray(sect[target], dtype=float).copy()
    if np.any(np.isnan(array[index])):
        raise ValueError(
            f"{keyword} touches undefined entries of {target!r}")

    if keyword == "ADD":
        array[index] += number
    elif keyword == "MULTIPLY":
        array[index] *= number
    elif keyword == "MAXVALUE":
        array[index] = np.minimum(array[index], number)
    elif keyword == "MINVALUE":
        array[index] = np.maximum(array[index], number)
    sect[target] = array
    return sect


def apply_operator_simple(sect: Dict, records: Sequence[Sequence], keyword: str,
                          cart_dims: Sequence[int],
                          box: Optional[Sequence[int]] = None) -> Dict:
    """Apply every record of one operator keyword."""
    whole = [1, int(cart_dims[0]), 1, int(cart_dims[1]), 1, int(cart_dims[2])]
    default = list(box) if box else whole
    for record in records:
        if len(record) < 2:
            raise ValueError(
                f"a {keyword} record needs a target and a value, got {record}")
        # For every operator but one, a record reads `TARGET VALUE [box]`.
        # `COPY` is the exception: ECLIPSE writes `COPY source target [box]`,
        # and MRST's implementation spells the direction out as `y <- x` with
        # `x = rec{1}`. Reading COPY like the others reverses the copy, so
        # `COPY 'PERMX' 'PERMY'` would overwrite PERMX from PERMY. It raises
        # when the target does not exist yet, which is the lucky case; when
        # both arrays exist it silently copies the wrong way.
        if keyword.upper() == "COPY":
            target = str(record[1]).strip().strip("'\"")
            value = record[0]
        else:
            target = str(record[0]).strip().strip("'\"")
            value = record[1]
        given = list(record[2:])
        if given and len(given) != 6:
            raise ValueError(
                f"a {keyword} box needs six indices (I1 I2 J1 J2 K1 K2), got "
                f"{len(given)}: {record}")
        sect = apply_operator(sect, keyword, target, value,
                              given or default, cart_dims)
    return sect


applyOperatorSimple = apply_operator_simple


if __name__ == "__main__":

    DIMS = (2, 2, 2)

    def sect():
        return {"PERMX": np.full(8, 100.0), "PORO": np.full(8, 0.3)}

    def test_multiply_scales_the_whole_array() -> None:
        got = apply_operator_simple(sect(), [["'PERMX'", 0.1]], "MULTIPLY", DIMS)
        assert got["PERMX"].tolist() == [10.0] * 8
        print("ok test_multiply_scales_the_whole_array")

    def test_a_box_limits_the_change() -> None:
        """Only layer 2 is scaled; layer 1 keeps its value."""
        got = apply_operator_simple(sect(), [["'PERMX'", 0.1, 1, 2, 1, 2, 2, 2]],
                                    "MULTIPLY", DIMS)
        assert got["PERMX"][:4].tolist() == [100.0] * 4
        assert got["PERMX"][4:].tolist() == [10.0] * 4
        print("ok test_a_box_limits_the_change")

    def test_add() -> None:
        got = apply_operator_simple(sect(), [["'PERMX'", 5.0]], "ADD", DIMS)
        assert got["PERMX"].tolist() == [105.0] * 8
        print("ok test_add")

    def test_equals_can_create_an_array() -> None:
        got = apply_operator_simple({}, [["'NTG'", 0.8]], "EQUALS", DIMS)
        assert got["NTG"].tolist() == [0.8] * 8
        print("ok test_equals_can_create_an_array")

    def test_copy_duplicates_another_array() -> None:
        """Source first, target second: `COPY 'PERMX' 'PERMY'` sets PERMY.

        This test previously passed the pair the other way round, matching the
        implementation rather than the format. Both were wrong together, so it
        passed.
        """
        got = apply_operator_simple(sect(), [["'PERMX'", "'PERMY'"]], "COPY", DIMS)
        assert got["PERMY"].tolist() == [100.0] * 8
        print("ok test_copy_duplicates_another_array")

    def test_maxvalue_is_an_upper_cap() -> None:
        """MAXVALUE sets the maximum, so it caps from above."""
        got = apply_operator_simple(sect(), [["'PERMX'", 50.0]], "MAXVALUE", DIMS)
        assert got["PERMX"].tolist() == [50.0] * 8
        print("ok test_maxvalue_is_an_upper_cap")

    def test_minvalue_is_a_lower_floor() -> None:
        got = apply_operator_simple(sect(), [["'PERMX'", 500.0]], "MINVALUE", DIMS)
        assert got["PERMX"].tolist() == [500.0] * 8
        print("ok test_minvalue_is_a_lower_floor")

    def test_several_records_apply_in_order() -> None:
        got = apply_operator_simple(sect(), [["'PERMX'", 0.1], ["'PERMX'", 3.0]],
                                    "MULTIPLY", DIMS)
        assert got["PERMX"].tolist() == [30.0] * 8
        print("ok test_several_records_apply_in_order")

    def test_the_box_is_one_based_and_inclusive() -> None:
        idx = box_indices([1, 1, 1, 1, 1, 1], DIMS)
        assert idx.tolist() == [0], idx
        idx = box_indices([1, 2, 1, 2, 1, 2], DIMS)
        assert sorted(idx.tolist()) == list(range(8))
        print("ok test_the_box_is_one_based_and_inclusive")

    def test_the_input_dict_is_not_shared() -> None:
        original = sect()
        before = original["PERMX"].copy()
        apply_operator_simple(dict(original), [["'PERMX'", 0.1]], "MULTIPLY", DIMS)
        assert original["PERMX"].tolist() == before.tolist()
        print("ok test_the_input_dict_is_not_shared")

    def test_modifying_an_unset_array_is_rejected() -> None:
        try:
            apply_operator_simple({}, [["'PERMX'", 0.1]], "MULTIPLY", DIMS)
        except ValueError as error:
            assert "which the deck has not set" in str(error)
            print("ok test_modifying_an_unset_array_is_rejected")
            return
        raise AssertionError("multiplying an undefined array must be rejected")

    def test_a_box_outside_the_grid_is_rejected() -> None:
        try:
            apply_operator_simple(sect(), [["'PERMX'", 0.1, 1, 9, 1, 2, 1, 2]],
                                  "MULTIPLY", DIMS)
        except ValueError as error:
            assert "outside 1..2" in str(error)
            print("ok test_a_box_outside_the_grid_is_rejected")
            return
        raise AssertionError("an out-of-range box must be rejected")

    def test_a_short_box_is_rejected() -> None:
        """MRST's arity check never fires; see the module docstring."""
        try:
            apply_operator_simple(sect(), [["'PERMX'", 0.1, 1, 2]], "MULTIPLY",
                                  DIMS)
        except ValueError as error:
            assert "six indices" in str(error)
            print("ok test_a_short_box_is_rejected")
            return
        raise AssertionError("a malformed box must be rejected")

    def test_an_unknown_operator_is_rejected() -> None:
        try:
            apply_operator_simple(sect(), [["'PERMX'", 1.0]], "SCALE", DIMS)
        except ValueError as error:
            assert "not a box operator" in str(error)
            print("ok test_an_unknown_operator_is_rejected")
            return
        raise AssertionError("an unknown operator must be rejected")

    test_multiply_scales_the_whole_array()
    test_a_box_limits_the_change()
    test_add()
    test_equals_can_create_an_array()
    test_copy_duplicates_another_array()
    test_maxvalue_is_an_upper_cap()
    test_minvalue_is_a_lower_floor()
    test_several_records_apply_in_order()
    test_the_box_is_one_based_and_inclusive()
    test_the_input_dict_is_not_shared()
    test_modifying_an_unset_array_is_rejected()
    test_a_box_outside_the_grid_is_rejected()
    test_a_short_box_is_rejected()
    test_an_unknown_operator_is_rejected()
    def test_copy_goes_source_then_target():
        """ECLIPSE writes `COPY source target`, not `COPY target source`.

        MRST spells the direction out in its own implementation as `y <- x`
        with `x = rec{1}`. The Egg model relies on it:

            COPY
             'PERMX' 'PERMY'  1 60 1 60 1 7 /

        Reading the record the other way round overwrites PERMX from PERMY.
        Here both arrays exist and differ, so a reversed copy is visible
        rather than merely raising.
        """
        sect = {"PERMX": np.full(8, 100.0), "PERMY": np.full(8, 7.0)}
        out = apply_operator_simple(dict(sect), [["PERMX", "PERMY"]], "COPY",
                                    (2, 2, 2))
        assert np.allclose(out["PERMY"], 100.0), out["PERMY"]
        assert np.allclose(out["PERMX"], 100.0), "source was modified"
        print("ok test_copy_goes_source_then_target")

    def test_copy_creates_the_target_when_absent():
        sect = {"PERMX": np.full(8, 100.0)}
        out = apply_operator_simple(dict(sect), [["PERMX", "PERMZ"]], "COPY",
                                    (2, 2, 2))
        assert np.allclose(out["PERMZ"], 100.0), out.get("PERMZ")
        print("ok test_copy_creates_the_target_when_absent")

    test_copy_goes_source_then_target()
    test_copy_creates_the_target_when_absent()
    print("\nAll tests passed!")
