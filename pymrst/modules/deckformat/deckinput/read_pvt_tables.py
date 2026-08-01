"""PVT tables (MRST ``readImmisciblePVTTable.m`` and ``readMisciblePVTTable.m``).

Two shapes:

- **immiscible** (PVDO, PVDG, PVTW): plain rows of `n_col` numbers, one table
  per PVT region.
- **miscible** (PVTO, PVTG): records of the form `key row [row ...]`, where the
  key is Rs for live oil or Rv for wet gas, and each following row is a point
  on that key's branch. Records accumulate until an empty one ends the table.

The miscible form comes back packed - `key`, `pos`, `data` - which is exactly
what `preprocess_table_pvt` expands. Keeping it packed here means the deck
reader does no physics.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor,
    strip_quotes,
)


@dataclass
class MisciblePVT:
    """A packed PVTO/PVTG table: one branch per key."""

    key: np.ndarray
    pos: np.ndarray
    data: np.ndarray


def _numbers(fields) -> np.ndarray:
    return np.array([float(strip_quotes(f)) for f in fields], dtype=float)


def read_immiscible_pvt_table(cursor: DeckCursor, n_tab: int,
                              n_col: int) -> List[np.ndarray]:
    """Read ``n_tab`` immiscible PVT tables of ``n_col`` columns."""
    tables: List[np.ndarray] = []
    for _ in range(n_tab):
        values = _numbers(cursor.read_record_tokens())
        if values.size % n_col:
            raise ValueError(
                f"{values.size} values do not fill rows of {n_col}"
            )
        tables.append(values.reshape(-1, n_col))
    return tables


def fill_usat_invlinear(table: MisciblePVT) -> MisciblePVT:
    """Give saturated-only branches an undersaturated extension.

    MRST ``deckinput/private/fill_usat_invlinear.m``. A PVTO branch that
    states only its bubble point has no undersaturated behaviour of its own,
    so ECLIPSE derives one from a fully specified branch: the same pressure
    increments, with B and mu scaled by that branch's ratios relative to its
    own bubble point.

    Skipping this does not fail loudly - the table is still well formed, just
    short - so an undersaturated cell silently extrapolates off the end of a
    one-point branch instead of following the deck's intent.
    """
    counts = np.diff(table.pos)
    short = np.flatnonzero(counts == 1)
    full = np.flatnonzero(counts > 1)
    if short.size == 0 or full.size == 0:
        return table

    # MRST's donor rule: the first fully specified branch at or after this one.
    donors = np.searchsorted(full, short, side="left")
    donors = np.minimum(donors, full.size - 1)

    keys, positions, blocks = [], [0], []
    for index in range(table.key.size):
        block = table.data[table.pos[index]:table.pos[index + 1]]
        if counts[index] == 1:
            donor = full[donors[np.searchsorted(short, index)]]
            reference = table.data[table.pos[donor]:table.pos[donor + 1]]
            head = reference[0]
            # pressure increments, and ratios of B and mu, from the donor
            extra = np.empty((reference.shape[0] - 1, block.shape[1]))
            extra[:, 0] = block[0, 0] + (reference[1:, 0] - head[0])
            extra[:, 1] = block[0, 1] * (reference[1:, 1] / head[1])
            extra[:, 2] = block[0, 2] * (reference[1:, 2] / head[2])
            block = np.vstack([block, extra])
        keys.append(table.key[index])
        blocks.append(block)
        positions.append(positions[-1] + block.shape[0])

    return MisciblePVT(key=np.asarray(keys, dtype=float),
                       pos=np.asarray(positions, dtype=np.int64),
                       data=np.vstack(blocks))


def read_miscible_pvt_table(cursor: DeckCursor, n_tab: int,
                            n_col: int) -> List[MisciblePVT]:
    """Read ``n_tab`` miscible PVT tables, returned packed by key."""
    tables: List[MisciblePVT] = []
    for _ in range(n_tab):
        keys: List[float] = []
        positions: List[int] = [0]
        rows: List[np.ndarray] = []
        while cursor:
            values = _numbers(cursor.read_record_tokens())
            if values.size == 0:
                break                       # empty record ends this table
            if values.size % n_col != 1:
                raise ValueError(
                    f"a miscible PVT record must be a key followed by rows of "
                    f"{n_col}; got {values.size} values"
                )
            keys.append(values[0])
            block = values[1:].reshape(-1, n_col)
            rows.append(block)
            positions.append(positions[-1] + block.shape[0])
        tables.append(fill_usat_invlinear(MisciblePVT(
            key=np.asarray(keys, dtype=float),
            pos=np.asarray(positions, dtype=np.int64),
            data=(np.vstack(rows) if rows else np.zeros((0, n_col))),
        )))
    return tables


readImmisciblePVTTable = read_immiscible_pvt_table
readMisciblePVTTable = read_miscible_pvt_table


if __name__ == "__main__":
    from pymrst.modules.deckformat.deckinput.read_defaulted_record import tokenize

    def test_immiscible_table_reshapes_into_rows() -> None:
        text = "100 1.0 0.5\n200 0.9 0.6 /\n"
        table = read_immiscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        assert table.shape == (2, 3), table.shape
        assert np.allclose(table[1], [200, 0.9, 0.6])
        print("ok test_immiscible_table_reshapes_into_rows")

    def test_immiscible_reads_one_table_per_region() -> None:
        text = "100 1.0 /\n200 2.0 /\n"
        tables = read_immiscible_pvt_table(DeckCursor(tokenize(text)), 2, 2)
        assert len(tables) == 2
        assert np.isclose(tables[1][0, 0], 200)
        print("ok test_immiscible_reads_one_table_per_region")

    def test_immiscible_rejects_a_ragged_table() -> None:
        try:
            read_immiscible_pvt_table(DeckCursor(tokenize("1 2 3 /\n")), 1, 2)
        except ValueError:
            print("ok test_immiscible_rejects_a_ragged_table")
            return
        raise AssertionError("a ragged table must be rejected")

    def test_miscible_packs_one_branch_per_key() -> None:
        """PVTO shape: Rs then rows of (p, B, mu), continuation rows allowed."""
        text = (
            "0.0  100 1.10 1.0\n"
            "     200 1.08 1.1 /\n"
            "50.0 200 1.30 0.9 /\n"
            "/\n"
        )
        table = read_miscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        assert np.allclose(table.key, [0.0, 50.0]), table.key
        assert np.allclose(table.data[1], [200, 1.08, 1.1])
        # branch 2 states only its bubble point and is extended from branch 1
        assert np.array_equal(table.pos, [0, 2, 4]), table.pos
        print("ok test_miscible_packs_one_branch_per_key")

    def test_miscible_positions_delimit_the_branches() -> None:
        """With no fully specified branch there is nothing to extend from."""
        text = "0.0 100 1.1 1.0 /\n1.0 200 1.2 1.1 /\n/\n"
        table = read_miscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        for index in range(table.key.size):
            block = table.data[table.pos[index]:table.pos[index + 1]]
            assert block.shape[0] == 1, block
        print("ok test_miscible_positions_delimit_the_branches")

    def test_saturated_only_branch_inherits_the_donor_shape() -> None:
        """Pressure increments are copied; B and mu scale by the donor ratios."""
        text = (
            "0.0  100 1.00 2.00\n"
            "     150 0.80 3.00 /\n"
            "50.0 200 2.00 4.00 /\n"
            "/\n"
        )
        table = read_miscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        derived = table.data[table.pos[1] + 1]
        assert np.isclose(derived[0], 200 + (150 - 100)), derived
        assert np.isclose(derived[1], 2.00 * (0.80 / 1.00)), derived
        assert np.isclose(derived[2], 4.00 * (3.00 / 2.00)), derived
        print("ok test_saturated_only_branch_inherits_the_donor_shape")

    def test_fill_leaves_a_fully_specified_table_alone() -> None:
        text = "0.0 100 1.1 1.0\n 200 1.0 1.2 /\n1.0 200 1.2 1.1\n 300 1.1 1.3 /\n/\n"
        table = read_miscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        assert np.array_equal(table.pos, [0, 2, 4]), table.pos
        assert table.data.shape == (4, 3)
        print("ok test_fill_leaves_a_fully_specified_table_alone")

    def test_miscible_feeds_preprocess_table_pvt() -> None:
        """The packed form must be exactly what the interpolator expands."""
        from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
            PackedPVT, preprocess_table_pvt,
        )
        text = "0.0 100 1.1 1.0\n 200 1.08 1.1 /\n50.0 200 1.3 0.9 /\n/\n"
        packed = read_miscible_pvt_table(DeckCursor(tokenize(text)), 1, 3)[0]
        expanded = preprocess_table_pvt(
            PackedPVT(key=packed.key, pos=packed.pos, data=packed.data))
        assert len(expanded.expanded) == 2
        assert np.allclose(expanded.sat.x, [100.0, 200.0])
        print("ok test_miscible_feeds_preprocess_table_pvt")

    def test_miscible_rejects_a_record_without_a_key() -> None:
        try:
            read_miscible_pvt_table(DeckCursor(tokenize("100 1.1 1.0 /\n/\n")),
                                    1, 3)
        except ValueError as error:
            assert "key followed by rows" in str(error)
            print("ok test_miscible_rejects_a_record_without_a_key")
            return
        raise AssertionError("a keyless record must be rejected")

    test_immiscible_table_reshapes_into_rows()
    test_immiscible_reads_one_table_per_region()
    test_immiscible_rejects_a_ragged_table()
    test_miscible_packs_one_branch_per_key()
    test_miscible_positions_delimit_the_branches()
    test_saturated_only_branch_inherits_the_donor_shape()
    test_fill_leaves_a_fully_specified_table_alone()
    test_miscible_feeds_preprocess_table_pvt()
    test_miscible_rejects_a_record_without_a_key()
    print("\nAll tests passed!")
