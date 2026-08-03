"""Build wells from an ECLIPSE schedule control record.

Port of MRST ``processWells.m``, at the MRST-shaped path.

The implementation is not repeated here. PyMRST already carries the whole
pipeline in ``pymrst.modules.compositional.facility``, split into the two steps
MRST fuses into one function:

* ``wells_from_deck_completion_records`` reads ``WELSPECS`` and ``COMPDAT``,
  expands completions, converts one-based deck indices to zero-based cells, and
  computes well indices;
* ``apply_deck_well_controls`` applies ``WCONPROD``, ``WCONINJE``,
  ``WCONINJH``, ``WCONHIST``, ``WELOPEN`` and ``WELTARG``, including record
  defaulting, control selection, limits and producer/injector signs.

This module exists so a reader following MRST's layout finds the function where
MRST puts it. Keeping one implementation means a fix reaches both entry points,
which is why the logic is delegated rather than duplicated.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def process_wells(g, rock, control: Mapping[str, Any], *,
                  compi: Sequence[float] | None = None,
                  gravity: Sequence[float] | None = None,
                  **kwargs) -> list:
    """Wells described by one ``SCHEDULE`` control record.

    ``control`` is a mapping of deck keywords, as MRST's
    ``deck.SCHEDULE.control(1)`` provides. Returns a list of ``Well`` objects
    with completions, well indices and controls already applied.
    """
    from pymrst.modules.compositional.facility import (
        apply_deck_well_controls,
        wells_from_deck_completion_records,
    )

    compi = np.array([1.0, 0.0, 0.0]) if compi is None else np.asarray(compi, float)
    gravity = np.array([0.0, 0.0, 9.80665]) if gravity is None else np.asarray(gravity, float)

    wells = wells_from_deck_completion_records(
        g, rock, control, compi=compi, gravity=gravity, **kwargs)
    return apply_deck_well_controls(wells, control, grid=g)


__all__ = ["process_wells"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import BARSA, DARCY, MILLI

    def _case():
        g = compute_geometry(cart_grid([5, 5, 3], [500.0, 500.0, 30.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        control = {
            "WELSPECS": [["PROD", "G1", 5, 5, 0.0, "OIL"],
                         ["INJ", "G1", 1, 1, 0.0, "WATER"]],
            "COMPDAT": [["PROD", 5, 5, 1, 3, "OPEN", -1, -1.0, 0.2],
                        ["INJ", 1, 1, 1, 3, "OPEN", -1, -1.0, 0.2]],
            "WCONPROD": [["PROD", "OPEN", "BHP", 0, 0, 0, 0, 0, 200 * BARSA]],
            "WCONINJE": [["INJ", "WATER", "OPEN", "RATE", 100.0]],
        }
        return g, rock, control

    def test_completions_map_to_the_right_cells():
        """Deck indices are one-based; cells are zero-based.

        A producer at `(5, 5)` over layers 1 to 3 in a 5x5x3 grid occupies
        cells 24, 49 and 74. An off-by-one here perforates the wrong column and
        the run still completes, which is what makes the check worth making.
        """
        g, rock, control = _case()
        wells = process_wells(g, rock, control)
        by_name = {w.name: np.asarray(w.cells).tolist() for w in wells}
        assert by_name["PROD"] == [24, 49, 74], by_name["PROD"]
        assert by_name["INJ"] == [0, 25, 50], by_name["INJ"]
        print("ok test_completions_map_to_the_right_cells")

    def test_controls_and_signs_are_applied():
        g, rock, control = _case()
        wells = {w.name: w for w in process_wells(g, rock, control)}
        prod, inj = wells["PROD"], wells["INJ"]
        assert prod.type == "bhp" and np.isclose(float(prod.val), 200 * BARSA)
        assert prod.sign == -1, prod.sign
        assert inj.type == "rate" and np.isclose(float(inj.val), 100.0)
        assert inj.sign == 1, inj.sign
        print("ok test_controls_and_signs_are_applied")

    def test_well_indices_are_positive():
        g, rock, control = _case()
        for w in process_wells(g, rock, control):
            wi = np.asarray(w.WI, dtype=float)
            assert wi.size == np.asarray(w.cells).size, (w.name, wi.size)
            assert np.all(wi > 0.0), (w.name, wi)
        print("ok test_well_indices_are_positive")

    def test_delegation_matches_calling_the_pieces_directly():
        """The wrapper must add nothing but the call order.

        If it ever diverges from the two functions it delegates to, the
        MRST-shaped entry point would quietly become a second implementation.
        """
        from pymrst.modules.compositional.facility import (
            apply_deck_well_controls,
            wells_from_deck_completion_records,
        )
        g, rock, control = _case()
        direct = apply_deck_well_controls(
            wells_from_deck_completion_records(
                g, rock, control, compi=np.array([1.0, 0.0, 0.0]),
                gravity=np.array([0.0, 0.0, 9.80665])),
            control, grid=g)
        viawrapper = process_wells(g, rock, control)
        assert len(direct) == len(viawrapper)
        for a, b in zip(direct, viawrapper):
            assert a.name == b.name and a.type == b.type
            assert np.isclose(float(a.val), float(b.val))
            assert np.array_equal(np.asarray(a.cells), np.asarray(b.cells))
        print("ok test_delegation_matches_calling_the_pieces_directly")

    test_completions_map_to_the_right_cells()
    test_controls_and_signs_are_applied()
    test_well_indices_are_positive()
    test_delegation_matches_calling_the_pieces_directly()
    print("\nAll tests passed!")
