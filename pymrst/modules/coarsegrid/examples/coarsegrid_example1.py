"""Introduction to Coarse Grids in MRST (``coarsegridExample1.m``).

A coarse grid is always defined as a partition of another grid, the fine grid.
This walks through the basics: partition a small Cartesian grid, build the
coarse structure, look at how blocks and interfaces are numbered, and then solve
the same pressure problem on both scales.

The last part is the one worth having as code rather than a picture. It shows
that a coarse grid is not only a plotting convenience: `compute_trans` and
`incomp_tpfa` take it directly, because its topology lives in the same fields a
fine grid uses. `CoarseGrid.to_grid` provides that view.

Two departures from the script, both because a figure is not a test:

* MRST labels cells and faces with `text` and leaves the reader to compare the
  numbers. Here `numbering` returns the labels and their positions, and the
  self-tests assert that a block's label sits inside the block.
* MRST plots the two pressure solutions side by side and leaves the comparison
  to the eye. Here they are compared numerically, which is what the figure is
  for: the coarse solution should reproduce the volume-averaged fine one block
  by block, and on this problem it does so to round-off, about 2e-13 bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
from pymrst.modules.coarsegrid.utils.coarsen_bc import coarsen_bc
from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
from pymrst.params.rock.make_rock import make_rock
from pymrst.params.wells_and_bc.pside import pside
from pymrst.solvers.compute_trans import compute_trans
from pymrst.solvers.init_state import init_state
from pymrst.utils.units import BARSA


def small_partition(dims=(4, 4), blocks=(2, 2)):
    """The tutorial's opening case: a 4x4 grid in 2x2 blocks."""
    g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
    p = partition_cart_grid(list(dims) + [1], list(blocks) + [1])
    return g, p, coarsen_geometry(generate_coarse_grid(g, p))


def numbering(cg):
    """Labels and positions for cells, blocks and both sets of faces.

    Returns a dict of ``(positions, labels)`` pairs, which is what MRST draws
    with `text`. Keeping it as data rather than as a figure is what lets the
    placement be checked.
    """
    from pymrst.modules.coarsegrid.plotting.plot_cell_numbers import (
        cell_number_labels,
    )
    from pymrst.modules.coarsegrid.plotting.plot_face_numbers import (
        face_number_labels,
    )

    g = cg.parent
    return {
        "fine cells": cell_number_labels(g),
        "blocks": cell_number_labels(cg),
        "fine faces": face_number_labels(g),
        "coarse faces": face_number_labels(cg),
    }


@dataclass
class TwoScaleSolution:
    """The same pressure problem solved on both grids."""

    g: Any
    cg: Any
    fine_pressure: np.ndarray
    coarse_pressure: np.ndarray

    def blockwise_fine_pressure(self) -> np.ndarray:
        """Fine pressure averaged over each block, for comparison."""
        p = np.asarray(self.cg.partition, dtype=np.int64)
        volumes = np.asarray(self.g.cells.volumes, dtype=float)
        weighted = np.bincount(p, weights=self.fine_pressure * volumes,
                               minlength=self.cg.n_blocks)
        return weighted / np.bincount(p, weights=volumes,
                                      minlength=self.cg.n_blocks)


def two_scale_pressure(dims=(20, 20), blocks=(5, 2),
                       inlet: float = 100 * BARSA) -> TwoScaleSolution:
    """Solve a left-to-right pressure drop on the fine and coarse grids.

    Permeability is uniform, so no upscaling is needed and the coarse
    permeability is simply the same constant. With a real field one would
    upscale here instead, which is what the `upscaling` module is for.
    """
    from pymrst.modules.incomp import incomp_tpfa, init_single_fluid

    g = compute_geometry(cart_grid(list(dims), [1.0, 1.0]))
    fluid = init_single_fluid(mu=1.0, rho=1.0)

    rock = make_rock(g, perm=np.ones(g.cells.num), poro=0.3)
    bc = pside(None, g, "left", inlet)
    bc = pside(bc, g, "right", 0.0)
    state = incomp_tpfa(init_state(g, None, 0.0), g,
                        compute_trans(g, rock), fluid, bc=bc)

    p = partition_cart_grid(list(dims) + [1], list(blocks) + [1])
    cg = coarsen_geometry(generate_coarse_grid(g, p))
    coarse = cg.to_grid()
    coarse_rock = make_rock(coarse, perm=np.ones(coarse.cells.num), poro=0.3)
    coarse_state = incomp_tpfa(init_state(coarse, None, 0.0), coarse,
                               compute_trans(coarse, coarse_rock), fluid,
                               bc=coarsen_bc(cg, bc))

    return TwoScaleSolution(g=g, cg=cg,
                            fine_pressure=np.asarray(state.pressure, dtype=float),
                            coarse_pressure=np.asarray(coarse_state.pressure,
                                                       dtype=float))


def plot_example1(**kwargs):
    """Draw the tutorial's figures. ``None`` without a plotting backend."""
    try:
        from pymrst.plotting import plot_cell_data
    except ImportError:
        return None
    g, p, cg = small_partition()
    return [plot_cell_data(g, p.astype(float), **kwargs)]


__all__ = ["TwoScaleSolution", "numbering", "plot_example1", "small_partition",
           "two_scale_pressure"]


if __name__ == "__main__":
    def test_the_opening_case_has_four_blocks():
        g, p, cg = small_partition()
        assert g.cells.num == 16 and cg.n_blocks == 4
        assert np.array_equal(np.bincount(p), np.full(4, 4)), np.bincount(p)
        print(f"ok test_the_opening_case_has_four_blocks "
              f"({cg.num_faces} coarse faces)")

    def test_coarsen_geometry_conserves_volume():
        g, _, cg = small_partition()
        assert np.isclose(cg.cell_volumes.sum(), float(np.sum(g.cells.volumes)),
                          rtol=1e-12)
        print("ok test_coarsen_geometry_conserves_volume")

    def test_every_label_sits_on_the_thing_it_names():
        """The figure's whole content, checked instead of drawn.

        A block label drawn at the wrong centroid gives a picture that reads
        perfectly and identifies the wrong block.
        """
        g, p, cg = small_partition()
        marks = numbering(cg)
        centroids = np.asarray(g.cells.centroids, dtype=float)
        positions, labels = marks["blocks"]
        for b, label in enumerate(labels):
            assert label == str(b)
            inside = centroids[p == b]
            assert np.all(positions[b] >= inside.min(axis=0) - 1e-9), b
            assert np.all(positions[b] <= inside.max(axis=0) + 1e-9), b
        assert len(marks["fine cells"][1]) == g.cells.num
        assert len(marks["coarse faces"][1]) == cg.num_faces
        print(f"ok test_every_label_sits_on_the_thing_it_names "
              f"({len(labels)} blocks, {cg.num_faces} coarse faces)")

    def test_the_coarse_grid_solves_the_same_problem():
        """The point of the last section, and it is a numerical claim.

        MRST plots the two solutions and leaves the comparison to the eye.
        With uniform permeability the coarse solution should reproduce the
        fine one block by block.
        """
        s = two_scale_pressure()
        fine = s.blockwise_fine_pressure()
        assert np.allclose(s.coarse_pressure, fine, rtol=1e-6), (
            np.abs(s.coarse_pressure - fine).max() / BARSA)
        print(f"ok test_the_coarse_grid_solves_the_same_problem "
              f"(worst block differs by "
              f"{np.abs(s.coarse_pressure - fine).max() / BARSA:.2e} bar)")

    def test_the_pressure_drop_is_reproduced():
        s = two_scale_pressure()
        assert np.isclose(s.fine_pressure.max() / BARSA, 97.5)
        assert np.isclose(s.fine_pressure.min() / BARSA, 2.5)
        # Five blocks across, so the coarse extremes sit further in.
        assert np.isclose(s.coarse_pressure.max() / BARSA, 90.0)
        assert np.isclose(s.coarse_pressure.min() / BARSA, 10.0)
        print(f"ok test_the_pressure_drop_is_reproduced "
              f"(fine 2.5 to 97.5 bar, coarse 10.0 to 90.0 bar)")

    def test_the_solution_is_monotone_left_to_right():
        """Flow runs from the high-pressure face to the low one.

        A sign error in `coarsen_bc` or in the coarse normals would reverse it
        while leaving a plausible-looking pressure field.
        """
        s = two_scale_pressure()
        x = np.asarray(s.cg.cell_centroids, dtype=float)[:, 0]
        order = np.argsort(x)
        p = s.coarse_pressure[order]
        assert np.all(np.diff(p) <= 1e-9), p
        print("ok test_the_solution_is_monotone_left_to_right")

    def test_the_boundary_conditions_survive_coarsening():
        """Both sides must still be represented after coarsening."""
        from pymrst.modules.incomp import init_single_fluid  # noqa: F401
        g = compute_geometry(cart_grid([20, 20], [1.0, 1.0]))
        bc = pside(None, g, "left", 100 * BARSA)
        bc = pside(bc, g, "right", 0.0)
        p = partition_cart_grid([20, 20, 1], [5, 2, 1])
        cg = coarsen_geometry(generate_coarse_grid(g, p))
        coarse_bc = coarsen_bc(cg, bc)
        assert coarse_bc is not None
        values = np.asarray(coarse_bc.value, dtype=float) / BARSA
        assert set(np.round(values, 6).tolist()) == {100.0, 0.0}, values
        assert set(coarse_bc.type) == {"pressure"}
        print(f"ok test_the_boundary_conditions_survive_coarsening "
              f"({coarse_bc.face.size} coarse conditions from {bc.face.size} fine)")

    def test_plotting_is_optional():
        """`show=False` keeps this finite; see `explosion_view` for why."""
        out = plot_example1(show=False)
        assert out is None or isinstance(out, list), type(out)
        print("ok test_plotting_is_optional")

    test_the_opening_case_has_four_blocks()
    test_coarsen_geometry_conserves_volume()
    test_every_label_sits_on_the_thing_it_names()
    test_the_coarse_grid_solves_the_same_problem()
    test_the_pressure_drop_is_reproduced()
    test_the_solution_is_monotone_left_to_right()
    test_the_boundary_conditions_survive_coarsening()
    test_plotting_is_optional()
    print("\nAll tests passed!")
