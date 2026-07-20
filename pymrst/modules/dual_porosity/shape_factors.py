"""Shape factors for dual-porosity transfer (MRST dual-porosity/).

A naturally fractured rock is modelled as two overlapping continua: fast,
low-storage fractures that carry the flow, and slow, high-storage matrix
blocks that hold the fluid. What couples them is a transfer term

    T = sigma * (b * kr / mu) * (p_fracture - p_matrix)

and the shape factor sigma is the geometry of that exchange - how much
matrix surface the fractures see, given a fracture spacing (lx, ly, lz).

By MRST's convention sigma **carries the matrix permeability**, so it has
units of k/L^2 and the transfer term comes out as a rate. All the shape
factors here follow that convention (see `lim_aziz` for the one place MRST
does not).
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

from pymrst.params.rock import Rock

Spacing = Union[float, np.ndarray]


def _lengths(block_dimension: Spacing, nc: int) -> np.ndarray:
    """Fracture spacing as (nc, 3)."""
    l = np.atleast_2d(np.asarray(block_dimension, dtype=np.float64))
    if l.shape == (1, 1):
        l = np.tile(l, (1, 3))
    if l.shape[1] != 3:
        raise ValueError(f"fracture spacing must be (lx, ly, lz), got {l.shape}")
    if l.shape[0] == 1:
        l = np.tile(l, (nc, 1))
    if l.shape[0] != nc:
        raise ValueError(f"expected 1 or {nc} rows of spacing, got {l.shape[0]}")
    if np.any(l <= 0):
        raise ValueError("fracture spacing must be positive")
    return l


def _perms(rock: Rock, nc: int) -> np.ndarray:
    """Matrix permeability as (nc, 3); an isotropic rock is broadcast."""
    k = np.atleast_2d(np.asarray(rock.perm, dtype=np.float64))
    if k.shape[0] != nc:
        raise ValueError(f"rock has {k.shape[0]} cells, expected {nc}")
    if k.shape[1] == 1:
        return np.tile(k, (1, 3))
    if k.shape[1] >= 3:
        return k[:, :3]                      # diagonal of a full tensor
    raise ValueError(f"cannot read a permeability of shape {k.shape}")


@dataclass
class ShapeFactor:
    """Base class: sigma = 1 (MRST's ShapeFactor)."""

    block_dimension: Spacing = 1.0

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return np.ones(np.atleast_2d(rock_matrix.perm).shape[0])


@dataclass
class ConstantShapeFactor(ShapeFactor):
    """A shape factor the user supplies outright."""

    value: float = 1.0

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        nc = np.atleast_2d(rock_matrix.perm).shape[0]
        return np.full(nc, float(self.value))


def _anisotropic(rock_matrix: Rock, block_dimension: Spacing,
                 coefficient: float) -> np.ndarray:
    """coefficient * (kx/lx^2 + ky/ly^2 + kz/lz^2).

    Kazemi, Warren-Root and Coats differ only in the coefficient. For an
    isotropic rock this collapses to the k * sigma quoted in the papers.
    """
    nc = np.atleast_2d(rock_matrix.perm).shape[0]
    k = _perms(rock_matrix, nc)
    l = _lengths(block_dimension, nc)
    return coefficient * (k / l**2).sum(axis=1)


@dataclass
class KazemiShapeFactor(ShapeFactor):
    """Kazemi (1976): sigma = 4 * sum(k_i / l_i^2)."""

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return _anisotropic(rock_matrix, self.block_dimension, 4.0)


@dataclass
class WarrenRootShapeFactor(ShapeFactor):
    """Warren & Root (1963), one set of normal fractures: 12 * sum(k_i/l_i^2)."""

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return _anisotropic(rock_matrix, self.block_dimension, 12.0)


@dataclass
class CoatsShapeFactor(ShapeFactor):
    """Coats (1989): sigma = 8 * sum(k_i / l_i^2)."""

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        return _anisotropic(rock_matrix, self.block_dimension, 8.0)


@dataclass
class LimAzizShapeFactor(ShapeFactor):
    """Lim & Aziz (1995): sigma = pi^2 * sum(k_i / l_i^2).

    A deliberate departure from MRST, which computes

        sigma = pi^2 / (kx ky kz)^(1/3) * sum(k_i / l_i^2)

    i.e. it divides the permeability back out, leaving the classical
    dimensionless-in-k form of 1/L^2. Every transfer function in the module
    assumes sigma already carries the permeability (KazemiTwoPhaseTransfer
    even reads `km = rock_matrix.perm(:,1)` and then never uses it), so
    MRST's LimAziz is inconsistent with its own transfer functions: with a
    1 darcy matrix it over-predicts transfer by a factor of ~1e12, and the
    matrix equilibrates with the fracture instantly.

    We keep the permeability in, which is dimensionally consistent with the
    other shape factors and reduces to pi^2 k (1/lx^2 + 1/ly^2 + 1/lz^2) for
    an isotropic rock, as Lim & Aziz give it. Set `mrst_compatible=True` to
    reproduce MRST's number exactly.
    """

    mrst_compatible: bool = False

    def sigma(self, rock_matrix: Rock) -> np.ndarray:
        s = _anisotropic(rock_matrix, self.block_dimension, np.pi**2)
        if not self.mrst_compatible:
            return s
        nc = np.atleast_2d(rock_matrix.perm).shape[0]
        k = _perms(rock_matrix, nc)
        return s / np.cbrt(k.prod(axis=1))     # MRST divides the perm out


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, MILLI

    def _rock(g, kx, ky=None, kz=None):
        if ky is None:
            return make_rock(g, np.full((g.cells.num, 1), kx),
                             np.full((g.cells.num, 1), 0.1))
        k = np.tile([kx, ky, kz], (g.cells.num, 1))
        return make_rock(g, k, np.full((g.cells.num, 1), 0.1))

    G = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))

    def test_isotropic_collapses_to_the_textbook_formula():
        """For isotropic k, sigma = C * k * sum(1/l_i^2), as the papers give."""
        k = 1 * DARCY
        rock = _rock(G, k)
        lx, ly, lz = 10.0, 20.0, 5.0
        geom = 1 / lx**2 + 1 / ly**2 + 1 / lz**2

        for cls, c in ((KazemiShapeFactor, 4.0), (CoatsShapeFactor, 8.0),
                       (WarrenRootShapeFactor, 12.0),
                       (LimAzizShapeFactor, np.pi**2)):
            sf = cls(block_dimension=[lx, ly, lz])
            assert np.allclose(sf.sigma(rock), c * k * geom, rtol=1e-14), cls
        print("ok test_isotropic_collapses_to_the_textbook_formula")

    def test_anisotropic_permeability():
        """Each direction is weighted by its own permeability."""
        kx, ky, kz = 1 * DARCY, 0.1 * DARCY, 0.01 * DARCY
        rock = _rock(G, kx, ky, kz)
        sf = KazemiShapeFactor(block_dimension=[2.0, 4.0, 8.0])
        want = 4.0 * (kx / 4.0 + ky / 16.0 + kz / 64.0)
        assert np.allclose(sf.sigma(rock), want, rtol=1e-14)
        print("ok test_anisotropic_permeability")

    def test_ordering_of_the_coefficients():
        """Kazemi < Coats < Warren-Root, as their coefficients say."""
        rock = _rock(G, 100 * MILLI * DARCY)
        d = [5.0, 5.0, 5.0]
        ka = KazemiShapeFactor(block_dimension=d).sigma(rock)
        co = CoatsShapeFactor(block_dimension=d).sigma(rock)
        wr = WarrenRootShapeFactor(block_dimension=d).sigma(rock)
        assert np.all(ka < co) and np.all(co < wr)
        assert np.allclose(co, 2 * ka) and np.allclose(wr, 3 * ka)
        print("ok test_ordering_of_the_coefficients")

    def test_lim_aziz_carries_permeability_but_mrst_does_not():
        """The documented departure from MRST, pinned down.

        MRST's LimAziz divides the permeability out, so it is a factor 1/k
        away from every other shape factor - about 1e12 for a darcy rock.
        """
        k = 1 * DARCY
        rock = _rock(G, k)
        d = [10.0, 10.0, 10.0]

        ours = LimAzizShapeFactor(block_dimension=d).sigma(rock)
        theirs = LimAzizShapeFactor(block_dimension=d,
                                    mrst_compatible=True).sigma(rock)
        kaz = KazemiShapeFactor(block_dimension=d).sigma(rock)

        # ours sits alongside the others: same units, same order
        assert np.allclose(ours / kaz, np.pi**2 / 4.0, rtol=1e-14)
        # MRST's is 1/k times that - a factor of 1e12 for this rock
        assert np.allclose(theirs, ours / k, rtol=1e-14)
        assert theirs[0] / ours[0] > 1e11
        print(f"ok test_lim_aziz_carries_permeability_but_mrst_does_not "
              f"(MRST/ours = {theirs[0] / ours[0]:.1e})")

    def test_constant_and_base():
        """The constant and base shape factors ignore the rock."""
        rock = _rock(G, 1 * DARCY)
        assert np.allclose(ConstantShapeFactor(value=0.25).sigma(rock), 0.25)
        assert np.allclose(ShapeFactor().sigma(rock), 1.0)
        print("ok test_constant_and_base")

    def test_per_cell_spacing_and_bad_input():
        """Spacing may vary by cell; nonsense is rejected."""
        rock = _rock(G, 1 * DARCY)
        d = np.tile([10.0, 10.0, 10.0], (G.cells.num, 1))
        d[0] = [5.0, 5.0, 5.0]                    # one tighter block
        s = KazemiShapeFactor(block_dimension=d).sigma(rock)
        assert s[0] == 4 * s[1], "halving the spacing must quadruple sigma"

        for bad in ([0.0, 1.0, 1.0], [-1.0, 1.0, 1.0]):
            try:
                KazemiShapeFactor(block_dimension=bad).sigma(rock)
            except ValueError:
                pass
            else:
                raise AssertionError("non-positive spacing should raise")
        print("ok test_per_cell_spacing_and_bad_input")

    test_isotropic_collapses_to_the_textbook_formula()
    test_anisotropic_permeability()
    test_ordering_of_the_coefficients()
    test_lim_aziz_carries_permeability_but_mrst_does_not()
    test_constant_and_base()
    test_per_cell_spacing_and_bad_input()
    print("\nAll tests passed!")
