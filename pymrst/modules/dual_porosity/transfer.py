"""Matrix-fracture transfer functions (MRST dual-porosity/transfer_models).

The transfer term is what makes a dual-porosity model more than two
uncoupled simulations. Per unit bulk volume,

    T_alpha = sigma * (b_alpha * kr_alpha / mu_alpha) * (p_alpha,f - p_alpha,m)

with sigma the shape factor (which carries the matrix permeability, see
shape_factors). Two details matter numerically:

  - the relative permeability is *upstream-weighted*: the fracture's kr is
    used where flow runs fracture -> matrix and the matrix's kr where it
    runs the other way. Without this, imbibition into a dry matrix would be
    throttled by the matrix's own (near-zero) kr and never start.

  - capillary pressure enters through the phase pressures, and in an
    imbibition problem it *is* the driving force: with pcOW_matrix > 0 the
    matrix sucks water in even when the oil pressures are equal.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from pymrst.modules.ad_props.init_simple_adi_fluid import ADFluid
from pymrst.modules.dual_porosity.shape_factors import ShapeFactor
from pymrst.params.rock import Rock
from pymrst.utils.adi import value


def _upstream(from_fracture: np.ndarray, kr_fracture, kr_matrix):
    """kr of whichever side the flow comes from (MRST's krwt / krot)."""
    f = from_fracture.astype(np.float64)
    return f * kr_fracture + (1.0 - f) * kr_matrix


@dataclass
class KazemiSinglePhaseTransfer:
    """Single-phase transfer (MRST KazemiSinglePhaseTransferFunction)."""

    shape_factor: ShapeFactor

    def transfer(self, rock_matrix: Rock, fluid_matrix: ADFluid,
                 p_fracture, p_matrix):
        """T_w per unit bulk volume, positive when the fracture feeds the matrix."""
        sigma = self.shape_factor.sigma(rock_matrix)
        b_wm = fluid_matrix.b_w(p_matrix)
        mu_wm = fluid_matrix.mu[0]
        return (sigma * b_wm / mu_wm) * (p_fracture - p_matrix)


@dataclass
class KazemiTwoPhaseTransfer:
    """Two-phase oil-water transfer (MRST KazemiTwoPhaseTransferFunction)."""

    shape_factor: ShapeFactor
    pc_ow_fracture: Optional[Callable] = None
    pc_ow_matrix: Optional[Callable] = None

    def transfer(self, rock_matrix: Rock, fluid_fracture: ADFluid,
                 fluid_matrix: ADFluid, po_fracture, sw_fracture,
                 po_matrix, sw_matrix) -> Tuple:
        """(T_w, T_o) per unit bulk volume, positive fracture -> matrix."""
        so_f = 1.0 - sw_fracture
        so_m = 1.0 - sw_matrix

        pc_f = (0.0 if self.pc_ow_fracture is None
                else self.pc_ow_fracture(sw_fracture))
        pc_m = (0.0 if self.pc_ow_matrix is None
                else self.pc_ow_matrix(sw_matrix))
        pw_fracture = po_fracture - pc_f
        pw_matrix = po_matrix - pc_m

        kr_w_f = fluid_fracture.kr_w(sw_fracture)
        kr_o_f = fluid_fracture.kr_o(so_f)
        kr_w_m = fluid_matrix.kr_w(sw_matrix)
        kr_o_m = fluid_matrix.kr_o(so_m)

        # upstream direction, per phase and per cell
        w_from_fracture = value(pw_matrix - pw_fracture) <= 0
        o_from_fracture = value(po_matrix - po_fracture) <= 0
        kr_wt = _upstream(w_from_fracture, kr_w_f, kr_w_m)
        kr_ot = _upstream(o_from_fracture, kr_o_f, kr_o_m)

        mu_wm, mu_om = fluid_matrix.mu[0], fluid_matrix.mu[1]
        # MRST evaluates these b-factors on the *fracture* fluid even though
        # they are taken at matrix pressures; kept as-is. In every MRST
        # example the two fluids are identical, so it makes no difference.
        b_wm = fluid_fracture.b_w(pw_matrix)
        b_om = fluid_fracture.b_o(po_matrix)

        sigma = self.shape_factor.sigma(rock_matrix)
        t_w = (sigma * b_wm * kr_wt / mu_wm) * (pw_fracture - pw_matrix)
        t_o = (sigma * b_om * kr_ot / mu_om) * (po_fracture - po_matrix)
        return t_w, t_o


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.modules.dual_porosity.shape_factors import KazemiShapeFactor
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    G = compute_geometry(cart_grid([5, 1, 1], [5.0, 1.0, 1.0]))
    NC = G.cells.num
    ROCK = make_rock(G, np.full((NC, 1), 1 * DARCY), np.full((NC, 1), 0.3))
    SF = KazemiShapeFactor(block_dimension=[10.0, 10.0, 10.0])

    def _fluid(c=(1e-8, 1e-8)):
        return init_simple_adi_fluid(
            mu=[1 * CENTI * POISE, 5 * CENTI * POISE], rho=[1000.0, 800.0],
            n=[2.0, 2.0], c=list(c), p_ref=100 * BARSA)

    def test_single_phase_sign_and_magnitude():
        """Transfer runs from high pressure to low, and vanishes at equality."""
        f = _fluid()
        tf = KazemiSinglePhaseTransfer(shape_factor=SF)
        pm = np.full(NC, 100 * BARSA)

        hot = tf.transfer(ROCK, f, np.full(NC, 120 * BARSA), pm)
        assert np.all(hot > 0), "fracture above matrix must feed the matrix"

        cold = tf.transfer(ROCK, f, np.full(NC, 80 * BARSA), pm)
        assert np.all(cold < 0), "matrix above fracture must feed the fracture"

        none = tf.transfer(ROCK, f, pm.copy(), pm)
        assert np.allclose(none, 0.0)

        # against the closed form
        sigma = SF.sigma(ROCK)
        want = sigma * f.b_w(pm) / f.mu[0] * (120 * BARSA - 100 * BARSA)
        assert np.allclose(hot, want, rtol=1e-14)
        print("ok test_single_phase_sign_and_magnitude")

    def test_single_phase_derivatives():
        """The transfer differentiates cleanly through the AD layer."""
        f = _fluid()
        tf = KazemiSinglePhaseTransfer(shape_factor=SF)
        pf0 = np.full(NC, 120 * BARSA)
        pm0 = np.full(NC, 100 * BARSA)

        pf, pm = init_variables_adi(pf0, pm0)
        t = tf.transfer(ROCK, f, pf, pm)
        j_f = np.asarray(t.jac[0].todense())
        j_m = np.asarray(t.jac[1].todense())

        h = 1.0
        fd_f = (tf.transfer(ROCK, f, pf0 + h, pm0)
                - tf.transfer(ROCK, f, pf0 - h, pm0)) / (2 * h)
        fd_m = (tf.transfer(ROCK, f, pf0, pm0 + h)
                - tf.transfer(ROCK, f, pf0, pm0 - h)) / (2 * h)
        assert np.allclose(np.diag(j_f), fd_f, rtol=1e-6)
        assert np.allclose(np.diag(j_m), fd_m, rtol=1e-6)
        print("ok test_single_phase_derivatives")

    def test_two_phase_upstream_relperm():
        """kr is taken from whichever side the flow comes from.

        A dry matrix (sW = 0) fed by a wet fracture must transfer water: if
        the matrix's own kr_w(0) = 0 were used, imbibition could never start.
        """
        ff, fm = _fluid(), _fluid()
        tf = KazemiTwoPhaseTransfer(shape_factor=SF)

        sw_f = np.full(NC, 0.9)               # fracture full of water
        sw_m = np.zeros(NC)                   # matrix bone dry
        po_f = np.full(NC, 120 * BARSA)
        po_m = np.full(NC, 100 * BARSA)

        t_w, t_o = tf.transfer(ROCK, ff, fm, po_f, sw_f, po_m, sw_m)
        assert np.all(t_w > 0), "water must enter the dry matrix"
        assert np.allclose(t_w, SF.sigma(ROCK) * ff.b_w(po_m)
                           * ff.kr_w(sw_f) / fm.mu[0]
                           * (po_f - po_m), rtol=1e-14), \
            "the fracture's kr_w should have been used"

        # reverse the gradient: now the matrix is upstream and its kr rules
        t_w2, _ = tf.transfer(ROCK, ff, fm, po_m, sw_f, po_f, sw_m)
        assert np.allclose(t_w2, 0.0), \
            "with the dry matrix upstream, kr_w(0) = 0 must stop the transfer"
        print("ok test_two_phase_upstream_relperm")

    def test_capillary_imbibition_drives_transfer():
        """With equal oil pressures, matrix pc alone pulls water in."""
        ff, fm = _fluid(), _fluid()
        Pe = 5 * BARSA
        tf = KazemiTwoPhaseTransfer(shape_factor=SF,
                                    pc_ow_matrix=lambda sw: Pe)

        p = np.full(NC, 100 * BARSA)
        sw_f = np.full(NC, 0.8)
        sw_m = np.full(NC, 0.2)

        t_w, t_o = tf.transfer(ROCK, ff, fm, p, sw_f, p.copy(), sw_m)
        # water pressure in the matrix is p - Pe, below the fracture's p,
        # so water is drawn in and oil is pushed back out
        assert np.all(t_w > 0), "capillarity must imbibe water"
        assert np.allclose(t_o, 0.0), "equal oil pressures, so no oil transfer"

        # no capillary pressure -> nothing happens at all
        flat = KazemiTwoPhaseTransfer(shape_factor=SF)
        t_w0, t_o0 = flat.transfer(ROCK, ff, fm, p, sw_f, p.copy(), sw_m)
        assert np.allclose(t_w0, 0.0) and np.allclose(t_o0, 0.0)
        print("ok test_capillary_imbibition_drives_transfer")

    def test_two_phase_derivatives():
        """The two-phase transfer differentiates in all four variables."""
        ff, fm = _fluid(), _fluid()
        tf = KazemiTwoPhaseTransfer(shape_factor=SF,
                                    pc_ow_matrix=lambda sw: 5 * BARSA * (1 - sw))
        x0 = [np.full(NC, 120 * BARSA), np.full(NC, 0.7),
              np.full(NC, 100 * BARSA), np.full(NC, 0.3)]

        pf, swf, pm, swm = init_variables_adi(*x0)
        t_w, _ = tf.transfer(ROCK, ff, fm, pf, swf, pm, swm)

        for k, h in enumerate((1.0, 1e-6, 1.0, 1e-6)):
            up = [x.copy() for x in x0]
            dn = [x.copy() for x in x0]
            up[k] += h
            dn[k] -= h
            fd = (tf.transfer(ROCK, ff, fm, *up)[0]
                  - tf.transfer(ROCK, ff, fm, *dn)[0]) / (2 * h)
            ad = np.diag(np.asarray(t_w.jac[k].todense()))
            scale = max(np.abs(fd).max(), 1e-30)
            assert np.abs(ad - fd).max() / scale < 1e-5, f"variable {k}"
        print("ok test_two_phase_derivatives")

    test_single_phase_sign_and_magnitude()
    test_single_phase_derivatives()
    test_two_phase_upstream_relperm()
    test_capillary_imbibition_drives_transfer()
    test_two_phase_derivatives()
    print("\nAll tests passed!")
