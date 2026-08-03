"""Fine fluxes summed onto coarse faces (MRST ``coarsenFlux.m``).

The flux through a coarse face is the signed sum of the fluxes through the fine
faces behind it, signed because a fine face can be oriented against the coarse
face it belongs to. See `fine_to_coarse_sign`.

This is exact, not an approximation: mass crossing a coarse face is exactly the
mass crossing its fine faces, so a coarse flux field built this way satisfies
the same balance on the blocks that the fine field satisfied on the cells. That
makes it the natural reference for any upscaling method, which is what MRST
uses it for.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign


def coarsen_flux(cg, flux) -> np.ndarray:
    """Sum `flux`, one value per fine face, onto the coarse faces."""
    if cg.parent is None:
        raise ValueError("the coarse grid has no parent; coarsening needs it")
    flux = np.asarray(flux, dtype=float)
    if flux.ndim == 2 and flux.shape[1] == 1:
        flux = flux.ravel()
    if flux.ndim != 1 or flux.size != int(cg.parent.faces.num):
        raise ValueError(
            f"flux must hold one value per fine face ({cg.parent.faces.num}), "
            f"got shape {np.shape(flux)}")

    counts = np.diff(np.asarray(cg.conn_pos, dtype=np.int64))
    face_no = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    contribution = flux[cg.fconn] * fine_to_coarse_sign(cg)
    return np.bincount(face_no, weights=contribution, minlength=counts.size)


__all__ = ["coarsen_flux"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(6, 4, 2), part=(3, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        p = partition_cart_grid(list(dims), list(part))
        return g, p, generate_coarse_grid(g, p)

    def _divergence(neighbors, flux, n):
        """Net outflow per cell, with -1 meaning outside."""
        out = np.zeros(n)
        a, b = neighbors[:, 0], neighbors[:, 1]
        np.add.at(out, a[a >= 0], flux[a >= 0])
        np.subtract.at(out, b[b >= 0], flux[b >= 0])
        return out

    def test_a_balanced_fine_field_stays_balanced():
        """The property that makes this exact rather than approximate.

        A divergence-free fine flux must give a divergence-free coarse flux.
        A wrong sign leaves the coarse field finite and plausible, and only
        the balance catches it.
        """
        g, p, cg = _case()
        # Uniform flow along x: exactly divergence-free on a Cartesian grid,
        # with no linear solve to introduce a tolerance of its own.
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        normals = np.asarray(g.faces.normals, dtype=float)
        flux = normals[:, 0]                     # unit velocity along x
        div = _divergence(nb, flux, g.cells.num)
        assert np.allclose(div, 0.0, atol=1e-9), np.abs(div).max()

        cflux = coarsen_flux(cg, flux)
        cdiv = _divergence(cg.neighbors, cflux, cg.n_blocks)
        assert np.allclose(cdiv, 0.0, atol=1e-9), np.abs(cdiv).max()
        print(f"ok test_a_balanced_fine_field_stays_balanced "
              f"(max |div| {np.abs(cdiv).max():.2e})")

    def test_block_inflow_equals_the_fine_inflow():
        """Whatever crossed a block's boundary must still cross it.

        Checked against the fine grid directly rather than through the coarse
        topology, so a consistent-but-wrong grouping would not hide.
        """
        g, p, cg = _case()
        rng = np.random.default_rng(1)
        flux = rng.normal(size=int(g.faces.num))
        cflux = coarsen_flux(cg, flux)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        fine_div = _divergence(nb, flux, g.cells.num)
        coarse_div = _divergence(cg.neighbors, cflux, cg.n_blocks)
        for b in range(cg.n_blocks):
            assert np.isclose(coarse_div[b], fine_div[p == b].sum(), atol=1e-9), b
        print("ok test_block_inflow_equals_the_fine_inflow")

    def test_a_constant_field_is_summed_not_averaged():
        g, _, cg = _case()
        flux = np.ones(int(g.faces.num))
        cflux = coarsen_flux(cg, flux)
        from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import (
            fine_to_coarse_sign,
        )
        sgn = fine_to_coarse_sign(cg)
        counts = np.diff(cg.conn_pos)
        face_no = np.repeat(np.arange(cg.num_faces), counts)
        expected = np.bincount(face_no, weights=sgn, minlength=cg.num_faces)
        assert np.allclose(cflux, expected)
        print("ok test_a_constant_field_is_summed_not_averaged")

    def test_it_is_linear():
        g, _, cg = _case()
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=int(g.faces.num)), rng.normal(size=int(g.faces.num))
        assert np.allclose(coarsen_flux(cg, 3.0 * a - b),
                           3.0 * coarsen_flux(cg, a) - coarsen_flux(cg, b))
        print("ok test_it_is_linear")

    def test_intra_block_flux_is_discarded():
        """Flow inside a block is not flow across any coarse face."""
        g, p, cg = _case()
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        inside = np.all(nb >= 0, axis=1) & (p[np.maximum(nb[:, 0], 0)]
                                            == p[np.maximum(nb[:, 1], 0)])
        assert np.any(inside)
        flux = np.zeros(int(g.faces.num))
        flux[inside] = 1.0
        assert np.allclose(coarsen_flux(cg, flux), 0.0)
        print(f"ok test_intra_block_flux_is_discarded ({inside.sum()} faces)")

    def test_a_column_vector_is_accepted():
        g, _, cg = _case()
        flux = np.ones((int(g.faces.num), 1))
        assert coarsen_flux(cg, flux).shape == (cg.num_faces,)
        print("ok test_a_column_vector_is_accepted")

    def test_a_wrong_length_flux_is_rejected():
        g, _, cg = _case()
        try:
            coarsen_flux(cg, np.ones(int(g.faces.num) - 1))
        except ValueError as exc:
            assert "one value per fine face" in str(exc), exc
        else:
            raise AssertionError("expected a wrong-length flux to fail")
        print("ok test_a_wrong_length_flux_is_rejected")

    test_a_balanced_fine_field_stays_balanced()
    test_block_inflow_equals_the_fine_inflow()
    test_a_constant_field_is_summed_not_averaged()
    test_it_is_linear()
    test_intra_block_flux_is_discarded()
    test_a_column_vector_is_accepted()
    test_a_wrong_length_flux_is_rejected()
    print("\nAll tests passed!")
