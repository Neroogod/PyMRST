"""Flow-based partition from METIS (MRST ``partitionMETIS.m``).

Every other partition constructor in this module works on logical indices: they
split the model into boxes and take no notice of what is in them. This one
partitions the *connection graph*, weighted by transmissibility, so the block
boundaries follow the flow. A high-permeability channel stays inside one block
instead of being cut in half by a box boundary that happens to run across it.

The graph is the TPFA pressure matrix. MRST obtains it by calling `incompTPFA`
with a no-op linear solver purely to get the matrix out, which is a way of
reusing code rather than a step in the method. The port assembles it directly in
`connection_matrix`, which is the same matrix and needs no fluid, no state and
no solver, and which is asserted against MRST's own transmissibility harmonics.

`use_log` takes the logarithm of the transmissibilities first. Permeability
spans orders of magnitude, so without it a single strong connection dominates
the weights and METIS balances everything else against that one edge.

**`use_log` inverts the weights on SI input, in MRST and here.** A
transmissibility in SI is around 1e-14, so its logarithm is about -14, and
`callMetisMatrix` takes the absolute value before scaling. A strong connection
at 1e-11 becomes 11 and a weak one at 1e-14 becomes 14, so the strong connection
ends up with the *smaller* weight and METIS cuts it first. Measured on a 1000:1
channel: the strongest connection gets weight 7960 where the weakest gets 10000.
This port reproduces MRST rather than quietly correcting it, since a different
weighting is a different partition, but it warns when every transmissibility is
below 1, which is exactly when the inversion happens. Passing
transmissibilities in a unit system where they exceed 1 avoids it.

**METIS is an optional dependency** (`pip install pymrst[metis]`). Everything
here up to the partitioning call works without it, and the call itself raises
with a message naming the package. See `call_metis_matrix` for why a substitute
partitioner was rejected: it would change the partitions, which is the opposite
of parity.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.process_partition import process_partition
from pymrst.modules.coarsegrid.utils.call_metis_matrix import call_metis_matrix


def connection_matrix(g, t):
    """The TPFA pressure matrix for transmissibilities `t`.

    `t` may be one value per half-face, as `compute_trans` returns, or one per
    face, in which case it is used directly. Half-face values are combined
    harmonically, which is what makes the face transmissibility the series
    resistance of the two half-cells.
    """
    t = np.asarray(t, dtype=float).ravel()
    n_faces, n_cells = int(g.faces.num), int(g.cells.num)

    if t.size == n_faces:
        face_trans = t
    elif t.size == int(np.asarray(g.cells.faces).size):
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
        with np.errstate(divide="ignore"):
            face_trans = 1.0 / np.bincount(cell_faces, weights=1.0 / t,
                                           minlength=n_faces)
    else:
        raise ValueError(
            f"t must hold one value per face ({n_faces}) or per half-face "
            f"({np.asarray(g.cells.faces).size}), got {t.size}")

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    interior = np.flatnonzero(np.all(neighbors >= 0, axis=1))
    c1, c2 = neighbors[interior, 0], neighbors[interior, 1]
    v = face_trans[interior]

    rows = np.concatenate([c1, c2, c1, c2])
    cols = np.concatenate([c2, c1, c1, c2])
    data = np.concatenate([-v, -v, v, v])
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(n_cells, n_cells)).tocsr()


def partition_metis(g, t, n_blocks, use_log: bool = False, **metis_options):
    """Partition `g` into `n_blocks` blocks following its transmissibilities.

    Returns ``(partition, matrix)``: a 0-based partition and the connection
    matrix it came from, as MRST returns `[p, A]`.
    """
    requested = n_blocks
    n_blocks = int(np.ceil(n_blocks))
    if n_blocks != requested:
        warnings.warn(
            f"the number of blocks must be a whole number; rounding "
            f"{requested} up to {n_blocks}", RuntimeWarning, stacklevel=2)

    t = np.asarray(t, dtype=float).ravel()
    if use_log:
        if np.any(t <= 0.0):
            raise ValueError(
                "use_log needs strictly positive transmissibilities; "
                f"the smallest here is {t.min():.3e}")
        if t.max() < 1.0:
            warnings.warn(
                f"every transmissibility is below 1 (largest {t.max():.3e}), so "
                "their logarithms are all negative and the weights METIS sees "
                "are inverted: the strongest connections get the smallest "
                "weights and are cut first. This is MRST's behaviour. Supply "
                "transmissibilities in a unit system where they exceed 1 to "
                "avoid it.", RuntimeWarning, stacklevel=2)
        t = np.log10(t)

    a = connection_matrix(g, t)
    p = call_metis_matrix(a, n_blocks, **metis_options)
    return process_partition(g, compress_partition(p)), a


__all__ = ["connection_matrix", "partition_metis"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.utils.call_metis_matrix import HAVE_METIS
    from pymrst.params.rock.make_rock import make_rock
    from pymrst.solvers.compute_trans import compute_trans
    from pymrst.utils.units import DARCY, MILLI

    def _case(dims=(6, 6, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        rock = make_rock(g, perm=100 * MILLI * DARCY, poro=0.3)
        return g, compute_trans(g, rock)

    def test_the_matrix_has_one_entry_per_interior_connection():
        g, t = _case()
        a = connection_matrix(g, t)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        interior = int(np.count_nonzero(np.all(nb >= 0, axis=1)))
        off = a - sp.diags(a.diagonal())
        assert off.nnz == 2 * interior, (off.nnz, interior)
        print(f"ok test_the_matrix_has_one_entry_per_interior_connection "
              f"({interior} connections)")

    # Transmissibilities are ~1e-14 in SI, so `np.isclose`'s default atol of
    # 1e-8 calls any two of them equal. Every comparison below is relative.
    def _close(a, b):
        return np.allclose(a, b, rtol=1e-12, atol=0.0)

    def test_each_row_sums_to_zero():
        """The pressure operator annihilates a constant.

        A wrong sign or a missing diagonal contribution leaves a matrix that
        still looks like a Laplacian and no longer conserves. Measured against
        the row's own magnitude, since an absolute tolerance on values of order
        1e-14 would pass whatever the matrix held.
        """
        g, t = _case()
        a = connection_matrix(g, t)
        residual = np.abs(np.asarray(a.sum(axis=1)).ravel())
        scale = np.abs(a).sum(axis=1).A.ravel()
        assert np.all(residual <= 1e-12 * scale), (residual / scale).max()
        print("ok test_each_row_sums_to_zero")

    def test_the_matrix_is_symmetric():
        g, t = _case()
        a = connection_matrix(g, t)
        assert (a - a.T).nnz == 0, "the matrix is not exactly symmetric"
        print("ok test_the_matrix_is_symmetric")

    def test_off_diagonals_are_the_harmonic_face_transmissibility():
        """Stated as the definition, checked against it face by face.

        Averaging the half-face values arithmetically instead gives a matrix of
        the right shape whose connections are all twice too strong, and no
        structural test notices. The final assertion keeps the case honest: it
        fails if the fixture ever stops distinguishing the two.
        """
        g, t = _case()
        a = connection_matrix(g, t).tolil()
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
        nb = np.asarray(g.faces.neighbors, dtype=np.int64)
        for f in np.flatnonzero(np.all(nb >= 0, axis=1)):
            halves = t[cell_faces == f]
            assert halves.size == 2, f
            harmonic = 1.0 / np.sum(1.0 / halves)
            c1, c2 = nb[f]
            assert _close(a[c1, c2], -harmonic), (f, a[c1, c2], -harmonic)
            assert not _close(harmonic, halves.mean()), \
                "the case cannot tell harmonic from arithmetic"
        print("ok test_off_diagonals_are_the_harmonic_face_transmissibility")

    def test_face_transmissibilities_are_accepted_directly():
        g, t = _case()
        from_half = connection_matrix(g, t)
        cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
        face_trans = 1.0 / np.bincount(cell_faces, weights=1.0 / t,
                                       minlength=int(g.faces.num))
        from_face = connection_matrix(g, face_trans)
        assert _close(from_half.toarray(), from_face.toarray())
        print("ok test_face_transmissibilities_are_accepted_directly")

    def test_a_permeability_contrast_reaches_the_weights():
        """The whole point: the graph must know where the channel is.

        A partitioner given uniform weights returns a geometric partition, and
        it looks entirely reasonable.
        """
        from pymrst.modules.coarsegrid.utils.call_metis_matrix import metis_graph
        g = compute_geometry(cart_grid([6, 6, 1], [6.0, 6.0, 1.0]))
        perm = np.full(g.cells.num, MILLI * DARCY)
        c = np.asarray(g.cells.centroids, dtype=float)
        perm[(c[:, 1] > 2.0) & (c[:, 1] < 4.0)] = DARCY      # a channel in y
        rock = make_rock(g, perm=perm, poro=0.3)
        _, _, w = metis_graph(connection_matrix(g, compute_trans(g, rock)))
        assert w.max() > 10 * w.min(), (w.min(), w.max())
        print(f"ok test_a_permeability_contrast_reaches_the_weights "
              f"(weights {w.min()} to {w.max()})")

    def _channel_case(contrast=1000.0):
        from pymrst.modules.coarsegrid.utils.call_metis_matrix import metis_graph
        g = compute_geometry(cart_grid([6, 6, 1], [6.0, 6.0, 1.0]))
        perm = np.full(g.cells.num, MILLI * DARCY)
        c = np.asarray(g.cells.centroids, dtype=float)
        perm[(c[:, 1] > 2.0) & (c[:, 1] < 4.0)] = contrast * MILLI * DARCY
        rock = make_rock(g, perm=perm, poro=0.3)
        t = compute_trans(g, rock)
        return g, t, metis_graph

    def test_use_log_compresses_the_contrast():
        """Orders of magnitude become a manageable spread."""
        g, t, metis_graph = _channel_case()
        plain = metis_graph(connection_matrix(g, t))[2]
        logged = metis_graph(connection_matrix(g, np.log10(t)))[2]
        assert logged.min() / logged.max() > plain.min() / plain.max()
        print(f"ok test_use_log_compresses_the_contrast "
              f"(spread {plain.max() / plain.min():.0f}x to "
              f"{logged.max() / logged.min():.2f}x)")

    def test_use_log_inverts_the_weights_on_si_input():
        """MRST's behaviour, reproduced and pinned rather than corrected.

        SI transmissibilities are around 1e-14, so their logarithms are
        negative, and `metis_graph` takes the absolute value. The strongest
        connection therefore ends up with the smallest weight and is cut first.
        Nothing about the partition looks wrong afterwards; it simply follows
        the low-permeability rock.
        """
        g, t, metis_graph = _channel_case()
        plain = metis_graph(connection_matrix(g, t))[2]
        logged = metis_graph(connection_matrix(g, np.log10(t)))[2]
        strongest = int(np.argmax(plain))
        assert t.max() < 1.0, "the case is not in SI-scale units"
        assert logged[strongest] < logged.max(), \
            "the inversion is gone; the docstring and the warning need updating"
        print(f"ok test_use_log_inverts_the_weights_on_si_input "
              f"(strongest connection weighted {logged[strongest]} of "
              f"{logged.max()})")

    def test_the_inversion_is_warned_about():
        g, t = _case()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                partition_metis(g, t, 4, use_log=True)
            except ImportError:
                pass                      # pymetis absent; the warning still fired
            assert any("inverted" in str(w.message) for w in caught), \
                [str(w.message) for w in caught]
        print("ok test_the_inversion_is_warned_about")

    def test_no_warning_when_the_units_avoid_it():
        """Transmissibilities above 1 give positive logs and the right order."""
        g, t = _case()
        scaled = t / t.min() * 10.0            # every value now exceeds 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                partition_metis(g, scaled, 4, use_log=True)
            except ImportError:
                pass
            assert not any("inverted" in str(w.message) for w in caught)
        print("ok test_no_warning_when_the_units_avoid_it")

    def test_use_log_refuses_non_positive_transmissibilities():
        """log10 of a zero transmissibility is -inf, and METIS takes integers."""
        g, t = _case()
        t = t.copy()
        t[0] = 0.0
        try:
            partition_metis(g, t, 4, use_log=True)
        except ValueError as exc:
            assert "strictly positive" in str(exc), exc
        else:
            raise AssertionError("expected a zero transmissibility to be refused")
        print("ok test_use_log_refuses_non_positive_transmissibilities")

    def test_a_fractional_block_count_is_rounded_up_with_a_warning():
        g, t = _case()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                partition_metis(g, t, 3.5)
            except ImportError:
                pass                      # pymetis absent; the warning still fired
            assert any("rounding" in str(w.message) for w in caught), \
                [str(w.message) for w in caught]
        print("ok test_a_fractional_block_count_is_rounded_up_with_a_warning")

    def test_a_wrong_length_transmissibility_is_rejected():
        g, t = _case()
        try:
            connection_matrix(g, t[:-1])
        except ValueError as exc:
            assert "one value per face" in str(exc), exc
        else:
            raise AssertionError("expected a wrong-length t to fail")
        print("ok test_a_wrong_length_transmissibility_is_rejected")

    def test_partitioning_when_the_package_is_present():
        """Skipped without pymetis, and loudly."""
        if not HAVE_METIS:
            print("SKIP test_partitioning_when_the_package_is_present "
                  "(pymetis is not installed; this call path is unexercised)")
            return
        g, t = _case()
        p, a = partition_metis(g, t, 4)
        assert p.size == g.cells.num
        assert np.array_equal(process_partition(g, p), p), "a block is disconnected"
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        assert a.shape == (g.cells.num, g.cells.num)
        print(f"ok test_partitioning_when_the_package_is_present "
              f"({p.max() + 1} blocks)")

    test_the_matrix_has_one_entry_per_interior_connection()
    test_each_row_sums_to_zero()
    test_the_matrix_is_symmetric()
    test_off_diagonals_are_the_harmonic_face_transmissibility()
    test_face_transmissibilities_are_accepted_directly()
    test_a_permeability_contrast_reaches_the_weights()
    test_use_log_compresses_the_contrast()
    test_use_log_inverts_the_weights_on_si_input()
    test_the_inversion_is_warned_about()
    test_no_warning_when_the_units_avoid_it()
    test_use_log_refuses_non_positive_transmissibilities()
    test_a_fractional_block_count_is_rounded_up_with_a_warning()
    test_a_wrong_length_transmissibility_is_rejected()
    test_partitioning_when_the_package_is_present()
    print("\nAll tests passed!")
