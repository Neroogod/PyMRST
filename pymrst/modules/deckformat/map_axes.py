"""Map local grid coordinates onto a survey coordinate system.

Port of MRST ``mapAxes.m``.

ECLIPSE decks give the `MAPAXES` keyword as six numbers, three points laid out
as `[x_y, y_y, x_o, y_o, x_x, y_x]`: the origin sits in the middle pair, with a
point on the y axis first and a point on the x axis last. The two axis
directions are formed relative to the origin, normalised, and used to place
each local `(x, y)` into the survey frame.

A left-handed `MAPAXES` flips the orientation of the coordinate system, which
silently mirrors the model. MRST warns rather than refusing, and the port keeps
that behaviour so a deck that loads in MRST also loads here.
"""

from __future__ import annotations

import warnings

import numpy as np


def map_axes(pos, ma) -> np.ndarray:
    """Transform local `(x, y)` coordinates into the `MAPAXES` frame.

    `pos` is an `n x 2` array of local coordinates and `ma` the six-element
    `MAPAXES` record. Returns the transformed coordinates; the input is not
    modified.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"pos must be n x 2, got {pos.shape}")
    ma = np.asarray(ma, dtype=float).ravel()
    if ma.size != 6:
        raise ValueError(f"MAPAXES must hold six values, got {ma.size}")

    origin = ma[2:4]
    e1 = ma[4:6] - origin
    e2 = ma[0:2] - origin
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("MAPAXES axis vectors must be non-degenerate")
    e1, e2 = e1 / n1, e2 / n2

    # z component of e1 x e2. Negative means the pair is left-handed, so the
    # transform mirrors the model rather than merely rotating it.
    msign = e1[0] * e2[1] - e1[1] * e2[0]
    if msign < 0.0:
        warnings.warn("mapAxes may change signature of coordinate system.",
                      RuntimeWarning, stacklevel=2)

    return pos[:, 0:1] * e1[None, :] + pos[:, 1:2] * e2[None, :] + origin[None, :]


__all__ = ["map_axes"]


if __name__ == "__main__":
    def test_identity_mapaxes_is_a_no_op():
        """Origin at 0, x along +x, y along +y: coordinates are unchanged."""
        ma = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # y-point, origin, x-point
        pos = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]])
        assert np.allclose(map_axes(pos, ma), pos)
        print("ok test_identity_mapaxes_is_a_no_op")

    def test_translation_moves_the_origin():
        ma = [100.0, 201.0, 100.0, 200.0, 101.0, 200.0]
        pos = np.array([[0.0, 0.0], [2.0, 3.0]])
        out = map_axes(pos, ma)
        assert np.allclose(out[0], [100.0, 200.0]), out[0]
        assert np.allclose(out[1], [102.0, 203.0]), out[1]
        print("ok test_translation_moves_the_origin")

    def test_rotation_preserves_distances():
        """A rotated frame is still orthonormal, so lengths survive it."""
        c, s = np.cos(0.7), np.sin(0.7)
        # x axis along (c, s); y axis along (-s, c), both from the origin.
        ma = [-s, c, 0.0, 0.0, c, s]
        pos = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, -2.0]])
        out = map_axes(pos, ma)
        assert np.allclose(np.linalg.norm(out, axis=1),
                           np.linalg.norm(pos, axis=1)), out
        print("ok test_rotation_preserves_distances")

    def test_left_handed_axes_warn():
        """A mirrored frame is accepted with a warning, as MRST does.

        Refusing would reject decks that load in MRST, so the behaviour has to
        match even though the result flips the model.
        """
        ma = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # y-point on +x, x-point on +y
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            map_axes(np.array([[1.0, 1.0]]), ma)
            assert any("signature" in str(x.message) for x in w), [str(x.message) for x in w]
        print("ok test_left_handed_axes_warn")

    def test_bad_input_is_rejected():
        for bad in ([1.0, 2.0, 3.0], np.zeros(7)):
            try:
                map_axes(np.zeros((2, 2)), bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {np.size(bad)} values")
        try:
            map_axes(np.zeros(4), [0, 1, 0, 0, 1, 0])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a 1-D pos array")
        print("ok test_bad_input_is_rejected")

    def test_input_is_not_modified():
        pos = np.array([[1.0, 2.0]])
        before = pos.copy()
        # Right-handed frame, so this test does not trip the mirroring warning.
        map_axes(pos, [5.0, 6.0, 5.0, 5.0, 6.0, 5.0])
        assert np.array_equal(pos, before), "input was mutated"
        print("ok test_input_is_not_modified")

    test_identity_mapaxes_is_a_no_op()
    test_translation_moves_the_origin()
    test_rotation_preserves_distances()
    test_left_handed_axes_warn()
    test_bad_input_is_rejected()
    test_input_is_not_modified()
    print("\nAll tests passed!")
