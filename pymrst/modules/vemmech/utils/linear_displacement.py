"""Analytic linear displacement fields (MRST `linearDisplacement`).

These are the fields a linear-elasticity discretization must reproduce
*exactly*: constant translations, the linear coordinate fields, and the rigid
rotations and stretches built from them. They are what the patch tests in
`squareTest` compare against, so an error here silently weakens every test
built on top.
"""

from typing import Callable, List, Tuple

import numpy as np

# MRST's 3-D permutations and signs, transcribed 0-based. `rigid` carries the
# sign pattern: with the signs it is a rotation, with `abs` a stretch.
_PERM_3D = np.array([[1, 0, 2],
                     [0, 2, 1],
                     [2, 1, 0]], dtype=np.int64)
_RIGID_3D = np.array([[1.0, -1.0, 0.0],
                      [0.0, 1.0, -1.0],
                      [1.0, 0.0, -1.0]])
_PLANE_NAMES = ("xy", "yz", "xz")


def _constant(dim: int, axis: int) -> Callable[[np.ndarray], np.ndarray]:
    const = np.zeros(dim)
    const[axis] = 1.0
    return lambda cc: np.tile(const, (np.asarray(cc).shape[0], 1))


def _linear(dim: int, axis: int) -> Callable[[np.ndarray], np.ndarray]:
    const = np.zeros(dim)
    const[axis] = 1.0
    return lambda cc: np.asarray(cc, dtype=float) * const


def _mixed(perm: np.ndarray, sign: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda cc: np.asarray(cc, dtype=float)[:, perm] * sign


def linear_displacement(
    dim: int,
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], List[str]]:
    """Return ``(fields, names)``: analytic displacement fields for ``dim``.

    Each field maps coordinates ``(n, dim)`` to displacements ``(n, dim)``.
    Names are MRST's verbatim, including its ``steatch``/``streatch``
    inconsistency between 2-D and 3-D, because they label test cases that a
    MATLAB reference dump identifies by name.
    """
    if dim not in (2, 3):
        raise ValueError("linear displacement fields are defined for dim 2 or 3")

    fields: List[Callable[[np.ndarray], np.ndarray]] = []
    names: List[str] = []

    for axis in range(dim):                              # constant translations
        fields.append(_constant(dim, axis))
        names.append(f"cons_disp_{axis + 1}")
    for axis in range(dim):                              # linear coordinate fields
        fields.append(_linear(dim, axis))
        names.append(f"lin_disp_{axis + 1}")

    if dim == 2:
        fields.append(_mixed(np.array([1, 0]), np.array([1.0, 1.0])))
        names.append("steatch_xy")                       # MRST's spelling
        fields.append(_mixed(np.array([1, 0]), np.array([1.0, -1.0])))
        names.append("rot_xy")
    else:
        for index, plane in enumerate(_PLANE_NAMES):
            perm, sign = _PERM_3D[index], _RIGID_3D[index]
            fields.append(_mixed(perm, sign))
            names.append(f"rot_{plane}")
            fields.append(_mixed(perm, np.abs(sign)))
            names.append(f"streatch_{plane}")

    return fields, names


linearDisplacement = linear_displacement


if __name__ == "__main__":

    def test_2d_field_names_match_mrst() -> None:
        _, names = linear_displacement(2)
        assert names == ["cons_disp_1", "cons_disp_2", "lin_disp_1",
                         "lin_disp_2", "steatch_xy", "rot_xy"], names
        print("ok test_2d_field_names_match_mrst")

    def test_3d_field_names_match_mrst() -> None:
        _, names = linear_displacement(3)
        assert names == ["cons_disp_1", "cons_disp_2", "cons_disp_3",
                         "lin_disp_1", "lin_disp_2", "lin_disp_3",
                         "rot_xy", "streatch_xy", "rot_yz", "streatch_yz",
                         "rot_xz", "streatch_xz"], names
        print("ok test_3d_field_names_match_mrst")

    def test_constant_fields_are_uniform() -> None:
        fields, names = linear_displacement(3)
        cc = np.random.default_rng(0).random((7, 3))
        for axis in range(3):
            u = fields[axis](cc)
            assert u.shape == (7, 3)
            assert np.allclose(u[:, axis], 1.0)
            assert np.allclose(np.delete(u, axis, axis=1), 0.0)
        print("ok test_constant_fields_are_uniform")

    def test_linear_fields_pick_out_one_coordinate() -> None:
        fields, _ = linear_displacement(3)
        cc = np.random.default_rng(1).random((5, 3))
        for axis in range(3):
            u = fields[3 + axis](cc)
            assert np.allclose(u[:, axis], cc[:, axis])
            assert np.allclose(np.delete(u, axis, axis=1), 0.0)
        print("ok test_linear_fields_pick_out_one_coordinate")

    def test_rotations_have_zero_symmetric_strain() -> None:
        """A rigid rotation must produce no strain: grad(u) + grad(u).T = 0."""
        fields, names = linear_displacement(3)
        basis = np.eye(3)
        for field, name in zip(fields, names):
            if not name.startswith("rot_"):
                continue
            gradient = np.column_stack([field(basis)[:, k] for k in range(3)])
            symmetric = gradient + gradient.T
            assert np.allclose(symmetric, 0.0), (name, symmetric)
        print("ok test_rotations_have_zero_symmetric_strain")

    def test_stretches_are_symmetric() -> None:
        fields, names = linear_displacement(3)
        basis = np.eye(3)
        for field, name in zip(fields, names):
            if not name.startswith("streatch_"):
                continue
            gradient = np.column_stack([field(basis)[:, k] for k in range(3)])
            assert np.allclose(gradient, gradient.T), (name, gradient)
        print("ok test_stretches_are_symmetric")

    def test_2d_rotation_has_zero_symmetric_strain() -> None:
        fields, names = linear_displacement(2)
        basis = np.eye(2)
        field = fields[names.index("rot_xy")]
        gradient = np.column_stack([field(basis)[:, k] for k in range(2)])
        assert np.allclose(gradient + gradient.T, 0.0), gradient
        print("ok test_2d_rotation_has_zero_symmetric_strain")

    def test_rejects_other_dimensions() -> None:
        for dim in (1, 4):
            try:
                linear_displacement(dim)
            except ValueError:
                continue
            raise AssertionError(f"dim {dim} should be rejected")
        print("ok test_rejects_other_dimensions")

    test_2d_field_names_match_mrst()
    test_3d_field_names_match_mrst()
    test_constant_fields_are_uniform()
    test_linear_fields_pick_out_one_coordinate()
    test_rotations_have_zero_symmetric_strain()
    test_stretches_are_symmetric()
    test_2d_rotation_has_zero_symmetric_strain()
    test_rejects_other_dimensions()
    print("\nAll tests passed!")
